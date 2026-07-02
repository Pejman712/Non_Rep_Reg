#!/usr/bin/env python3.10
"""
lio_degen_reuse.py  —  on degeneracy, REUSE the motion from the highest-
                       confidence recent registration (relative-motion model).

Front end is GICP scan-to-submap + P2P fallback.  The degeneracy branch:

  A fresh GICP on a degenerate scan is ill-conditioned and pulls the pose toward
  garbage.  An earlier attempt fed the non-rep processor's *absolute* prediction
  as the measurement — but a confidently-wrong prediction teleports the pose and
  diverges.  Instead we keep a short history of the per-scan world-frame motion
  (velocity + yaw-rate) of recently ACCEPTED registrations, each tagged with its
  confidence.  When a scan is degenerate we take the motion of the
  highest-confidence recent registration and carry the pose forward by it for
  this scan's dt — a "best recent motion" model rather than a teleport — applied
  as a soft, confidence-weighted measurement.

If there's no confident recent motion, it falls back to the IMU coast (skip).

    ros2 run regnonrep lio_degen_reuse.py --ros-args -p debug_csv:=/tmp/degen_reuse.csv
"""

import collections
import numpy as np

from lio_base import SuperLioBase, run_node


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SuperLioDegenReuse(SuperLioBase):
    NODE_NAME = "super_lio_degen_reuse"
    USE_NONREP = True
    USE_GICP = True
    VARIANT_DESC = "GICP + P2P, degeneracy -> reuse best recent motion"

    def __init__(self):
        super().__init__()
        self.reuse_min_conf = float(self.declare_parameter("reuse_min_conf", 0.40).value)
        self.reuse_noise_scale = float(self.declare_parameter("reuse_noise_scale", 4.0).value)
        self.reuse_max_age = float(self.declare_parameter("reuse_max_age", 1.5).value)  # s
        self.reuse_hist = int(self.declare_parameter("reuse_hist", 20).value)
        self._vel_hist = collections.deque(maxlen=self.reuse_hist)  # (v3, omega, conf, t)
        self._prev_p = None
        self._prev_yaw = None
        self._prev_t = None
        self._n_reuse = 0
        self.get_logger().info(
            f"  reuse_min_conf={self.reuse_min_conf} scale={self.reuse_noise_scale} "
            f"max_age={self.reuse_max_age}s")

    def _yaw(self):
        return float(np.arctan2(self.kf.R[1, 0], self.kf.R[0, 0]))

    def _reuse_noise(self, conf):
        k = self.reuse_noise_scale / max(float(conf), 0.1)
        R_n = np.zeros((6, 6))
        R_n[0:3, 0:3] = np.eye(3) * (self.meas_noise_rot * k) ** 2
        R_n[3:6, 3:6] = np.eye(3) * (self.meas_noise_pos * k) ** 2
        return R_n

    def _record_velocity(self, t, conf):
        """Store the world-frame motion of this accepted scan."""
        if self._prev_p is not None and self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 1e-3:
                v = (self.kf.p - self._prev_p) / dt
                om = _wrap(self._yaw() - self._prev_yaw) / dt
                self._vel_hist.append((v.copy(), float(om), float(conf), float(t)))

    def _update_prev(self, t):
        self._prev_p = self.kf.p.copy()
        self._prev_yaw = self._yaw()
        self._prev_t = float(t)

    def _reuse_motion(self, t):
        """Carry the pose by the highest-confidence recent motion.  Returns
        True if a measurement was applied."""
        if self._prev_p is None:
            return False
        cand = [h for h in self._vel_hist if (t - h[3]) < self.reuse_max_age]
        if not cand:
            return False
        v, om, conf, _ = max(cand, key=lambda h: h[2])
        if conf < self.reuse_min_conf:
            return False
        dt = t - self._prev_t
        if dt <= 0:
            return False
        p_meas = self._prev_p + v * dt
        if self.force_z_zero:
            p_meas[2] = self.kf.p[2]
        yaw_meas = self._prev_yaw + om * dt
        R_meas = self._yaw_to_R(yaw_meas)
        acc, chi2 = self.kf.update_pose(R_meas, p_meas,
                                        self._reuse_noise(conf), self.chi2_threshold)
        self.last_method = "degen_reuse" if acc else "degen_reuse_rej"
        self.last_conf = float(conf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(acc)
        if acc:
            self._n_reuse += 1
        return acc

    def _register(self):
        t = float(self.kf.current_time)
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            ncorr, rms = self._observe_point_to_plane()
            self.last_method = "p2p"
            self.last_conf = self._p2p_conf(ncorr, rms)
            self.last_chi2 = 0.0
            self.last_accepted = 1 if ncorr > 0 else 0
            self._update_prev(t)
            return

        feat, pred_pose, pred_conf, T_init = self._nonrep_init_guess(scan_o3d)

        # ---- degeneracy: reuse best recent motion ---------------------------
        if self.degen_enable and self._scan_degeneracy(scan_pts):
            if not self._degen_active:
                self._degen_active = True
                self.get_logger().warn(
                    f"Scan {self._scan_counter}: degenerate geometry — "
                    f"reusing best recent motion")
            self._n_degen_skip += 1
            if not self._reuse_motion(t):
                self._skip_map = True                # no usable motion -> coast
                self.last_method = "degen_skip"
                self.last_conf = 0.0
                self.last_chi2 = 0.0
                self.last_accepted = 0
            self._feed_processor(feat, pred_pose, self.last_conf)
            self._update_prev(t)
            return
        if self._degen_active:
            self._degen_active = False
            self.get_logger().info(
                f"Scan {self._scan_counter}: geometry recovered — resuming GICP")

        # ---- normal path: GICP + P2P fallback ------------------------------
        accepted, chi2, reg_conf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_meas, p_meas, reg_conf, R_n = self._gicp_to_pose(scan_o3d, submap, T_init)
            accepted, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        self.last_method = "gicp" if accepted else "p2p"
        if not accepted:
            self._n_p2p_fallback += 1
            ncorr, rms = self._observe_point_to_plane()
            reg_conf = self._p2p_conf(ncorr, rms)

        self.last_conf = float(reg_conf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(accepted)
        self._record_velocity(t, reg_conf if accepted else 0.0)
        self._feed_processor(feat, pred_pose, reg_conf)
        self._update_prev(t)


def main():
    run_node(SuperLioDegenReuse)


if __name__ == "__main__":
    main()
