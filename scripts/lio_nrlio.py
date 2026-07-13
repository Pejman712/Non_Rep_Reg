#!/usr/bin/env python3.10
"""
lio_nrlio.py — variant "nrlio": a mapping estimator purpose-built for the
narrow / low-FoV *non-repetitive* Livox sensors (Avia 70x70, Horizon 80x27,
Mid-360 360x59).

On top of gen_lio it adds, in pipeline order:

  (1) Motion-adaptive, coverage-targeted accumulation  [FRONT-END].
      The Livox rosette fills the FoV over integration time (measured knee
      ~200-300 ms Avia / ~200 ms Horizon).  nrlio accumulates deskewed scans up
      to a sensor-specific knee `nr_knee_ms`, BUT shrinks the window the instant
      the gyro rate is high (fast turns smear the accumulated cloud).  Buffered
      scans are motion-compensated into the current frame.

  (2) Per-point continuous-time deskew (inherited from lio_base).

  (3) Nonrep-seeded scale-switched hybrid GICP/P2P (inherited from gen_lio).

  (4) Scale-aware adaptive voxel budget, kept finer indoors (inherited).

  (5) Degeneracy -> GICP routing with the map frozen (inherited from gen_lio).

  (6) ZUPT stationarity gate.  When the platform is still the IMU keeps
      integrating noise and registration jitters against the map, so the pose
      drifts even though nothing moved.  Detect "no motion" from the IMU (low
      gyro + low accel-magnitude variation) and, once it persists, HOLD the last
      pose, zero the velocity, and freeze the map instead of registering.

nrlio writes the evaluated TUM itself (the launch does not start odom_to_tum for
it), streaming each scan's pose live.

Note: an earlier keyframe loop-closure pose-graph back-end was removed — it
degraded accuracy and performance on these indoor sequences.

    ros2 run regnonrep lio_nrlio.py --ros-args -p nr_knee_ms:=300.0
    # Horizon / Mid-360: -p nr_knee_ms:=200.0
"""
from collections import deque

import numpy as np

from lio_base import run_node, voxel_downsample, rot_to_quat
from lio_gen_lio import SuperLioGen

# nrlio owns the evaluated TUM (the launch skips odom_to_tum for any nrlio exe).
TUM_PATH = "/u/97/habibip1/unix/ros2_ws/src/regnonrep/tum/lio_odom.tum"


class SuperLioNRLIO(SuperLioGen):
    NODE_NAME = "super_lio_nrlio"
    VARIANT_DESC = ("NR-LIO: nonrep hybrid + gyro-adaptive coverage accumulation "
                    "+ ZUPT stationarity gate")

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter

        # ---- (1) gyro-adaptive, coverage-targeted accumulation --------------
        self.nr_knee_ms = float(gp("nr_knee_ms", 300.0).value)   # sensor coverage knee
        self.nr_scan_ms = float(gp("nr_scan_ms", 100.0).value)   # nominal scan period
        self.nr_accum_max = max(1, int(round(self.nr_knee_ms / max(self.nr_scan_ms, 1.0))))
        self.nr_accum_voxel = float(gp("nr_accum_voxel", 0.12).value)
        # gyro gating [rad/s]: >=hi -> single scan; <=lo -> full knee; linear between
        self.nr_gyro_hi = float(gp("nr_gyro_hi", 1.0).value)     # ~57 deg/s
        self.nr_gyro_lo = float(gp("nr_gyro_lo", 0.15).value)    # ~9 deg/s
        # keep the adaptive voxel finer / denser indoors (narrow FoV small rooms)
        self.gen_d_max = min(self.gen_d_max, float(gp("nr_voxel_max", 0.5).value))
        self.gen_n_min = max(self.gen_n_min, int(gp("nr_npts_min", 1200).value))
        self._acc = deque(maxlen=max(0, self.nr_accum_max - 1))  # (pts_body, R, p)
        self._nr_cur = None
        self._nr_frames_hist = []       # DIAG: accumulated-frame count per scan
        self.tum_out = str(gp("nr_tum_out", TUM_PATH).value)

        # ---- (6) stationarity gate (ZUPT + pose hold) -----------------------
        self.zupt_enable = bool(gp("zupt_enable", True).value)
        self.zupt_gyro_thresh = float(gp("zupt_gyro_thresh", 0.015).value)  # rad/s (~0.9°/s)
        # coefficient of variation of |accel| — unit-independent (Avia g / Xsens m/s²)
        self.zupt_acc_cv = float(gp("zupt_acc_cv", 0.012).value)
        self.zupt_min_count = int(gp("zupt_min_count", 3).value)   # debounce (scans)
        self._still_count = 0
        self._zupt_anchor = None            # (R, p) held pose
        self._n_zupt = 0
        self._last_pose_R = np.eye(3)
        self._last_pose_p = np.zeros(3)

        # open the evaluated TUM for live streaming
        try:
            self._tum_fh = open(self.tum_out, "w")
        except OSError as e:                                      # noqa: BLE001
            self.get_logger().error(f"nrlio: cannot open TUM {self.tum_out}: {e}")
            self._tum_fh = None

        # per-pose annotation sidecar (which mechanism fired each scan) for the
        # annotated route plot: aligned 1:1 with the TUM rows.
        self._nr_last_frames = 1
        self.ann_out = (self.tum_out[:-4] + ".ann.csv"
                        if self.tum_out.endswith(".tum") else self.tum_out + ".ann.csv")
        try:
            self._ann_fh = open(self.ann_out, "w")
            self._ann_fh.write("stamp,x,y,z,method,accum,voxel_d,proc_ms\n")
        except OSError:                                          # noqa: BLE001
            self._ann_fh = None

        self.get_logger().info(
            f"  nrlio: accum<=({self.nr_accum_max}f @ knee {self.nr_knee_ms:.0f}ms) "
            f"gyro[{self.nr_gyro_lo},{self.nr_gyro_hi}]rad/s accum_voxel={self.nr_accum_voxel} "
            f"| ZUPT={'on' if self.zupt_enable else 'off'} "
            f"(gyro<{self.zupt_gyro_thresh} acc_cv<{self.zupt_acc_cv} n>={self.zupt_min_count}) "
            f"-> {self.tum_out}")

    # ---- stationarity detection (IMU-only, available pre-registration) -------
    def _is_stationary(self):
        imu = getattr(self, "_meas_imu", [])
        if not imu:
            return False
        gyr = np.array([g for (_, g, _) in imu], dtype=float)
        acc = np.array([a for (a, _, _) in imu], dtype=float)
        w = float(np.linalg.norm(gyr - self.kf.bg, axis=1).mean())
        amag = np.linalg.norm(acc, axis=1)
        m = float(amag.mean())
        acc_cv = float(amag.std() / m) if m > 1e-6 else 1.0
        return (w < self.zupt_gyro_thresh) and (acc_cv < self.zupt_acc_cv)

    # ---- (1) how many frames to accumulate for THIS scan --------------------
    def _nr_target_frames(self):
        """Gyro-gated accumulation depth (frames incl. current), targeting the
        coverage knee when slow and collapsing to a single scan when turning."""
        gyrs = [g for (_, g, _) in getattr(self, "_meas_imu", [])]
        if not gyrs:
            return self.nr_accum_max
        w = float(np.mean([np.linalg.norm(np.asarray(g) - self.kf.bg) for g in gyrs]))
        if w >= self.nr_gyro_hi:
            return 1
        if w <= self.nr_gyro_lo:
            return self.nr_accum_max
        frac = (self.nr_gyro_hi - w) / (self.nr_gyro_hi - self.nr_gyro_lo)
        return max(1, int(round(1 + frac * (self.nr_accum_max - 1))))

    def _gen_preprocess(self):
        super()._gen_preprocess()
        cur = self._scan_undistort
        if cur is None or cur.shape[0] == 0:
            self._nr_cur = None
            return
        self._nr_cur = voxel_downsample(cur, self.nr_accum_voxel)
        n_use = min(len(self._acc), self._nr_target_frames() - 1)
        self._nr_frames_hist.append(n_use + 1)
        self._nr_last_frames = n_use + 1
        if n_use <= 0:
            return
        # motion-compensate the most-recent n_use buffered scans into the current
        # (propagated-prior) frame:  world = R_i·p_i + t_i ; body_now = R0ᵀ(world−t0)
        R0, p0 = self.kf.R, self.kf.p
        merged = [cur]
        for pts_i, R_i, p_i in list(self._acc)[-n_use:]:
            world = pts_i @ R_i.T + p_i
            merged.append((world - p0) @ R0)
        self._scan_undistort = voxel_downsample(np.vstack(merged), self.nr_accum_voxel)

    # ---- registration with a stationarity gate in front ---------------------
    def _register(self):
        if self.zupt_enable and self._is_stationary():
            self._still_count += 1
            if self._still_count >= self.zupt_min_count:
                # engage hold: snap to the anchor captured at ZUPT onset, zero
                # velocity (arrest drift), freeze the map, skip registration.
                if self._zupt_anchor is None:
                    self._zupt_anchor = (self._last_pose_R.copy(),
                                         self._last_pose_p.copy())
                Ra, pa = self._zupt_anchor
                self.kf.R = Ra.copy()
                self.kf.p = pa.copy()
                self.kf.v = np.zeros(3)
                self._skip_map = True
                self._nr_cur = None            # don't buffer duplicate scans
                if not hasattr(self, "_pts_body"):
                    self._pts_body = np.zeros((0, 3))
                self.last_method = "stationary(zupt)"
                self.last_conf = 1.0
                self.last_chi2 = 0.0
                self.last_accepted = 1
                self._n_zupt += 1
                return
        else:
            self._still_count = 0
            self._zupt_anchor = None
        super()._register()

    # ---- output: stream the TUM live + buffer for accumulation --------------
    def _output(self):
        super()._output()
        R = self.kf.R.copy()
        p = self.kf.p.copy()
        stamp = float(self.kf.current_time)
        # remember this scan's final pose — the anchor a following ZUPT holds to
        self._last_pose_R = R.copy()
        self._last_pose_p = p.copy()
        if self._tum_fh is not None:
            qx, qy, qz, qw = rot_to_quat(R)
            self._tum_fh.write(
                f"{stamp:.18e} {p[0]:.18e} {p[1]:.18e} {p[2]:.18e} "
                f"{qx:.18e} {qy:.18e} {qz:.18e} {qw:.18e}\n")
            self._tum_fh.flush()
        if self._ann_fh is not None:
            meth = self.last_method.replace(",", ";")
            self._ann_fh.write(
                f"{stamp:.6f},{p[0]:.5f},{p[1]:.5f},{p[2]:.5f},{meth},"
                f"{getattr(self, '_nr_last_frames', 1)},{getattr(self, '_gen_d', 0.0):.3f},"
                f"{getattr(self, 'last_proc_ms', 0.0):.2f}\n")
            self._ann_fh.flush()
        # buffer this scan (with its corrected end-of-scan pose) for accumulation
        if self._nr_cur is not None and self._acc.maxlen:
            self._acc.append((self._nr_cur, R, p))

    def shutdown(self):
        super().shutdown()
        if self._nr_frames_hist:
            h = np.asarray(self._nr_frames_hist)
            self.get_logger().info(
                f"[nrlio] accumulation depth: mean={h.mean():.2f} "
                f"median={int(np.median(h))} max={int(h.max())} (knee={self.nr_accum_max}) "
                f"single-scan(turns)={int((h == 1).sum())}/{h.size} scans")
        self.get_logger().info(
            f"[nrlio] ZUPT held {self._n_zupt} stationary scans (pose frozen, "
            f"velocity zeroed, map frozen)")
        if self._tum_fh is not None:
            self._tum_fh.close()
        if self._ann_fh is not None:
            self._ann_fh.close()


def main():
    run_node(SuperLioNRLIO)


if __name__ == "__main__":
    main()
