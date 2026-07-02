#!/usr/bin/env python3.10
"""
lio_nonrep_gicp_p2p_degen.py  —  Super-LIO + non-rep/GICP + P2P fallback with an
information-matrix degeneracy guard that falls back to gyro/IMU-only motion.

This is lio_nonrep_gicp_p2p.py (GICP scan-to-submap, with point-to-plane as the
fallback) plus the degeneracy detector from degeneracy.py:

  * Every scan is tested with InfoDegeneracyDetector (the same two
    information-matrix methods — min-eigenvalue + condition number — that the
    standalone degeneracy.py node uses).
  * When the scan is declared DEGENERATE (e.g. staring at one flat wall, a
    feature-poor corridor / tunnel where translation along the wall is
    unobservable), NO LiDAR correction is applied: GICP and the point-to-plane
    fallback are both skipped, so the pose is carried forward by the gyro/IMU
    propagation alone (dead-reckoning).  Map insertion is paused so the bad
    geometry can't corrupt the map, and resumes automatically once geometry
    recovers.
  * Otherwise the pipeline behaves exactly like variant 3 (GICP, P2P fallback).

The rationale: when the scene is geometrically degenerate the LiDAR cannot
constrain (some of) the 6 DoF, and forcing a registration there pulls the
estimate toward a wrong solution.  The IMU/gyro propagation is the more reliable
short-term motion source while geometry is degenerate.

    ros2 run regnonrep lio_nonrep_gicp_p2p_degen.py --ros-args \
        -p debug_csv:=/tmp/gicp_p2p_degen.csv
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lio_base import SuperLioBase, run_node, so3_log
from degeneracy import InfoDegeneracyDetector


class SuperLioGICPP2PDegen(SuperLioBase):
    NODE_NAME = "super_lio_gicp_p2p_degen"
    USE_NONREP = True
    USE_GICP = True
    VARIANT_DESC = "non-rep/GICP + P2P, info-matrix degeneracy -> gyro/IMU-only"

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        # ---- info-matrix degeneracy detector (degeneracy.py) ---------------
        self._degen = InfoDegeneracyDetector(
            use_eig=bool(gp("degen_use_eig", True).value),
            use_cond=bool(gp("degen_use_cond", True).value),
            consensus=int(gp("degen_consensus", 1).value),
            trans_eig_thresh=float(gp("degen_trans_eig_thresh", 0.04).value),
            rot_eig_ratio=float(gp("degen_rot_eig_ratio", 0.002).value),
            trans_cond_thresh=float(gp("degen_trans_cond_thresh", 25.0).value),
            rot_cond_thresh=float(gp("degen_rot_cond_thresh", 500.0).value),
            voxel_size=float(gp("degen_voxel_size", 0.2).value),
            normal_radius=float(gp("degen_normal_radius", 0.5).value),
            normal_max_nn=int(gp("degen_normal_max_nn", 30).value),
            min_points=int(gp("degen_info_min_points", 50).value))
        self.get_logger().info(
            f"  info-degeneracy: eig={self._degen.use_eig} "
            f"cond={self._degen.use_cond} consensus={self._degen.consensus} "
            f"-> gyro/IMU-only on degeneracy (enable={self.degen_enable})")

    def _register(self):
        # --- degeneracy guard (degeneracy.py info-matrix methods) -----------
        # Tested on the deskewed body scan; the detector does its own
        # voxel-downsample + normal estimation, exactly like the standalone node.
        if self.degen_enable:
            is_degen, _votes, details = self._degen.detect_from_xyz(self._scan_undistort)
            if is_degen:
                if not self._degen_active:
                    self._degen_active = True
                    self.get_logger().warn(
                        f"Scan {self._scan_counter}: DEGENERACY DETECTED "
                        f"({' | '.join(details)}) — skipping GICP & P2P, carrying "
                        f"pose with gyro/IMU motion only; pausing map until recovery")
                self._n_degen_skip += 1
                self._skip_map = True            # don't pollute the map
                self.last_method = "skip_degen_gyro"
                self.last_conf = 0.0
                self.last_chi2 = 0.0
                self.last_accepted = 0
                self._feed_processor(None, None, 0.0)
                return                            # IMU/gyro propagation carries it
            if self._degen_active:
                self._degen_active = False
                self.get_logger().info(
                    f"Scan {self._scan_counter}: geometry recovered — resuming "
                    f"GICP/P2P registration & mapping")

        # --- non-degenerate: identical to lio_nonrep_gicp_p2p ---------------
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            # No usable scan cloud for GICP — still try P2P on the body points.
            ncorr, rms = self._observe_point_to_plane()
            self.last_method = "p2p"
            self.last_conf = self._p2p_conf(ncorr, rms)
            self.last_chi2 = 0.0
            self.last_accepted = 1 if ncorr > 0 else 0
            return

        feat, pred_pose, _, T_init = self._nonrep_init_guess(scan_o3d)
        R_imu, p_imu = self.kf.R.copy(), self.kf.p.copy()

        accepted, chi2, reg_conf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_meas, p_meas, reg_conf, R_n = self._gicp_to_pose(scan_o3d, submap, T_init)
            accepted, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        self.last_method = "gicp" if accepted else "p2p"
        if not accepted:
            self._n_p2p_fallback += 1
            if submap is not None:   # GICP ran but was rejected
                dp = float(np.linalg.norm(p_meas - p_imu))
                dth = float(np.degrees(np.linalg.norm(so3_log(R_imu.T @ R_meas))))
                self.get_logger().warn(
                    f"Scan {self._scan_counter}: GICP rejected "
                    f"(chi2={chi2:.1f} > {self.chi2_threshold:.1f}, conf={reg_conf:.2f}, "
                    f"Δp={dp:.3f}m Δrot={dth:.2f}°) — falling back to point-to-plane")
            ncorr, rms = self._observe_point_to_plane()
            reg_conf = self._p2p_conf(ncorr, rms)

        self.last_conf = float(reg_conf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(accepted)
        self._feed_processor(feat, pred_pose, reg_conf)


def main():
    run_node(SuperLioGICPP2PDegen)


if __name__ == "__main__":
    main()
