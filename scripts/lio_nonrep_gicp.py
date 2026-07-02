#!/usr/bin/env python3.10
"""
lio_nonrep_gicp.py  —  Variant 2: Super-LIO + non-rep registration (GICP only).

Super-LIO's front/back end, but the point-to-plane Observe is replaced by:
  NonRepetitiveLiDARProcessor.predict_pose_adaptive (init guess)
  -> GICP scan-to-submap -> absolute pose -> iESKF update_pose (chi² gated).
NO point-to-plane fallback: if GICP can't run or is rejected, the pose simply
keeps the IMU-propagated prior for that scan.

    ros2 run regnonrep lio_nonrep_gicp.py --ros-args -p debug_csv:=/tmp/gicp.csv
"""

from lio_base import SuperLioBase, run_node


class SuperLioGICP(SuperLioBase):
    NODE_NAME = "super_lio_gicp"
    USE_NONREP = True
    USE_GICP = True
    VARIANT_DESC = "Super-LIO + non-rep/GICP (no P2P)"

    def _register(self):
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            self.last_method = "none"
            return

        feat, pred_pose, _, T_init = self._nonrep_init_guess(scan_o3d)

        accepted, chi2, reg_conf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_meas, p_meas, reg_conf, R_n = self._gicp_to_pose(scan_o3d, submap, T_init)
            accepted, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        self.last_method = "gicp" if accepted else "gicp_rej"
        self.last_conf = float(reg_conf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(accepted)
        self._feed_processor(feat, pred_pose, reg_conf)


def main():
    run_node(SuperLioGICP)


if __name__ == "__main__":
    main()
