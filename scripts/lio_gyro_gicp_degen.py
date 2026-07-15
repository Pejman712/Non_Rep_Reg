#!/usr/bin/env python3.10
"""
lio_gyro_gicp_degen.py  —  Variant 6: gyro rotation + GICP translation + degen.

Same split as lio_gyro_gicp (rotation from gyro, translation from GICP) but with
the degeneracy guard added: degenerate scans (concentrated / planar — likely the
cause of the late-run failure) are skipped (IMU-only, map paused) instead of
feeding a bad translation correction.

    ros2 run regnonrep lio_gyro_gicp_degen.py --ros-args -p debug_csv:=/tmp/gyro_gicp_degen.csv
"""

from lio_base import SuperLioBase, run_node
import numpy as np


class SuperLioGyroGICPDegen(SuperLioBase):
    NODE_NAME = "super_lio_gyro_gicp_degen"
    USE_NONREP = False
    USE_GICP = True
    VARIANT_DESC = "gyro rotation + GICP translation + degen guard"

    def _register(self):
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            self.last_method = "none"
            return

        # degeneracy guard: skip translation correction + map on bad geometry
        if self.degen_enable and self._scan_degeneracy(scan_pts):
            if not self._degen_active:
                self._degen_active = True
                self.get_logger().warn(
                    f"Scan {self._scan_counter}: degenerate geometry — "
                    f"skipping translation update & map (gyro keeps rotation)")
            self._n_degen_skip += 1
            self._skip_map = True
            self.last_method = "skip_degen"
            self.last_conf = 0.0
            self.last_chi2 = 0.0
            self.last_accepted = 0
            return
        if self._degen_active:
            self._degen_active = False
            self.get_logger().info(
                f"Scan {self._scan_counter}: geometry recovered — resuming")

        T_init = np.eye(4)
        T_init[:3, :3] = self.kf.R
        T_init[:3, 3] = self.kf.p

        accepted, chi2, reg_conf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_gicp, p_meas, reg_conf, R_n = self._gicp_to_pose(scan_o3d, submap, T_init)
            R_meas = self.kf.R.copy()          # gyro rotation
            accepted, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        self.last_method = "gyro+gicp_t+degen" if accepted else "gicp_rej"
        self.last_conf = float(reg_conf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(accepted)


def main():
    run_node(SuperLioGyroGICPDegen)


if __name__ == "__main__":
    main()
