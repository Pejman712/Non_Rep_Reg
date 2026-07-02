#!/usr/bin/env python3.10
"""
lio_gyro_gicp.py  —  Variant 5 (test): gyro handles ROTATION, GICP handles
                     TRANSLATION only.

Super-LIO front/back end, but the registration update is split:
  * rotation  comes purely from the IMU/gyro propagation (the ESKF prediction);
  * translation is corrected by GICP scan-to-submap.

Implementation: GICP gives an absolute pose (R_gicp, p_gicp).  We feed the iESKF
the *IMU-propagated* rotation as the rotation measurement (so the rotation
residual so3_log(Rᵀ·R_meas) is identically zero → the filter never corrects
rotation from lidar), while p_gicp is used as the translation measurement.  Net
effect: orientation rides on the gyro, position is anchored by GICP.

    ros2 run regnonrep lio_gyro_gicp.py --ros-args -p debug_csv:=/tmp/gyro_gicp.csv
"""

from lio_base import SuperLioBase, run_node
import numpy as np


class SuperLioGyroGICP(SuperLioBase):
    NODE_NAME = "super_lio_gyro_gicp"
    USE_NONREP = False    # no non-rep prediction; rotation is pure gyro
    USE_GICP = True
    VARIANT_DESC = "gyro rotation + GICP translation (test)"

    def _register(self):
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            self.last_method = "none"
            return

        # IMU-prior init guess (rotation from gyro propagation).
        T_init = np.eye(4)
        T_init[:3, :3] = self.kf.R
        T_init[:3, 3] = self.kf.p

        accepted, chi2, reg_conf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_gicp, p_meas, reg_conf, R_n = self._gicp_to_pose(scan_o3d, submap, T_init)
            # --- the split: keep the gyro's rotation, take GICP's translation
            R_meas = self.kf.R.copy()          # rotation residual -> 0 (gyro wins)
            accepted, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        self.last_method = "gyro+gicp_t" if accepted else "gicp_rej"
        self.last_conf = float(reg_conf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(accepted)


def main():
    run_node(SuperLioGyroGICP)


if __name__ == "__main__":
    main()
