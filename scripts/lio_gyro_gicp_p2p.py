#!/usr/bin/env python3.10
"""
lio_gyro_gicp_p2p.py  —  Variant 9: gyro rotation + GICP & P2P translation.

Rotation stays on the gyro; translation is refined by BOTH registrations:
  * GICP scan-to-submap gives the absolute translation (rotation fed back as the
    gyro's, so no lidar rotation correction), then
  * a TRANSLATION-ONLY point-to-plane update (rotation Jacobian zeroed) sharpens
    the position against the local map.
Two complementary translation sources, rotation untouched by lidar.

    ros2 run regnonrep lio_gyro_gicp_p2p.py --ros-args -p debug_csv:=/tmp/gyro_gicp_p2p.csv
"""

from lio_base import SuperLioBase, run_node
import numpy as np


class SuperLioGyroGICPP2P(SuperLioBase):
    NODE_NAME = "super_lio_gyro_gicp_p2p"
    USE_NONREP = False
    USE_GICP = True
    VARIANT_DESC = "gyro rotation + GICP & P2P translation"

    def _register(self):
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            ncorr, rms = self._observe_point_to_plane(trans_only=True)
            self.last_method = "p2p_t"
            self.last_conf = self._p2p_conf(ncorr, rms)
            self.last_chi2 = 0.0
            self.last_accepted = 1 if ncorr > 0 else 0
            return

        T_init = np.eye(4)
        T_init[:3, :3] = self.kf.R
        T_init[:3, 3] = self.kf.p

        accepted, chi2, reg_conf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_gicp, p_meas, reg_conf, R_n = self._gicp_to_pose(scan_o3d, submap, T_init)
            R_meas = self.kf.R.copy()          # gyro rotation (no lidar rotation)
            accepted, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        # translation-only P2P refinement (rotation left to gyro)
        ncorr, rms = self._observe_point_to_plane(trans_only=True)
        pconf = self._p2p_conf(ncorr, rms)

        if accepted and ncorr > 0:
            self.last_method = "gyro+gicp+p2p_t"
        elif accepted:
            self.last_method = "gyro+gicp_t"
        elif ncorr > 0:
            self.last_method = "p2p_t"
        else:
            self.last_method = "none"
        self.last_conf = max(reg_conf, pconf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(accepted or ncorr > 0)


def main():
    run_node(SuperLioGyroGICPP2P)


if __name__ == "__main__":
    main()
