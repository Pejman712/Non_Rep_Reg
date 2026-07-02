#!/usr/bin/env python3.10
"""
lio_nonrep_ndt.py  —  non-rep + NDT registration (NDT instead of small_gicp).

Identical pipeline to lio_nonrep_gicp.py, but the scan-to-submap registration
uses the Normal Distributions Transform (point-to-distribution, vectorised
Gauss-Newton on SE(3) — see SuperLioBase._ndt) instead of small_gicp GICP.

    ros2 run regnonrep lio_nonrep_ndt.py --ros-args -p ndt_resolution:=1.0
"""

from lio_base import SuperLioBase, run_node


class SuperLioNDT(SuperLioBase):
    NODE_NAME = "super_lio_ndt"
    USE_NONREP = True
    USE_GICP = True       # builds submap/processor machinery; registration via NDT
    VARIANT_DESC = "Super-LIO + non-rep/NDT (no P2P)"

    def _register(self):
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            self.last_method = "none"
            return

        feat, pred_pose, _, T_init = self._nonrep_init_guess(scan_o3d)

        accepted, chi2, reg_conf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_meas, p_meas, reg_conf, R_n = self._ndt_to_pose(scan_o3d, submap, T_init)
            accepted, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        self.last_method = "ndt" if accepted else "ndt_rej"
        self.last_conf = float(reg_conf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(accepted)
        self._feed_processor(feat, pred_pose, reg_conf)


def main():
    run_node(SuperLioNDT)


if __name__ == "__main__":
    main()
