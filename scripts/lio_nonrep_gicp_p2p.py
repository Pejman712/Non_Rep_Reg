#!/usr/bin/env python3.10
"""
lio_nonrep_gicp_p2p.py  —  Variant 3: Super-LIO + non-rep/GICP + P2P fallback.

Same as variant 2, but when GICP cannot run (map not ready) or its update is
rejected by the chi² gate, fall back to Super-LIO's point-to-plane Observe so the
scan still contributes a correction.  (This is the v3 pipeline WITHOUT the
degeneracy guard — that lives in variant 4.)

    ros2 run regnonrep lio_nonrep_gicp_p2p.py --ros-args -p debug_csv:=/tmp/gicp_p2p.csv
"""

import numpy as np
from lio_base import SuperLioBase, run_node, so3_log


class SuperLioGICPP2P(SuperLioBase):
    NODE_NAME = "super_lio_gicp_p2p"
    USE_NONREP = True
    USE_GICP = True
    VARIANT_DESC = "Super-LIO + non-rep/GICP + P2P fallback"

    def _register(self):
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
    run_node(SuperLioGICPP2P)


if __name__ == "__main__":
    main()
