#!/usr/bin/env python3.10
"""
lio_fused_gated_ndt.py  —  fused-gated, but NDT instead of GICP.

Same structure as lio_fused_gated.py (NDT anchors the pose, then a GATED,
down-weighted point-to-plane refines it only when the NDT update was accepted;
full-weight P2P fallback otherwise; degeneracy guard kept) — except the primary
scan-to-submap registration is NDT (SuperLioBase._ndt) rather than small_gicp.

    ros2 run regnonrep lio_fused_gated_ndt.py --ros-args \
        -p ndt_resolution:=1.0 -p p2p_fuse_weight:=150.0
"""

from lio_base import SuperLioBase, run_node


class SuperLioFusedGatedNDT(SuperLioBase):
    NODE_NAME = "super_lio_fused_gated_ndt"
    USE_NONREP = True
    USE_GICP = True
    VARIANT_DESC = "NDT + gated down-weighted P2P + degen"

    def __init__(self):
        super().__init__()
        self.p2p_fuse_weight = float(
            self.declare_parameter("p2p_fuse_weight", 150.0).value)
        self.get_logger().info(f"  p2p_fuse_weight={self.p2p_fuse_weight}")

    def _register(self):
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            ncorr, rms = self._observe_point_to_plane()
            self.last_method = "p2p"
            self.last_conf = self._p2p_conf(ncorr, rms)
            self.last_chi2 = 0.0
            self.last_accepted = 1 if ncorr > 0 else 0
            return

        if self.degen_enable and self._scan_degeneracy(scan_pts):
            if not self._degen_active:
                self._degen_active = True
                self.get_logger().warn(
                    f"Scan {self._scan_counter}: degenerate geometry — "
                    f"skipping registration & map update")
            self._n_degen_skip += 1
            self._skip_map = True
            self.last_method = "skip_degen"
            self.last_conf = 0.0
            self.last_chi2 = 0.0
            self.last_accepted = 0
            self._feed_processor(None, None, 0.0)
            return
        if self._degen_active:
            self._degen_active = False
            self.get_logger().info(
                f"Scan {self._scan_counter}: geometry recovered — resuming")

        feat, pred_pose, _, T_init = self._nonrep_init_guess(scan_o3d)

        n_acc, chi2, nconf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_meas, p_meas, nconf, R_n = self._ndt_to_pose(scan_o3d, submap, T_init)
            n_acc, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        if n_acc:
            # NDT anchored the pose; refine with a GENTLE (down-weighted) P2P.
            ncorr, rms = self._observe_point_to_plane(weight=self.p2p_fuse_weight)
            pconf = self._p2p_conf(ncorr, rms)
            self.last_method = "ndt+p2p" if ncorr > 0 else "ndt"
            self.last_conf = max(nconf, pconf)
        else:
            # NDT rejected/unavailable — full-weight P2P fallback.
            self._n_p2p_fallback += 1
            ncorr, rms = self._observe_point_to_plane()
            self.last_method = "p2p"
            self.last_conf = self._p2p_conf(ncorr, rms)

        self.last_chi2 = float(chi2)
        self.last_accepted = int(n_acc or ncorr > 0)
        self._feed_processor(feat, pred_pose, self.last_conf)


def main():
    run_node(SuperLioFusedGatedNDT)


if __name__ == "__main__":
    main()
