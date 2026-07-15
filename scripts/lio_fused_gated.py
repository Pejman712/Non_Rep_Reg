#!/usr/bin/env python3.10
"""
lio_fused_gated.py  —  Variant 8: GICP + GATED, down-weighted P2P + degen.

The plain fused variant always applied P2P as a second measurement, which
over-corrected and blew up at the end.  Here the P2P fusion is:
  * GATED   — applied as a refinement ONLY when the GICP update was accepted;
  * down-weighted (p2p_fuse_weight) so it sharpens rather than fights GICP.
If GICP is rejected, P2P runs at full weight as the usual fallback.  The
degeneracy guard is kept.

    ros2 run regnonrep lio_fused_gated.py --ros-args \
        -p debug_csv:=/tmp/fused_gated.csv -p p2p_fuse_weight:=150.0
"""

from lio_base import SuperLioBase, run_node


class SuperLioFusedGated(SuperLioBase):
    NODE_NAME = "super_lio_fused_gated"
    USE_NONREP = True
    USE_GICP = True
    VARIANT_DESC = "GICP + gated down-weighted P2P + degen"

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

        g_acc, chi2, gconf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_meas, p_meas, gconf, R_n = self._gicp_to_pose(scan_o3d, submap, T_init)
            g_acc, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        if g_acc:
            # GICP anchored the pose; refine with a GENTLE (down-weighted) P2P.
            ncorr, rms = self._observe_point_to_plane(weight=self.p2p_fuse_weight)
            pconf = self._p2p_conf(ncorr, rms)
            self.last_method = "gicp+p2p" if ncorr > 0 else "gicp"
            self.last_conf = max(gconf, pconf)
        else:
            # GICP rejected/unavailable — full-weight P2P fallback.
            self._n_p2p_fallback += 1
            ncorr, rms = self._observe_point_to_plane()
            self.last_method = "p2p"
            self.last_conf = self._p2p_conf(ncorr, rms)

        self.last_chi2 = float(chi2)
        self.last_accepted = int(g_acc or ncorr > 0)
        self._feed_processor(feat, pred_pose, self.last_conf)


def main():
    run_node(SuperLioFusedGated)


if __name__ == "__main__":
    main()
