#!/usr/bin/env python3.10
"""
lio_fused_gated_p2p_gicp_ndt.py  —  fused-gated cascade of GICP + NDT + P2P.

Combines all three registration measurements per scan, each gated through the
ESKF chi² test so a bad one can't corrupt the state:
  1. GICP (small_gicp) scan→submap  → primary absolute-pose update.
  2. NDT (point-to-distribution)     → refinement update, seeded from the
                                       post-GICP pose (or the nonrep guess if
                                       GICP was rejected).
  3. Point-to-plane                  → GATED, down-weighted refinement when GICP
                                       or NDT anchored the pose; full-weight
                                       fallback when both were rejected.
Degeneracy guard kept (skips registration + map update on ill-conditioned scans).

    ros2 run regnonrep lio_fused_gated_p2p_gicp_ndt.py --ros-args \
        -p ndt_resolution:=1.0 -p p2p_fuse_weight:=150.0
"""

import numpy as np

from lio_base import SuperLioBase, run_node


class SuperLioFusedGatedP2PGicpNDT(SuperLioBase):
    NODE_NAME = "super_lio_fused_p2p_gicp_ndt"
    USE_NONREP = True
    USE_GICP = True
    VARIANT_DESC = "GICP + NDT + gated down-weighted P2P + degen"

    def __init__(self):
        super().__init__()
        self.p2p_fuse_weight = float(
            self.declare_parameter("p2p_fuse_weight", 150.0).value)
        self.get_logger().info(f"  p2p_fuse_weight={self.p2p_fuse_weight}")

    def _pose_T(self):
        T = np.eye(4)
        T[:3, :3] = self.kf.R.copy()
        T[:3, 3] = self.kf.p.copy()
        return T

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

        g_acc, n_acc, chi2, conf, methods = False, False, 0.0, 0.0, []
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            # 1) GICP — primary anchor
            Rg, pg, gconf, Rng = self._gicp_to_pose(scan_o3d, submap, T_init)
            g_acc, chi2 = self.kf.update_pose(Rg, pg, Rng, self.chi2_threshold)
            if g_acc:
                methods.append("gicp")
                conf = max(conf, gconf)
            # 2) NDT — refinement (seed from post-GICP pose, else nonrep guess)
            ndt_init = self._pose_T() if g_acc else T_init
            Rn, pn, nconf, Rnn = self._ndt_to_pose(scan_o3d, submap, ndt_init)
            n_acc, nchi2 = self.kf.update_pose(Rn, pn, Rnn, self.chi2_threshold)
            if n_acc:
                methods.append("ndt")
                conf = max(conf, nconf)
                chi2 = nchi2

        # 3) Point-to-plane — gated refinement if anchored, else full fallback
        if g_acc or n_acc:
            ncorr, rms = self._observe_point_to_plane(weight=self.p2p_fuse_weight)
            if ncorr > 0:
                methods.append("p2p")
                conf = max(conf, self._p2p_conf(ncorr, rms))
        else:
            self._n_p2p_fallback += 1
            ncorr, rms = self._observe_point_to_plane()
            methods = ["p2p"]
            conf = self._p2p_conf(ncorr, rms)

        self.last_method = "+".join(methods) if methods else "none"
        self.last_conf = conf
        self.last_chi2 = float(chi2)
        self.last_accepted = int(g_acc or n_acc or ncorr > 0)
        self._feed_processor(feat, pred_pose, conf)


def main():
    run_node(SuperLioFusedGatedP2PGicpNDT)


if __name__ == "__main__":
    main()
