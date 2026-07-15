#!/usr/bin/env python3.10
"""
lio_nonrep_fused_degen.py  —  Variant 4: Super-LIO + non-rep/GICP + P2P FUSED
                              + degeneracy guard.

Instead of using point-to-plane only as a fallback (variant 3), this fuses BOTH
registrations as sequential measurements in the iESKF:

  1. degeneracy guard — if the scan is geometrically degenerate (near a wall /
     tight space), skip registration entirely (IMU-only) and pause the map
     insert, resuming automatically once geometry recovers;
  2. otherwise: GICP scan-to-submap -> kf.update_pose  (measurement #1),
     then  point-to-plane Observe -> kf.update_observe  (measurement #2).

Sequential ESKF fusion lets each measurement contribute according to its own
covariance — the GICP absolute pose anchors the estimate, and the point-to-plane
residuals sharpen it against the local map.

    ros2 run regnonrep lio_nonrep_fused_degen.py --ros-args -p debug_csv:=/tmp/fused.csv
"""

from lio_base import SuperLioBase, run_node


class SuperLioFusedDegen(SuperLioBase):
    NODE_NAME = "super_lio_fused_degen"
    USE_NONREP = True
    USE_GICP = True
    VARIANT_DESC = "Super-LIO + non-rep/GICP + P2P fused + degen guard"

    def _register(self):
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            # No usable scan cloud for GICP — fall back to P2P on body points.
            ncorr, rms = self._observe_point_to_plane()
            self.last_method = "p2p"
            self.last_conf = self._p2p_conf(ncorr, rms)
            self.last_chi2 = 0.0
            self.last_accepted = 1 if ncorr > 0 else 0
            return

        # --- degeneracy guard: skip both registrations + map while degenerate
        if self.degen_enable and self._scan_degeneracy(scan_pts):
            if not self._degen_active:
                self._degen_active = True
                self.get_logger().warn(
                    f"Scan {self._scan_counter}: degenerate geometry "
                    f"(concentrated/planar scan) — skipping registration & map "
                    f"update until geometry recovers")
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
                f"Scan {self._scan_counter}: geometry recovered — "
                f"resuming registration & mapping")

        feat, pred_pose, _, T_init = self._nonrep_init_guess(scan_o3d)

        # --- measurement #1: GICP absolute pose
        g_acc, chi2, gconf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_meas, p_meas, gconf, R_n = self._gicp_to_pose(scan_o3d, submap, T_init)
            g_acc, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        # --- measurement #2: point-to-plane residuals (sequential fusion)
        ncorr, rms = self._observe_point_to_plane()
        pconf = self._p2p_conf(ncorr, rms)

        if g_acc and ncorr > 0:
            self.last_method = "gicp+p2p"
        elif g_acc:
            self.last_method = "gicp"
        elif ncorr > 0:
            self.last_method = "p2p"
        else:
            self.last_method = "none"
            self._n_p2p_fallback += 1
        self.last_conf = max(gconf, pconf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(g_acc or ncorr > 0)
        self._feed_processor(feat, pred_pose, self.last_conf)


def main():
    run_node(SuperLioFusedDegen)


if __name__ == "__main__":
    main()
