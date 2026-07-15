#!/usr/bin/env python3.10
"""
lio_gen_lio.py  —  Variant "gen_lio": GenZ-LIO-style scale-aware adaptive
voxelization + a scale-switched hybrid metric (nonrep/GICP ↔ point-to-plane).

Inspired by GenZ-LIO (Lee et al., 2026).  Two ideas are ported into the
Super-LIO / non-rep framework:

  (1) Scale-aware adaptive voxelization.  A range-based *scale indicator*
      m̄_t (smoothed median point range over a sliding window) sets a
      scale-informed target voxel-point count N_desired.  A PD feedback
      controller then drives the downsampling voxel size d_t so the voxelized
      point count tracks that target — coarser in wide/open scenes (fewer
      points needed), finer in confined scenes (preserve geometry).  d_t drives
      BOTH registration paths (GICP scan cloud + the point-to-plane / map body
      cloud), re-voxelizing each scan.

  (2) Scale-switched hybrid metric.  Rather than fusing both metrics every scan,
      gen_lio SWITCHES the primary registration based on the scale indicator,
      with hysteresis to avoid flip-flop:
        * confined / small m̄_t   → point-to-plane  (planar structure reliable),
        * open / large m̄_t       → non-rep + GICP scan-to-submap (weak planarity).
      If GICP is unavailable/rejected on a given scan it falls back to
      point-to-plane so the scan is never dropped.  The degeneracy guard from the
      *_degen variants is kept.

Degenerate scans are routed to GICP (rather than dead-reckoned) so ill-conditioned
directions are still anchored while the map is frozen.

    # confined indoor (defaults switch low=4 m / high=7 m):
    ros2 run regnonrep lio_gen_lio.py
    # large / open dataset — raise the switch band so gicp only takes over
    # once the scene really opens up:
    ros2 run regnonrep lio_gen_lio.py --ros-args \
        -p gen_switch_low:=8.0 -p gen_switch_high:=15.0 -p gen_npts_max:=4000
"""
from collections import deque

import numpy as np

from lio_base import SuperLioBase, run_node, voxel_downsample


class SuperLioGen(SuperLioBase):
    NODE_NAME = "super_lio_gen"
    USE_NONREP = True
    USE_GICP = True
    VARIANT_DESC = "GenZ-style adaptive voxelization + scale-switched GICP/P2P"

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        # adaptive-voxelization bounds / target
        self.gen_d_min = float(gp("gen_voxel_min", 0.05).value)   # [m]
        self.gen_d_max = float(gp("gen_voxel_max", 1.00).value)   # [m]
        self.gen_n_min = int(gp("gen_npts_min", 800).value)       # N_min
        self.gen_n_max = int(gp("gen_npts_max", 4000).value)      # N_max
        self.gen_tau_m = float(gp("gen_scale_tau", 30.0).value)   # [m] scale saturation
        self.gen_p = float(gp("gen_setpoint_p", 2.0).value)       # setpoint exponent
        # PD controller gains (voxel[m] per point-count error); Δd clamped/scan
        self.gen_kp = float(gp("gen_kp", 5.0e-5).value)
        self.gen_kd = float(gp("gen_kd", 1.0e-5).value)
        self.gen_dd_max = float(gp("gen_dstep_max", 0.15).value)  # [m] max |Δd|/scan
        self.gen_window = int(gp("gen_window", 5).value)          # N_w
        # scale-switch hysteresis thresholds on m̄_t [m].  Defaults are tuned so
        # the p2p↔gicp switch is actually exercised on confined indoor scenes
        # (median ranges there are only a few metres); raise them for large/open
        # datasets so gicp only takes over once the scene really opens up.
        self.gen_switch_low = float(gp("gen_switch_low", 4.0).value)   # < → prefer P2P
        self.gen_switch_high = float(gp("gen_switch_high", 7.0).value)  # > → prefer GICP

        # controller / switch state
        self._gen_d = float(self.gicp_voxel_size)   # current voxel size d_t
        self._gen_e_prev = 0.0
        self._gen_win = deque(maxlen=max(1, self.gen_window))
        self._gen_mode = "p2p"                       # start planar until scale says otherwise
        self._gen_nswitch = 0

        self.get_logger().info(
            f"  gen_lio: voxel∈[{self.gen_d_min},{self.gen_d_max}] "
            f"N∈[{self.gen_n_min},{self.gen_n_max}] tau_m={self.gen_tau_m} "
            f"switch[{self.gen_switch_low},{self.gen_switch_high}]m "
            f"Kp={self.gen_kp} Kd={self.gen_kd}")

    # ---- (1) scale-aware adaptive voxelization ---------------------------
    def _gen_scale_and_voxel(self):
        """Update the smoothed scale indicator m̄_t and PD-control the voxel size
        d_t so the voxelized point count tracks a scale-informed setpoint.
        Returns m̄_t.  Sets self._gen_d and applies it to gicp_voxel_size and the
        body/map point cloud."""
        pts = self._scan_undistort
        if pts is None or pts.shape[0] == 0:
            return None

        # scale indicator: smoothed median point range
        rng = np.linalg.norm(pts, axis=1)
        m_t = float(np.median(rng))
        self._gen_win.append(m_t)
        m_bar = float(np.mean(self._gen_win))

        # temp voxelization at the previous voxel size to read the point count
        temp = voxel_downsample(pts, self._gen_d)
        n_temp = int(temp.shape[0])

        # scale-informed setpoint N_desired (GenZ Eq. 1), saturating power ρ
        if m_bar >= self.gen_tau_m:
            n_des = self.gen_n_max
        else:
            rho = 1.0 - (1.0 - m_bar / self.gen_tau_m) ** self.gen_p
            n_des = self.gen_n_min + (self.gen_n_max - self.gen_n_min) * rho

        # PD update on the voxel size.  e>0 ⇒ want MORE points ⇒ shrink voxel,
        # hence the leading minus sign (voxel size and point count are inverse).
        e = float(n_des) - float(n_temp)
        de = e - self._gen_e_prev
        dd = -(self.gen_kp * e + self.gen_kd * de)
        dd = float(np.clip(dd, -self.gen_dd_max, self.gen_dd_max))
        self._gen_d = float(np.clip(self._gen_d + dd, self.gen_d_min, self.gen_d_max))
        self._gen_e_prev = e

        # drive both registration paths with the adaptive voxel size
        self.gicp_voxel_size = self._gen_d
        ds = voxel_downsample(pts, self._gen_d)
        self._pts_body = ds
        self._pts_len = np.linalg.norm(ds, axis=1)
        return m_bar

    def _gen_pick_mode(self, m_bar):
        """Scale-switched metric selection with hysteresis."""
        if m_bar is None:
            return self._gen_mode
        new = self._gen_mode
        if m_bar < self.gen_switch_low:
            new = "p2p"
        elif m_bar > self.gen_switch_high:
            new = "gicp"
        if new != self._gen_mode:
            self._gen_nswitch += 1
            self._gen_mode = new
        return self._gen_mode

    # Hook for subclasses (gen_liotier1/2/3) to transform the deskewed scan
    # (accumulation, feature selection, …) BEFORE the adaptive voxelization and
    # registration.  Default no-op.
    def _gen_preprocess(self):
        pass

    # ---- registration ----------------------------------------------------
    def _register(self):
        self._gen_preprocess()
        # adaptive voxelization first (updates _pts_body / gicp_voxel_size)
        m_bar = self._gen_scale_and_voxel()

        # Degeneracy handling.  Instead of dead-reckoning through ill-conditioned
        # scans (which produced a stray excursion), we route them to GICP: its
        # point-to-point-style constraints still anchor the observable directions
        # where point-to-plane normals are unreliable (the GenZ rationale).  The
        # GICP update stays chi²-gated, so a bad one can't corrupt the state, and
        # we still FREEZE the map on degenerate scans so it isn't polluted.  Only
        # if GICP is unavailable/rejected do we fall back to IMU dead-reckoning.
        degenerate = bool(self.degen_enable and self._pts_body.shape[0]
                          and self._scan_degeneracy(self._pts_body))
        if degenerate:
            if not self._degen_active:
                self._degen_active = True
                self.get_logger().warn(
                    f"Scan {self._scan_counter}: degenerate geometry — "
                    f"GICP-only, map frozen")
            self._n_degen_skip += 1
            self._skip_map = True
            mode = "gicp"
        else:
            if self._degen_active:
                self._degen_active = False
                self.get_logger().info(
                    f"Scan {self._scan_counter}: geometry recovered — resuming")
            mode = self._gen_pick_mode(m_bar)

        tag = f"d={self._gen_d:.2f}"

        if mode == "gicp":
            scan_pts, scan_o3d = self._make_scan_cloud()
            if scan_o3d is not None:
                feat, pred_pose, _, T_init = self._nonrep_init_guess(scan_o3d)
                submap = self._get_submap()
                if submap is not None and len(submap.points) >= 30:
                    Rg, pg, gconf, Rng = self._gicp_to_pose(scan_o3d, submap, T_init)
                    g_acc, chi2 = self.kf.update_pose(
                        Rg, pg, Rng, self.chi2_threshold)
                    if g_acc:
                        self.last_method = f"{'gicp*' if degenerate else 'gicp'}[{tag}]"
                        self.last_conf = float(gconf)
                        self.last_chi2 = float(chi2)
                        self.last_accepted = 1
                        self._feed_processor(feat, pred_pose, gconf)
                        return
                # GICP unavailable / rejected.  On a degenerate scan don't risk a
                # bad P2P — dead-reckon this frame (map already frozen).
                if degenerate:
                    self.last_method = "skip_degen"
                    self.last_conf = self.last_chi2 = 0.0
                    self.last_accepted = 0
                    self._feed_processor(None, None, 0.0)
                    return
                self._n_p2p_fallback += 1
                ncorr, rms = self._observe_point_to_plane()
                self.last_method = f"gicp→p2p[{tag}]"
                self.last_conf = self._p2p_conf(ncorr, rms)
                self.last_chi2 = 0.0
                self.last_accepted = 1 if ncorr > 0 else 0
                self._feed_processor(feat, pred_pose, self.last_conf)
                return
            # scan too sparse for GICP
            if degenerate:
                self.last_method = "skip_degen"
                self.last_conf = self.last_chi2 = 0.0
                self.last_accepted = 0
                self._feed_processor(None, None, 0.0)
                return
            mode = "p2p"

        # point-to-plane branch (confined / planar)
        ncorr, rms = self._observe_point_to_plane()
        self.last_method = f"p2p[{tag}]"
        self.last_conf = self._p2p_conf(ncorr, rms)
        self.last_chi2 = 0.0
        self.last_accepted = 1 if ncorr > 0 else 0


def main():
    run_node(SuperLioGen)


if __name__ == "__main__":
    main()
