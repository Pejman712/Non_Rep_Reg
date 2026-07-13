#!/usr/bin/env python3.10
"""
lio_nrlio_op_den.py — variant "nrlio_op_den": nrlio_optimized with a DENSITY-based
open/closed detector for the p2p↔GICP switch, replacing the median-range one.

User idea: a compact/CLOSED scene returns points densely; an OPEN (less compact)
scene returns sparse points.  So switch on point DENSITY instead of distance.

Density metric = points per occupied cell at a fixed grid:
    d = N_points / (occupied cells of size `den_voxel`)
But *absolute* density is sensor-specific (Tier Avia ≈ 23 pts/cell, CERN L1 ≈ 4.5
just from the sensors' point counts), so a fixed threshold can't generalise.
Instead we compare the (smoothed) density to a slow self-calibrating BASELINE
and switch on the RELATIVE density — works on any sensor:
    d̄ / baseline  >  den_ratio_high  → denser than usual → CLOSED → point-to-plane
    d̄ / baseline  <  den_ratio_low   → sparser than usual → OPEN  → GICP
with hysteresis in between.

Everything else (campaign-best params, guards, ZUPT, degeneracy routing) is
inherited from nrlio_optimized.

    ros2 run regnonrep lio_nrlio_op_den.py --ros-args -p den_ratio_high:=0.9 -p den_ratio_low:=0.9
"""
from collections import deque

import numpy as np

from lio_base import run_node
from lio_nrlio_optimized import SuperLioNRLIOOpt


class SuperLioNRLIOOpDen(SuperLioNRLIOOpt):
    NODE_NAME = "super_lio_nrlio_op_den"
    VARIANT_DESC = ("nrlio_optimized + density-based open/closed switch "
                    "(dense=closed→p2p, sparse=open→GICP; sensor-normalised)")

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        self.den_voxel = float(gp("den_voxel", 0.3).value)         # density grid cell [m]
        self.den_subsample = int(gp("den_subsample", 4000).value)  # cap for the density estimate
        # Tuned live in open_close_web_server.py: a single 0.9 threshold (no
        # hysteresis band) — ratio>0.9 = denser than usual = CLOSED = p2p, else OPEN.
        self.den_ratio_high = float(gp("den_ratio_high", 0.9).value)   # > → closed → p2p
        self.den_ratio_low = float(gp("den_ratio_low", 0.9).value)     # < → open → GICP
        self.den_base_alpha = float(gp("den_base_alpha", 0.02).value)  # baseline EMA rate
        # LONG smoothing window: raw per-scan density swings ~3x within one closed
        # room purely from narrow-FoV viewing incidence (near wall head-on = dense,
        # across-room = sparse).  Averaging ~60 scans (~6 s) reflects the ENVIRONMENT,
        # not the instantaneous cone, so it stays p2p in uniformly-closed spaces and
        # only trips on a *sustained* density drop (genuine open space).
        self._den_win = deque(maxlen=max(1, int(gp("den_window", 60).value)))
        self._den_base = None
        self.get_logger().info(
            f"  [op_den] density switch: cell={self.den_voxel}m window={self._den_win.maxlen} "
            f"| ratio<{self.den_ratio_low}=open(GICP) >{self.den_ratio_high}=closed(p2p) "
            f"(smoothed density vs running baseline)")

    # replace gen_lio's median-range scale switch with a relative-density switch
    def _gen_pick_mode(self, m_bar):
        pts = getattr(self, "_scan_undistort", None)
        if pts is None or pts.shape[0] < 50:
            return self._gen_mode
        p = pts
        if p.shape[0] > self.den_subsample:                 # subsample (consistent → ratio unbiased)
            idx = np.linspace(0, p.shape[0] - 1, self.den_subsample).astype(int)
            p = p[idx]
        q = np.floor(p / self.den_voxel).astype(np.int64)
        n_occ = np.unique(q, axis=0).shape[0]
        d = p.shape[0] / max(1, n_occ)                       # points per occupied cell = density
        self._den_win.append(d)
        d_bar = float(np.mean(self._den_win))
        if self._den_base is None:
            self._den_base = d_bar
        else:
            self._den_base += self.den_base_alpha * (d_bar - self._den_base)
        ratio = d_bar / max(self._den_base, 1e-6)
        new = self._gen_mode
        if ratio > self.den_ratio_high:
            new = "p2p"        # denser than usual → compact / closed
        elif ratio < self.den_ratio_low:
            new = "gicp"       # sparser than usual → open
        if new != self._gen_mode:
            self._gen_nswitch += 1
            self._gen_mode = new
        return self._gen_mode


def main():
    run_node(SuperLioNRLIOOpDen)


if __name__ == "__main__":
    main()
