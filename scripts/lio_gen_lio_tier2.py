#!/usr/bin/env python3.10
"""
lio_gen_lio_tier2.py  —  gen_liotier2: Tier 2 (observability) on top of Tier 1.

Tier 2 isolates the effect of OBSERVABILITY / LOCALIZABILITY awareness.  On top of
Tier 1's denser clouds, it measures how well the (accumulated) scan constrains the
pose — the eigenvalue spread (condition) of the point distribution — and forces
GICP whenever the geometry is poorly localizable (weak-planar / near-degenerate),
rather than trusting point-to-plane there.  This is the condition-number diagnostic
used by GenZ-LIO / X-ICP / LODESTAR, and it makes the p2p↔gicp switch geometry-aware
instead of purely scale-driven.

    ros2 run regnonrep lio_gen_lio_tier2.py --ros-args -p t2_loc_thresh:=0.02
"""
import numpy as np

from lio_base import run_node
from lio_gen_lio_tier1 import SuperLioGenT1


class SuperLioGenT2(SuperLioGenT1):
    NODE_NAME = "super_lio_gen_t2"
    VARIANT_DESC = "+ T2: observability-aware switch (condition number)"

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        # min eigenvalue ratio (λ_min/λ_max) below which geometry is 'poorly localizable'
        self.t2_loc_thresh = float(gp("t2_loc_thresh", 0.02).value)
        self.last_loc = 1.0
        self.get_logger().info(f"  T2: loc_thresh={self.t2_loc_thresh}")

    def _localizability(self):
        """Scan localizability ∈ [0,1] = λ_min/λ_max of the point distribution.
        Near 0 ⇒ collapsed to a plane/line (weakly constrained)."""
        pts = self._pts_body
        if pts is None or pts.shape[0] < 10:
            return 0.0
        c = pts - pts.mean(axis=0)
        cov = (c.T @ c) / len(c)
        w = np.linalg.eigvalsh(cov)          # ascending
        return float(max(w[0], 0.0) / max(w[-1], 1e-9))

    def _gen_pick_mode(self, m_bar):
        mode = super()._gen_pick_mode(m_bar)
        self.last_loc = self._localizability()
        if self.last_loc < self.t2_loc_thresh:       # weakly constrained → GICP
            if self._gen_mode != "gicp":
                self._gen_nswitch += 1
            self._gen_mode = mode = "gicp"
        return mode


def main():
    run_node(SuperLioGenT2)


if __name__ == "__main__":
    main()
