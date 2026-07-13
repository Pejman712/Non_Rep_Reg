#!/usr/bin/env python3.10
"""
lio_nrlio_optB.py — variant "nrlio_optB": nrlio_optimized with a QUALITY-GATED
GICP switch instead of the scene-scale switch.

Rationale: whether GICP beats point-to-plane depends on the scene *geometry*, not
on how far the walls are.  So instead of switching on the median range, this
variant switches on point-to-plane *quality*: it tracks the most recent p2p
registration confidence and uses GICP while that confidence is weak, periodically
re-probing with p2p so it can recover when the geometry improves.  This is
self-adapting — GICP fires exactly where p2p struggles, with no scene-scale
threshold — and exposes a single tunable (`optB_p2p_conf`) for the campaign.

Degeneracy routing (inherited) still forces GICP on geometrically degenerate
scans; this gate only affects the otherwise-p2p scans.

    ros2 run regnonrep lio_nrlio_optB.py --ros-args -p optB_p2p_conf:=0.6 -p optB_reprobe:=5
"""
from lio_base import run_node
from lio_nrlio_optimized import SuperLioNRLIOOpt


class SuperLioNRLIOOptB(SuperLioNRLIOOpt):
    NODE_NAME = "super_lio_nrlio_optB"
    VARIANT_DESC = ("nrlio_optimized + quality-gated GICP "
                    "(B: use GICP when point-to-plane is weak)")

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        # use GICP while the most-recent p2p confidence is below this; re-probe
        # p2p every `optB_reprobe` scans so it recovers when geometry improves.
        self.gen_p2p_conf_switch = float(gp("optB_p2p_conf", 0.6).value)
        self.gen_qg_reprobe = int(gp("optB_reprobe", 5).value)
        self._qg_p2p_conf = 1.0
        self._qg_since_probe = 0
        self.get_logger().info(
            f"  [optB] quality-gated GICP: use GICP while p2p_conf<{self.gen_p2p_conf_switch} "
            f"(re-probe p2p every {self.gen_qg_reprobe} scans)")

    # override gen_lio's scale-based mode pick with a p2p-quality gate
    def _gen_pick_mode(self, m_bar):
        lm = getattr(self, "last_method", "")
        if "p2p" in lm:                       # fresh p2p measurement available
            self._qg_p2p_conf = float(getattr(self, "last_conf", 1.0))
            self._qg_since_probe = 0
        else:
            self._qg_since_probe += 1
        weak = self._qg_p2p_conf < self.gen_p2p_conf_switch
        if weak and self._qg_since_probe >= self.gen_qg_reprobe:
            self._qg_since_probe = 0
            return "p2p"                       # periodic re-probe to re-evaluate
        return "gicp" if weak else "p2p"


def main():
    run_node(SuperLioNRLIOOptB)


if __name__ == "__main__":
    main()
