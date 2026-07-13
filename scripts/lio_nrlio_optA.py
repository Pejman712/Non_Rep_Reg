#!/usr/bin/env python3.10
"""
lio_nrlio_optA.py — variant "nrlio_optA": nrlio_optimized with the p2p↔GICP
*scale-switch* thresholds lowered so GICP is actually integrated on the confined
indoor Tier scenes.

Motivation: on the indoor Tier sequences the median point range m̄ stays ~2-4 m,
well below the default open-scene threshold (gen_switch_high=7 m), so GICP fires
<1 % of the time (the hybrid is effectively point-to-plane only).  This variant
lowers the switch band so GICP takes over in the more-open parts of the room.

This is approach "A" (empirical scale threshold): a fixed heuristic, tunable via
the campaign.  See lio_nrlio_optB.py for the quality-gated alternative.

    ros2 run regnonrep lio_nrlio_optA.py --ros-args -p optA_switch_low:=2.0 -p optA_switch_high:=3.0
"""
from lio_base import run_node
from lio_nrlio_optimized import SuperLioNRLIOOpt


class SuperLioNRLIOOptA(SuperLioNRLIOOpt):
    NODE_NAME = "super_lio_nrlio_optA"
    VARIANT_DESC = ("nrlio_optimized + scale-switch tuned to integrate GICP indoors "
                    "(A: lowered gen_switch band)")

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        # lower the p2p↔GICP scale-switch so m̄ (median range) crosses it indoors
        self.gen_switch_low = float(gp("optA_switch_low", 2.0).value)   # was 4.0
        self.gen_switch_high = float(gp("optA_switch_high", 3.0).value)  # was 7.0
        self.get_logger().info(
            f"  [optA] scale-switched GICP: low={self.gen_switch_low}m "
            f"high={self.gen_switch_high}m (m̄ > high → GICP)")


def main():
    run_node(SuperLioNRLIOOptA)


if __name__ == "__main__":
    main()
