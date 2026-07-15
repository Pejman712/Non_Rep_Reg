#!/usr/bin/env python3.10
"""
lio_p2p.py  —  Variant 1: vanilla Super-LIO (point-to-plane only).

The faithful Super-LIO baseline: IMU-propagated prior + point-to-plane iESKF
Observe against the OctVoxMap.  No NonRepetitiveLiDARProcessor, no GICP — this
is the reference the other three variants are compared against.

    ros2 run regnonrep lio_p2p.py --ros-args -p debug_csv:=/tmp/p2p.csv
"""

from lio_base import SuperLioBase, run_node


class SuperLioP2P(SuperLioBase):
    NODE_NAME = "super_lio_p2p"
    USE_NONREP = False
    USE_GICP = False
    VARIANT_DESC = "Super-LIO + point-to-plane (baseline)"

    def _register(self):
        ncorr, rms = self._observe_point_to_plane()
        self.last_method = "p2p"
        self.last_conf = self._p2p_conf(ncorr, rms)
        self.last_chi2 = 0.0
        self.last_accepted = 1 if ncorr > 0 else 0


def main():
    run_node(SuperLioP2P)


if __name__ == "__main__":
    main()
