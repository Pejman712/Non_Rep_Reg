#!/usr/bin/env python3.10
"""
lio_gyro_slowrot.py  —  Variant 7: gyro rotation + SLOW GICP rotation correction.

The plain gyro+GICP-t variant leaves rotation entirely to the gyro, so heading
bias accumulates and the run drifts at the end.  Here GICP DOES correct rotation,
but only weakly: its rotation measurement covariance is inflated by
`slowrot_factor`, so each scan applies just a small fraction of the rotation
correction.  Net effect: the gyro handles fast rotation, while GICP slowly bleeds
out the long-term heading bias.  Translation is taken from GICP at full weight.

    ros2 run regnonrep lio_gyro_slowrot.py --ros-args \
        -p debug_csv:=/tmp/gyro_slowrot.csv -p slowrot_factor:=100.0
"""

from lio_base import SuperLioBase, run_node
import numpy as np


class SuperLioGyroSlowRot(SuperLioBase):
    NODE_NAME = "super_lio_gyro_slowrot"
    USE_NONREP = False
    USE_GICP = True
    VARIANT_DESC = "gyro rotation + slow GICP rotation correction"

    def __init__(self):
        super().__init__()
        # rotation covariance inflation: higher => GICP corrects rotation slower
        self.slowrot_factor = float(
            self.declare_parameter("slowrot_factor", 100.0).value)
        self.get_logger().info(f"  slowrot_factor={self.slowrot_factor}")

    def _register(self):
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            self.last_method = "none"
            return

        T_init = np.eye(4)
        T_init[:3, :3] = self.kf.R
        T_init[:3, 3] = self.kf.p

        accepted, chi2, reg_conf = False, 0.0, 0.0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            R_meas, p_meas, reg_conf, R_n = self._gicp_to_pose(scan_o3d, submap, T_init)
            # Slow rotation: decouple + inflate the rotation block so the GICP
            # rotation correction is applied weakly (gyro dominates short-term).
            R_n[0:3, 3:6] = 0.0
            R_n[3:6, 0:3] = 0.0
            R_n[0:3, 0:3] = R_n[0:3, 0:3] * self.slowrot_factor
            accepted, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        self.last_method = "gyro+slowrot" if accepted else "gicp_rej"
        self.last_conf = float(reg_conf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(accepted)


def main():
    run_node(SuperLioGyroSlowRot)


if __name__ == "__main__":
    main()
