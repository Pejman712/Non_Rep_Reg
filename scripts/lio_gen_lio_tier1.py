#!/usr/bin/env python3.10
"""
lio_gen_lio_tier1.py  —  gen_liotier1: Tier 1 (data density) on top of gen_lio.

Ablation ladder: gen_lio  ⊂  gen_liotier1  ⊂  gen_liotier2  ⊂  gen_liotier3.
Tier 1 isolates the effect of DATA DENSITY, which is the dominant lever for the
narrow-FoV Avia/Horizon sensors in small rooms:

  * Multi-frame scan accumulation — the last few deskewed scans are motion-
    compensated into the current frame (using each scan's corrected pose) and
    merged, so each registration cloud is denser and less degeneracy-prone.
    (Livox non-repetitive coverage grows with integration time.)
  * Small-room voxel tuning — the adaptive voxel is kept finer and the target
    point count denser so tight indoor geometry is not over-coarsened.

Everything else (adaptive voxelization, scale-switched p2p↔gicp, degenerate→gicp
routing) is inherited unchanged from gen_lio.

    ros2 run regnonrep lio_gen_lio_tier1.py --ros-args -p t1_accum_frames:=3
"""
from collections import deque

import numpy as np

from lio_base import run_node, voxel_downsample
from lio_gen_lio import SuperLioGen


class SuperLioGenT1(SuperLioGen):
    NODE_NAME = "super_lio_gen_t1"
    VARIANT_DESC = "gen_lio + T1: multi-frame accumulation (indoor density)"

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        self.t1_frames = max(1, int(gp("t1_accum_frames", 3).value))  # incl. current
        self.t1_voxel = float(gp("t1_accum_voxel", 0.12).value)
        # small-room tuning: keep the adaptive voxel finer / target denser
        self.gen_d_max = min(self.gen_d_max, float(gp("t1_voxel_max", 0.5).value))
        self.gen_n_min = max(self.gen_n_min, int(gp("t1_npts_min", 1200).value))
        self._acc = deque(maxlen=max(0, self.t1_frames - 1))   # (pts_body, R, p)
        self._t1_cur = None
        self.get_logger().info(
            f"  T1: accum_frames={self.t1_frames} accum_voxel={self.t1_voxel} "
            f"voxel_max={self.gen_d_max} n_min={self.gen_n_min}")

    def _gen_preprocess(self):
        super()._gen_preprocess()
        cur = self._scan_undistort
        if cur is None or cur.shape[0] == 0:
            self._t1_cur = None
            return
        # remember this scan (downsampled) to buffer with its corrected pose in _output
        self._t1_cur = voxel_downsample(cur, self.t1_voxel)
        if not self._acc:
            return
        # motion-compensate buffered scans into the current (propagated-prior) frame:
        #   world = R_i·p_i + t_i ;  body_now = R0ᵀ·(world − t0)
        R0, p0 = self.kf.R, self.kf.p
        merged = [cur]
        for pts_i, R_i, p_i in self._acc:
            world = pts_i @ R_i.T + p_i
            merged.append((world - p0) @ R0)
        self._scan_undistort = voxel_downsample(np.vstack(merged), self.t1_voxel)

    def _output(self):
        super()._output()
        # buffer the current scan with its now-corrected (end-of-scan) pose
        if self._t1_cur is not None and self._acc.maxlen:
            self._acc.append((self._t1_cur, self.kf.R.copy(), self.kf.p.copy()))


def main():
    run_node(SuperLioGenT1)


if __name__ == "__main__":
    main()
