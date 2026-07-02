#!/usr/bin/env python3.10
"""
lio_gen_lio_tier3.py  —  gen_liotier3: Tier 3 (features) on top of Tier 2.

Tier 3 isolates the effect of FEATURE SELECTION.  On top of Tier 1's accumulation
and Tier 2's observability switch, it replaces the raw voxelized cloud with a
geometrically-informative, curvature-balanced subset — a mix of planar (surface)
and high-curvature (edge/corner) points — so bland rooms dominated by flat walls
still constrain all 6 DoF (the LIO-Livox rationale).

Per-point curvature is estimated from the local covariance (Open3D KNN):
    curv = λ0 / (λ0+λ1+λ2)   (small ⇒ planar, large ⇒ edge/corner)
and a fixed fraction of the flattest and sharpest points is kept.

    ros2 run regnonrep lio_gen_lio_tier3.py --ros-args -p t3_edge_frac:=0.35
"""
import numpy as np
import open3d as o3d

from lio_base import run_node
from lio_gen_lio_tier2 import SuperLioGenT2


class SuperLioGenT3(SuperLioGenT2):
    NODE_NAME = "super_lio_gen_t3"
    VARIANT_DESC = "+ T3: curvature-balanced feature selection"

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        self.t3_knn = max(5, int(gp("t3_knn", 12).value))
        self.t3_edge_frac = float(gp("t3_edge_frac", 0.35).value)   # fraction kept as edges
        self.t3_max_pts = int(gp("t3_max_pts", 6000).value)
        self.get_logger().info(
            f"  T3: knn={self.t3_knn} edge_frac={self.t3_edge_frac} max_pts={self.t3_max_pts}")

    def _gen_preprocess(self):
        super()._gen_preprocess()                    # Tier-1 accumulation
        pts = self._scan_undistort
        if pts is None or pts.shape[0] < 50:
            return
        try:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pc.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(self.t3_knn))
            w = np.linalg.eigvalsh(np.asarray(pc.covariances))   # (N,3) ascending
            curv = w[:, 0] / np.maximum(w.sum(axis=1), 1e-12)    # 0=planar … large=edge
        except Exception:
            return
        n = pts.shape[0]
        keep_n = min(n, self.t3_max_pts)
        order = np.argsort(curv)                     # planar first, edges last
        n_edge = int(self.t3_edge_frac * keep_n)
        n_plane = keep_n - n_edge
        idx = order[:n_plane]
        if n_edge > 0:
            idx = np.concatenate([idx, order[-n_edge:]])
        self._scan_undistort = pts[np.unique(idx)]


def main():
    run_node(SuperLioGenT3)


if __name__ == "__main__":
    main()
