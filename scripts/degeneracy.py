#!/usr/bin/env python3
"""
degeneracy.py  —  standalone LiDAR degeneracy detector (ROS 2 node).

Subscribes to a PointCloud2 topic and, on every scan, runs TWO independent
degeneracy-detection methods.  When degeneracy is detected it prints
"DEGENERACY DETECTED" together with which method(s) fired and which
degrees-of-freedom (translation/rotation axes) are unobservable.

The two methods (all grounded in the degeneracy literature):

  1. INFORMATION-MATRIX MIN-EIGENVALUE  (Zhang et al. 2016 / X-ICP, Tuna 2023)
       Build the point-to-plane information (Hessian) matrix from the scan's own
       points and surface normals,  A = (1/N) Σ J_i J_iᵀ  with
       J_i = [ p_i × n_i ; n_i ].  Split into rotational (A_rr) and translational
       (A_tt) 3×3 blocks and eigen-decompose each.  A direction whose eigenvalue
       is below a threshold is unobservable — e.g. facing one flat wall leaves
       translation along the wall unconstrained.  Identifies the exact bad DoF.

  2. INFORMATION-MATRIX CONDITION NUMBER  (Hinduja et al. 2019)
       Use the same A, but the scale-invariant relative criterion: if the
       condition number (σ_max/σ_min) of the rotational or translational block
       exceeds a threshold, the optimization is ill-conditioned / degenerate.

A scan is reported degenerate when at least `consensus` of the enabled methods
agree (default 1 = any method).

Run:
  ros2 run regnonrep degeneracy.py --ros-args -p cloud_topic:=/avia/livox/points
  # horizon:  -p cloud_topic:=/livox/points
"""

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2


# ---------------------------------------------------------------------------
# Fast PointCloud2 -> (N,3) xyz
# ---------------------------------------------------------------------------
def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)
    off = {f.name: f.offset for f in msg.fields}
    if not all(k in off for k in ("x", "y", "z")):
        return np.empty((0, 3), dtype=np.float32)
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)

    def col(o):
        return raw[:, o:o + 4].copy().ravel().view(np.float32)

    xyz = np.column_stack([col(off["x"]), col(off["y"]), col(off["z"])])
    return xyz[np.isfinite(xyz).all(axis=1)]


# =============================================================================
# InfoDegeneracyDetector — reusable, ROS-free core (the two information-matrix
# methods).  Shared by the standalone node below AND by the LIO variants
# (e.g. lio_nonrep_gicp_p2p_degen.py) so they detect degeneracy identically.
# =============================================================================
class InfoDegeneracyDetector:
    """LiDAR degeneracy detection from the scan's own point-to-plane information
    (Hessian) matrix.  Pure numpy/open3d — no ROS.  Two methods:

      EIG  — minimum eigenvalue of the rot/trans blocks (Zhang 2016 / X-ICP).
      COND — condition number of the rot/trans blocks (Hinduja 2019).

    A scan is degenerate when at least `consensus` of the enabled methods fire.
    `detect(pts, normals)` runs on a prepared cloud; `detect_from_xyz(xyz)` does
    the voxel-downsample + normal-estimation first (mirrors the node's pipeline).
    """

    def __init__(self, *, use_eig=True, use_cond=True, consensus=1,
                 trans_eig_thresh=0.04, rot_eig_ratio=0.002,
                 trans_cond_thresh=25.0, rot_cond_thresh=500.0,
                 voxel_size=0.2, normal_radius=0.5, normal_max_nn=30,
                 min_points=50):
        self.use_eig = use_eig
        self.use_cond = use_cond
        self.consensus = consensus
        # A_tt uses UNIT normals so its eigenvalues are in [0,1] (sum=1) -> a
        # scale-free absolute threshold is meaningful for translation.
        self.trans_eig_thresh = trans_eig_thresh
        # The rotational block A_rr = Σ(p×n)(p×n)ᵀ is naturally ill-conditioned
        # (far points dominate), so its threshold is deliberately LOOSE.
        self.rot_eig_ratio = rot_eig_ratio
        self.trans_cond_thresh = trans_cond_thresh
        self.rot_cond_thresh = rot_cond_thresh
        self.voxel_size = voxel_size
        self.normal_radius = normal_radius
        self.normal_max_nn = normal_max_nn
        self.min_points = min_points

    @staticmethod
    def _info_matrix(pts, normals):
        c = pts - pts.mean(axis=0)
        J = np.hstack([np.cross(c, normals), normals])    # (N,6): [rot(3), trans(3)]
        return (J.T @ J) / len(pts)                       # 6x6 normalized Hessian

    def _detect_eig(self, A):
        wt = np.clip(np.linalg.eigvalsh(A[3:6, 3:6]), 0.0, None)   # translation
        wr = np.clip(np.linalg.eigvalsh(A[0:3, 0:3]), 0.0, None)   # rotation
        msgs = []
        n_trans_deg = int(np.sum(wt < self.trans_eig_thresh))
        if n_trans_deg > 0:
            msgs.append(f"trans DoF×{n_trans_deg} unobservable (λmin={wt[0]:.3g})")
        if wr[-1] > 1e-12 and wr[0] / wr[-1] < self.rot_eig_ratio:
            msgs.append(f"rot under-constrained (λmin/λmax={wr[0]/wr[-1]:.4f})")
        return (len(msgs) > 0), "EIG: " + ("; ".join(msgs) if msgs else "ok")

    def _detect_cond(self, A):
        def cond(B):
            w = np.clip(np.linalg.eigvalsh(B), 0.0, None)
            return float(w[-1] / w[0]) if w[0] > 1e-12 else float("inf")
        ct, cr = cond(A[3:6, 3:6]), cond(A[0:3, 0:3])
        bad = (ct > self.trans_cond_thresh) or (cr > self.rot_cond_thresh)
        flags = []
        if ct > self.trans_cond_thresh:
            flags.append("trans")
        if cr > self.rot_cond_thresh:
            flags.append("rot")
        tag = (" -> " + "+".join(flags)) if flags else ""
        return bad, (f"COND: trans={ct:.1f}(thr{self.trans_cond_thresh:.0f}) "
                     f"rot={cr:.1f}(thr{self.rot_cond_thresh:.0f}){tag}")

    def detect(self, pts, normals):
        """Run the enabled methods on prepared (pts, normals).
        Returns (is_degenerate, votes, details[list of str])."""
        votes, details = 0, []
        if pts.shape[0] < self.min_points or len(normals) != len(pts):
            return True, self.consensus, [f"too few points (n={pts.shape[0]})"]
        A = self._info_matrix(pts, normals)
        if self.use_eig:
            d, m = self._detect_eig(A)
            votes += int(d); details.append(m)
        if self.use_cond:
            d, m = self._detect_cond(A)
            votes += int(d); details.append(m)
        return votes >= self.consensus, votes, details

    def detect_from_xyz(self, xyz):
        """Voxel-downsample + estimate normals on raw (N,3) xyz, then detect.
        Returns (is_degenerate, votes, details[list of str])."""
        if xyz.shape[0] < self.min_points:
            return True, self.consensus, [f"too few points (n={xyz.shape[0]})"]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        if self.voxel_size > 0:
            cloud = cloud.voxel_down_sample(self.voxel_size)
        pts = np.asarray(cloud.points)
        if pts.shape[0] < self.min_points:
            return True, self.consensus, [f"too few points after voxel (n={pts.shape[0]})"]
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.normal_radius, max_nn=self.normal_max_nn))
        return self.detect(pts, np.asarray(cloud.normals))


class DegeneracyDetector(Node):
    def __init__(self):
        super().__init__("degeneracy_detector")
        gp = self.declare_parameter

        # ---- I/O ----
        self.cloud_topic = str(gp("cloud_topic", "/livox/points").value)
        self.min_points = int(gp("min_points", 50).value)
        # print every degenerate scan, or only on state change
        self.print_every = bool(gp("print_every", False).value)

        # ---- detector core (the two information-matrix methods) ----
        self.det = InfoDegeneracyDetector(
            use_eig=bool(gp("use_eig", True).value),
            use_cond=bool(gp("use_cond", True).value),
            consensus=int(gp("consensus", 1).value),
            trans_eig_thresh=float(gp("trans_eig_thresh", 0.04).value),
            rot_eig_ratio=float(gp("rot_eig_ratio", 0.002).value),
            trans_cond_thresh=float(gp("trans_cond_thresh", 25.0).value),
            rot_cond_thresh=float(gp("rot_cond_thresh", 500.0).value),
            voxel_size=float(gp("voxel_size", 0.2).value),
            normal_radius=float(gp("normal_radius", 0.5).value),
            normal_max_nn=int(gp("normal_max_nn", 30).value),
            min_points=self.min_points)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(PointCloud2, self.cloud_topic, self.cb, qos)

        self._was_degen = False
        self._n_scans = 0
        self._n_degen = 0
        self.get_logger().info(f"degeneracy_detector listening on {self.cloud_topic}")
        self.get_logger().info(
            f"methods: eig={self.det.use_eig} cond={self.det.use_cond} "
            f"consensus={self.det.consensus}")

    # ------------------------------------------------------------------
    def cb(self, msg: PointCloud2):
        self._n_scans += 1
        xyz = pointcloud2_to_xyz(msg)
        is_degen, _votes, details = self.det.detect_from_xyz(xyz)
        forced = is_degen and details and details[0].startswith("too few")
        self._report(is_degen, details, forced=forced)

    # ------------------------------------------------------------------
    def _report(self, is_degen, details, forced=False):
        if is_degen:
            self._n_degen += 1
            if self.print_every or not self._was_degen or forced:
                self.get_logger().warn(
                    f"DEGENERACY DETECTED  [scan {self._n_scans}]  "
                    f"({self._n_degen}/{self._n_scans} = "
                    f"{100.0*self._n_degen/self._n_scans:.0f}%)  | "
                    + " | ".join(details))
            self._was_degen = True
        else:
            if self._was_degen:
                self.get_logger().info(
                    f"geometry recovered [scan {self._n_scans}]  | "
                    + " | ".join(details))
            self._was_degen = False


def main():
    rclpy.init()
    node = DegeneracyDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
