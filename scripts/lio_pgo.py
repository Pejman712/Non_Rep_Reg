#!/usr/bin/env python3.10
"""
lio_pgo.py  —  gicp_p2p front end + SE(2) pose-graph BACKEND (loop closure).

The GICP+P2P front end (variant 3) produces the odometry; on top of it a backend
builds a keyframe pose graph and, at shutdown, closes loops and optimizes:

  * keyframes are added every kf_dist metres / kf_ang radians of motion, each
    storing its pose, timestamp and (downsampled) body cloud;
  * odometry edges connect consecutive keyframes (front-end relative pose);
  * loop closures: a keyframe within `loop_radius` of a much earlier one
    (index gap > loop_min_gap) is GICP-verified; if the fit is good enough a
    relative-pose edge is added;
  * the SE(2) pose graph is Gauss-Newton optimized (pose_graph_se2), and the
    corrected keyframe trajectory is written to debug_csv (replacing the raw
    front-end log) for evaluation.

    ros2 run regnonrep lio_pgo.py --ros-args -p debug_csv:=/tmp/pgo.csv
"""

import numpy as np
import open3d as o3d

from lio_nonrep_gicp_p2p import SuperLioGICPP2P
from lio_base import (run_node, rot_to_quat, apply_gicp_open3d,
                      estimate_registration_confidence)
import pose_graph_se2 as pg


class SuperLioPGO(SuperLioGICPP2P):
    NODE_NAME = "super_lio_pgo"
    VARIANT_DESC = "gicp_p2p front end + SE(2) pose-graph backend (loop closure)"

    def __init__(self):
        super().__init__()
        self.kf_dist = float(self.declare_parameter("kf_dist", 0.5).value)
        self.kf_ang = float(self.declare_parameter("kf_ang", 0.2).value)
        # tighter loop gating: small radius (opposite walls of a tight room are
        # close, so a large radius fabricates false loops), large index gap, high
        # GICP fitness, plus a residual sanity check.
        # loose radius (front-end drift puts a revisit ~metres away in odometry
        # coords) but rely on the strict fitness + shift filters below to reject
        # the false opposite-wall matches that a loose radius would otherwise add.
        self.loop_radius = float(self.declare_parameter("loop_radius", 2.5).value)
        self.loop_min_gap = int(self.declare_parameter("loop_min_gap", 50).value)
        self.loop_min_fitness = float(self.declare_parameter("loop_min_fitness", 0.85).value)
        # reject a loop whose GICP relative translation exceeds this (m) — a true
        # revisit (keyframes within loop_radius) should align with a small shift.
        self.loop_max_shift = float(self.declare_parameter("loop_max_shift", 1.5).value)

        self._kf_pose = []     # [x, y, theta]
        self._kf_stamp = []
        self._kf_z = []
        self._kf_cloud = []    # body-frame downsampled points
        self._last_xy = None
        self._last_th = None
        self.get_logger().info(
            f"  PGO backend: kf_dist={self.kf_dist} kf_ang={self.kf_ang} "
            f"loop_radius={self.loop_radius} min_gap={self.loop_min_gap}")

    # front end runs as usual, then we maybe snapshot a keyframe
    def _register(self):
        super()._register()
        p = self.kf.p
        th = float(np.arctan2(self.kf.R[1, 0], self.kf.R[0, 0]))
        if (self._last_xy is None
                or np.hypot(p[0] - self._last_xy[0], p[1] - self._last_xy[1]) > self.kf_dist
                or abs(pg.wrap(th - self._last_th)) > self.kf_ang):
            self._kf_pose.append([float(p[0]), float(p[1]), th])
            self._kf_stamp.append(float(self.kf.current_time))
            self._kf_z.append(float(p[2]))
            self._kf_cloud.append(self._pts_body.copy()
                                  if self._pts_body.shape[0] > 0 else np.zeros((0, 3)))
            self._last_xy = (p[0], p[1])
            self._last_th = th

    # ---- backend runs at shutdown ----------------------------------------
    def shutdown(self):
        super().shutdown()
        try:
            self._run_backend()
        except Exception as e:                       # noqa: BLE001
            self.get_logger().error(f"PGO backend failed: {e}")

    def _run_backend(self):
        N = len(self._kf_pose)
        if N < 3 or not self.debug_csv:
            self.get_logger().info(f"PGO: only {N} keyframes — skipping backend")
            return
        nodes = np.array(self._kf_pose, dtype=float)

        edges = []
        info_odom = np.diag([1.0 / 0.05**2, 1.0 / 0.05**2, 1.0 / 0.02**2])
        for k in range(1, N):
            z = pg.t2v(np.linalg.inv(pg.v2t(nodes[k - 1])) @ pg.v2t(nodes[k]))
            edges.append((k - 1, k, z, info_odom))

        nloop = self._add_loops(nodes, edges)
        err0 = pg.total_error(nodes, edges)
        self.get_logger().info(
            f"PGO: {N} keyframes, {nloop} loop closures; optimizing (err={err0:.1f})…")
        # write the RAW front-end keyframe trajectory (for isolating the backend)
        raw_path = self.debug_csv[:-4] + "_raw.csv" if self.debug_csv.endswith(".csv") \
            else self.debug_csv + "_raw"
        self._write_traj(nodes, raw_path)
        opt = pg.optimize(nodes, edges, iterations=40)
        self.get_logger().info(f"PGO: optimized (err={pg.total_error(opt, edges):.3f})")
        self._write_traj(opt, self.debug_csv)
        self.get_logger().info(
            f"PGO: wrote raw -> {raw_path}  optimized -> {self.debug_csv}")

    def _add_loops(self, nodes, edges):
        N = len(nodes)
        xy = nodes[:, 0:2]
        info_loop = np.diag([1.0 / 0.1**2, 1.0 / 0.1**2, 1.0 / 0.05**2])
        count = 0
        for j in range(self.loop_min_gap, N):
            lim = j - self.loop_min_gap
            d = np.hypot(xy[:lim, 0] - xy[j, 0], xy[:lim, 1] - xy[j, 1])
            if d.size == 0:
                continue
            i = int(np.argmin(d))
            if d[i] > self.loop_radius:
                continue
            ci, cj = self._kf_cloud[i], self._kf_cloud[j]
            if ci.shape[0] < 30 or cj.shape[0] < 30:
                continue
            z, fit = self._gicp_loop(cj, ci, nodes[i], nodes[j])
            if fit < self.loop_min_fitness:
                continue
            if np.hypot(z[0], z[1]) > self.loop_max_shift:
                continue                          # GICP shift too large -> false match
            edges.append((i, j, z, info_loop))
            count += 1
        return count

    def _gicp_loop(self, src_pts, tgt_pts, pose_i, pose_j):
        """GICP keyframe j's cloud onto keyframe i's cloud; return (z_ij SE2, fit)."""
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(src_pts.astype(np.float64))
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float64))
        rel = np.linalg.inv(pg.v2t(pose_i)) @ pg.v2t(pose_j)   # SE2 odometry guess
        init = np.eye(4)
        init[0:2, 0:2] = rel[0:2, 0:2]
        init[0:2, 3] = rel[0:2, 2]
        T = apply_gicp_open3d(src, tgt, init_T=init, voxel_size=0.0,
                              max_corr_distance=self.gicp_max_corr_distance,
                              max_iterations=self.gicp_max_iterations)
        fit = estimate_registration_confidence(src, tgt, T)
        Tse2 = np.array([[T[0, 0], T[0, 1], T[0, 3]],
                         [T[1, 0], T[1, 1], T[1, 3]],
                         [0.0, 0.0, 1.0]])
        return pg.t2v(Tse2), fit

    def _write_traj(self, poses, path):
        with open(path, "w") as f:
            f.write("scan,stamp,x,y,z,qx,qy,qz,qw\n")
            for k in range(len(poses)):
                x, y, th = poses[k]
                R = np.array([[np.cos(th), -np.sin(th), 0.0],
                              [np.sin(th), np.cos(th), 0.0],
                              [0.0, 0.0, 1.0]])
                qx, qy, qz, qw = rot_to_quat(R)
                f.write(f"{k+1},{self._kf_stamp[k]:.9f},{x:.6f},{y:.6f},"
                        f"{self._kf_z[k]:.6f},{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f}\n")


def main():
    run_node(SuperLioPGO)


if __name__ == "__main__":
    main()
