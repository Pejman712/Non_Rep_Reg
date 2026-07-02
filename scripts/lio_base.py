#!/usr/bin/env python3.10
"""
ros_lio_v3.py  —  Super-LIO Livox-Avia skeleton with the non-repetitive
                  registration observation (instead of point-to-plane).

This keeps Super-LIO's front/back end (18-DOF ESKF, midpoint IMU propagation and
per-point undistortion, 0.5 m DownSample, OctVoxMap), but **replaces the Observe
step**: rather than Super-LIO's point-to-plane residual, the scan is registered
with the regnonrep method —

    NonRepetitiveLiDARProcessor.predict_pose_adaptive  (feature / geometric /
        extrapolation prediction)  →  GICP scan-to-submap  (submap pulled from the
        kept OctVoxMap)  →  absolute world pose  →  information-form ESKF pose
        update (ESKF.update_pose).

The processor is fed back the corrected pose each scan (update_with_observation),
so its motion-pattern adaptation and feature database stay live, and its
prediction blends with the IMU-propagated pose to seed GICP.

Everything else (propagation, undistortion, map insert, output) is unchanged from
the faithful Super-LIO port.  The original Super-LIO pipeline (for reference):

  ESKF (18-DOF, state order  [R, p, v, bg, ba, g]):
      - midpoint IMU integration with online imu_scale (g-units → m/s²)
      - propagation Jacobian with right-Jacobian gyro-bias block
      - information-form iterated update:  A = P⁻¹ + HᵀV⁻¹H,  Q = A⁻¹,
        dx = Q·b + (Q·A − I)·dx_prior              (ESKF::UpdateObserve)

  Per-scan loop (SuperLIO::stateProcess):
      1. Propagation_Undistort  — propagate the filter over the scan's IMU batch,
         then motion-compensate every point into the scan-end IMU frame
         (rotation via slerp + const-accel translation).
      2. DownSample             — 0.5 m voxel-grid centroid filter.
      3. Observe                — point-to-plane iESKF update.  Each downsampled
         body point is transformed to world, its 5 nearest map points fit a plane
         (calc_plane_coeff), the residual r = nᵀ·x_w + d feeds H with
            J = [ p_body × (Rᵀn) ; n ],  weight V⁻¹ = 1000.
      4. UpdateMap              — insert the downsampled body points (world frame)
         into the voxel map.

  Init  (kf_init → map_init):
      - kf_init  accumulates ≥50 IMU samples, levels the state into a
        gravity-aligned, yaw-removed world frame, sets bg = mean gyro,
        imu_scale = g/‖mean accel‖.
      - map_init seeds the voxel map from the first few raw scans.

The OctVoxMap octree-hash KNN structure is reproduced as a 0.25 m sub-voxel hash
(running-mean representative point per sub-voxel, identical merge rule) queried
with a scipy cKDTree for the 5-NN — behaviourally equivalent to getTopK for the
plane fit, which is all Observe needs.

Reused from ros_lio.py: so3_exp, so3_log, _skew, rot_to_quat, ros_time_to_sec.

Run (Livox Avia bag with /avia/livox/points + /avia/livox/imu):
    ros2 run regnonrep ros_lio_v3.py --ros-args -p debug_csv:=/tmp/v3.csv
"""

import os
import sys
import collections
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Imu, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ros_lio import (  # noqa: E402
    so3_exp, so3_log, _skew, rot_to_quat, ros_time_to_sec,
    NonRepetitiveLiDARProcessor, estimate_registration_confidence,
    xyzi_to_open3d_cloud, apply_gicp_open3d,
)


# =============================================================================
# Helpers
# =============================================================================
def right_jacobian_so3(omega: np.ndarray, dt: float) -> np.ndarray:
    """J_r(omega·dt) — matches ESKF.cpp RightJacobianSO3 (returns J_r, not ·dt)."""
    n = float(np.linalg.norm(omega))
    if n < 1e-8:
        return np.eye(3)
    axis = omega / n
    ang = n * dt
    K = _skew(axis)
    a = (1.0 - np.cos(ang)) / ang
    b = 1.0 - np.sin(ang) / ang
    return np.eye(3) - a * K + b * (K @ K)


def quat_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → [x,y,z,w] quaternion (for slerp)."""
    x, y, z, w = rot_to_quat(R)
    return np.array([x, y, z, w], dtype=float)


def voxel_downsample(pts: np.ndarray, leaf: float) -> np.ndarray:
    """Centroid-per-voxel downsample (≈ pcl::VoxelGrid)."""
    if pts.shape[0] == 0 or leaf <= 0.0:
        return pts
    keys = np.floor(pts / leaf).astype(np.int64)
    # unique voxel, averaged centroid
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ks = keys[order]
    ps = pts[order]
    boundary = np.ones(ks.shape[0], dtype=bool)
    boundary[1:] = np.any(ks[1:] != ks[:-1], axis=1)
    grp = np.cumsum(boundary) - 1
    ng = grp[-1] + 1
    sums = np.zeros((ng, 3))
    np.add.at(sums, grp, ps)
    counts = np.bincount(grp, minlength=ng).reshape(-1, 1)
    return sums / counts


# =============================================================================
# OctVoxMap (Python) — sub-voxel hash with 5-NN via cKDTree
# =============================================================================
class OctVoxMapPy:
    """Super-LIO OctVoxMap (kept): one representative point per 0.25 m sub-voxel
    (resolution/2), updated as a running mean of points within 0.1 m of the
    stored point (max 20), exactly like OctVox::AddPoint.

    For the non-rep registration the map is queried as a local point-cloud
    submap (get_submap) rather than via 5-NN plane fits."""

    MERGE_DIST2 = 0.1 * 0.1
    MAX_PER_SUBVOX = 20

    def __init__(self, resolution: float = 0.5, capacity: int = 2_000_000):
        self.sub_res = resolution / 2.0
        self.inv_sub = 1.0 / self.sub_res
        self.capacity = capacity
        # cell key (i,j,k) -> [mean(3), count]
        self._cells: "collections.OrderedDict" = collections.OrderedDict()
        self._pts_cache: Optional[np.ndarray] = None
        self._tree: Optional[cKDTree] = None
        self._dirty = True

    def __len__(self) -> int:
        return len(self._cells)

    def insert(self, pts_world: np.ndarray) -> None:
        if pts_world.shape[0] == 0:
            return
        keys = np.floor(pts_world * self.inv_sub).astype(np.int64)
        cells = self._cells
        for i in range(pts_world.shape[0]):
            k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            p = pts_world[i]
            ent = cells.get(k)
            if ent is None:
                cells[k] = [p.copy(), 1]
                if len(cells) > self.capacity:
                    cells.popitem(last=False)
            else:
                mean, cnt = ent
                if cnt >= self.MAX_PER_SUBVOX:
                    continue
                if float(np.dot(p - mean, p - mean)) > self.MERGE_DIST2:
                    continue
                ent[0] = (mean * cnt + p) / (cnt + 1)
                ent[1] = cnt + 1
        self._dirty = True

    def all_points(self) -> np.ndarray:
        """(N,3) array of representative points (cached until next insert)."""
        if self._dirty or self._pts_cache is None:
            if len(self._cells) == 0:
                self._pts_cache = np.zeros((0, 3))
            else:
                self._pts_cache = np.array([v[0] for v in self._cells.values()],
                                           dtype=float)
            self._tree = None        # invalidate KD-tree too
            self._dirty = False
        return self._pts_cache

    def knn5(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """5-NN over representative points (Super-LIO getTopK equivalent) for the
        point-to-plane fallback.  Returns (dist (N,5), neighbor_pts (N,5,3));
        missing neighbours have dist = inf."""
        pts = self.all_points()
        if self._tree is None and pts.shape[0] >= 1:
            self._tree = cKDTree(pts)
        if self._tree is None or pts.shape[0] < 4:
            n = query.shape[0]
            return np.full((n, 5), np.inf), np.zeros((n, 5, 3))
        kk = min(5, pts.shape[0])
        dist, idx = self._tree.query(query, k=kk)
        if kk == 1:
            dist, idx = dist[:, None], idx[:, None]
        if kk < 5:
            pad = 5 - kk
            dist = np.concatenate([dist, np.full((dist.shape[0], pad), np.inf)], axis=1)
            idx = np.concatenate([idx, np.zeros((idx.shape[0], pad), dtype=idx.dtype)], axis=1)
        return dist, pts[idx]

    def get_submap(self, center: np.ndarray, radius: float) -> o3d.geometry.PointCloud:
        """Open3D cloud of representative points within `radius` of `center`."""
        pts = self.all_points()
        cloud = o3d.geometry.PointCloud()
        if pts.shape[0] == 0:
            return cloud
        d2 = np.sum((pts - center) ** 2, axis=1)
        inside = pts[d2 <= radius * radius]
        if inside.shape[0] > 0:
            cloud.points = o3d.utility.Vector3dVector(inside)
        return cloud


# =============================================================================
# ESKF — 18-DOF, state order [R, p, v, bg, ba, g]  (port of super_lio ESKF)
# =============================================================================
class ESKF:
    def __init__(self, gravity_norm: float = 9.8015):
        self.gravity_norm = gravity_norm
        # nominal state
        self.R = np.eye(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.g = np.array([0.0, 0.0, -gravity_norm])
        self.imu_scale = 1.0

        self.P = np.eye(18)
        self.Q = np.zeros((12, 12))

        # noise (set in set_initial_conditions)
        self.num_iterations = 4
        self.quit_eps = 1e-3

        self.current_time = 0.0
        self.last_imu_time = -1.0
        self.last_obs_time = 0.0
        self.current_obs_time = 0.0
        self.last_imu: Optional[Tuple[np.ndarray, np.ndarray, float]] = None  # (acc, gyr, secs)
        self.init = False

    # ---- noise -----------------------------------------------------------
    def build_noise(self, ng, na, nbg, nba):
        self.Q = np.zeros((12, 12))
        self.Q[0:3, 0:3] = np.eye(3) * ng     # gyro
        self.Q[3:6, 3:6] = np.eye(3) * na     # accel
        self.Q[6:9, 6:9] = np.eye(3) * nbg    # bias gyro
        self.Q[9:12, 9:12] = np.eye(3) * nba  # bias accel

    def set_initial_conditions(self, ng, na, nbg, nba, init_bg, imu_scale, gravity):
        self.build_noise(ng, na, nbg, nba)
        self.bg = init_bg.copy()
        self.ba = np.zeros(3)
        self.g = gravity.copy()
        self.imu_scale = imu_scale
        self.P = 1e-4 * np.eye(18)
        self.P[0:3, 0:3] = (0.1 * np.pi / 180.0) * np.eye(3)

    def set_x(self, R, p, v, t):
        self.R, self.p, self.v = R.copy(), p.copy(), v.copy()
        self.last_imu_time = t
        self.current_time = t

    def get_se3(self):
        return self.R.copy(), self.p.copy()

    # ---- propagation -----------------------------------------------------
    def predict(self, acc_raw, gyr_raw, secs) -> bool:
        """Mirror ESKF::Predict(imu).  Returns True if a step was integrated."""
        if self.last_imu is None or self.last_imu_time < 0:
            self.last_imu_time = secs
            self.last_imu = (acc_raw.copy(), gyr_raw.copy(), secs)
            return False
        if secs <= self.last_obs_time:
            self.last_imu_time = secs
            self.last_imu = (acc_raw.copy(), gyr_raw.copy(), secs)
            return False

        self.current_time = secs
        if self.last_imu_time < self.last_obs_time:
            dt = secs - self.last_obs_time
        elif secs > self.current_obs_time:
            dt = self.current_obs_time - self.last_imu_time
            self.current_time = self.current_obs_time
        else:
            dt = secs - self.last_imu_time

        if dt <= 0.0:
            self.last_imu_time = secs
            self.last_imu = (acc_raw.copy(), gyr_raw.copy(), secs)
            return False

        last_acc, last_gyr, _ = self.last_imu
        acc = 0.5 * (acc_raw + last_acc) * self.imu_scale - self.ba
        omega = 0.5 * (gyr_raw + last_gyr) - self.bg

        Jr = right_jacobian_so3(omega, dt)
        Jr_dt = Jr * dt
        R = self.R
        R_dt = R * dt

        F = np.eye(18)
        F[0:3, 0:3] = so3_exp(-omega * dt)
        F[0:3, 9:12] = -Jr_dt
        F[3:6, 6:9] = np.eye(3) * dt
        F[6:9, 0:3] = -(R @ _skew(acc)) * dt
        F[6:9, 12:15] = -R_dt
        F[6:9, 15:18] = np.eye(3) * dt

        Fw = np.zeros((18, 12))
        Fw[0:3, 0:3] = -Jr_dt
        Fw[6:9, 3:6] = -R_dt
        Fw[9:12, 6:9] = np.eye(3) * dt
        Fw[12:15, 9:12] = np.eye(3) * dt

        self.P = F @ self.P @ F.T + Fw @ self.Q @ Fw.T

        global_acc = R @ acc + self.g
        self.p = self.p + self.v * dt + 0.5 * global_acc * dt * dt
        self.v = self.v + global_acc * dt
        self.R = R @ so3_exp(omega * dt)

        self.last_imu_time = secs
        self.last_imu = (acc_raw.copy(), gyr_raw.copy(), secs)
        # dynamic state for undistortion (world-frame accel, debiased omega)
        self._last_dyn = (self.current_time, self.R.copy(), self.p.copy(),
                          self.v.copy(), global_acc.copy())
        return True

    def dynamic_state(self):
        return (self.current_time, self.R.copy(), self.p.copy(),
                self.v.copy(), np.zeros(3))

    # ---- iterated information-form update --------------------------------
    def update_observe(self, obs_fn) -> None:
        """obs_fn(R, p, need_converge) -> (HTVH(6x6), HTVr(6))."""
        R_pred = self.R.copy()
        p_pred = self.p.copy()
        v_pred = self.v.copy()
        bg_pred = self.bg.copy()
        ba_pred = self.ba.copy()
        g_pred = self.g.copy()
        P_pred = self.P.copy()

        Qk = np.eye(18)
        dx = np.zeros(18)

        for it in range(self.num_iterations):
            need_converge = it > 2
            HTVH, HTVr = obs_fn(self.R, self.p, need_converge)

            dx_prior = np.zeros(18)
            dx_prior[0:3] = so3_log(R_pred.T @ self.R)
            dx_prior[3:6] = self.p - p_pred
            dx_prior[6:9] = self.v - v_pred
            dx_prior[9:12] = self.bg - bg_pred
            dx_prior[12:15] = self.ba - ba_pred
            dx_prior[15:18] = self.g - g_pred

            G_prior = np.eye(18)
            G_prior[0:3, 0:3] = np.eye(3) - 0.5 * _skew(dx_prior[0:3])
            Pk = G_prior @ P_pred @ G_prior.T
            dx_prior = G_prior @ dx_prior

            HTRH = np.zeros((18, 18))
            HTRH[0:6, 0:6] = HTVH
            try:
                A = np.linalg.inv(Pk) + HTRH
                Qk = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                break

            b = np.zeros(18)
            b[0:6] = HTVr
            K_x = Qk @ HTRH
            dx = Qk @ b + (K_x - np.eye(18)) @ dx_prior

            self.R = self.R @ so3_exp(dx[0:3])
            self.p = self.p + dx[3:6]
            self.v = self.v + dx[6:9]
            self.bg = self.bg + dx[9:12]
            self.ba = self.ba + dx[12:15]
            self.g = self.g + dx[15:18]
            gn = float(np.linalg.norm(self.g))
            if gn > 1e-9:
                self.g = self.gravity_norm * (self.g / gn)

            if it > 0 and float(np.max(np.abs(dx))) < self.quit_eps:
                break

        self.P = Qk
        G_reset = np.eye(18)
        G_reset[0:3, 0:3] = np.eye(3) - 0.5 * _skew(dx[0:3])
        self.P = G_reset @ self.P @ G_reset.T
        self.P = 0.5 * (self.P + self.P.T)
        self.last_obs_time = self.current_obs_time

    # ---- absolute-pose update (registration observation) -----------------
    def update_pose(self, R_meas, p_meas, R_n, chi2_threshold: float = 0.0):
        """Fuse an absolute world pose (R_meas, p_meas) from registration via the
        information-form iterated update.  R_n is 6x6 in [rot(3), pos(3)] order,
        matching the state's [R(0:3), p(3:6)] blocks.  Returns (accepted, chi2)."""
        r0 = np.zeros(6)
        r0[0:3] = so3_log(self.R.T @ R_meas)
        r0[3:6] = p_meas - self.p
        S = self.P[0:6, 0:6] + R_n
        try:
            chi2 = float(r0 @ np.linalg.solve(S, r0))
        except np.linalg.LinAlgError:
            chi2 = 0.0
        if chi2_threshold > 0.0 and chi2 > chi2_threshold:
            return False, chi2
        try:
            V_inv = np.linalg.inv(R_n)
        except np.linalg.LinAlgError:
            return False, chi2

        def obs_fn(R, p, need_converge):
            r = np.zeros(6)
            r[0:3] = so3_log(R.T @ R_meas)
            r[3:6] = p_meas - p
            return V_inv, V_inv @ r

        self.update_observe(obs_fn)
        return True, chi2


# =============================================================================
# SuperLIO base node — shared Super-LIO core; subclasses implement _register()
# =============================================================================
class SuperLioBase(Node):
    # --- per-variant configuration (override in subclasses) ------------------
    NODE_NAME = "super_lio_base"
    USE_NONREP = False   # build the NonRepetitiveLiDARProcessor (prediction)
    USE_GICP = False     # build the GICP backend (small_gicp / open3d)
    VARIANT_DESC = "base"

    def __init__(self):
        super().__init__(self.NODE_NAME)

        # ---- parameters (defaults = super_lio config/livox_avia.yaml) ------
        gp = self.declare_parameter
        self.lidar_topic = gp("lidar_topic", "/avia/livox/points").value
        self.imu_topic = gp("imu_topic", "/avia/livox/imu").value
        self.gravity_norm = float(gp("gravity_norm", 9.8015).value)
        self.blind2 = float(gp("blind", 0.5).value) ** 2
        self.maxrange2 = float(gp("maxrange", 100.0).value) ** 2
        self.filter_rate = int(gp("filter_rate", 3).value)
        self.voxel_filter_size = float(gp("voxel_filter_size", 0.5).value)
        self.imu_ng = float(gp("imu_ng", 0.1).value)
        self.imu_na = float(gp("imu_na", 0.1).value)
        self.imu_nbg = float(gp("imu_nbg", 1e-4).value)
        self.imu_nba = float(gp("imu_nba", 1e-4).value)
        self.kf_max_iterations = int(gp("kf_max_iterations", 4).value)
        self.kf_quit_eps = float(gp("kf_quit_eps", 1e-3).value)
        self.vox_resolution = float(gp("vox_resolution", 0.5).value)
        self.hash_capacity = int(gp("hash_capacity", 2_000_000).value)
        self.kf_init_imu_count = int(gp("kf_init_imu_count", 50).value)
        self.map_init_frames = int(gp("map_init_frames", 3).value)
        self.pub_step = int(gp("pub_step", 1).value)
        self.publish_cloud = bool(gp("publish_cloud", True).value)
        self.debug_csv = str(gp("debug_csv", "").value)
        # lidar→imu extrinsic (avia default), row-major R
        ext = list(gp("lidar_imu", [0.04165, 0.02326, -0.0284,
                                    1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0]).value)
        self.TLI_t = np.array(ext[0:3], dtype=float)
        self.TLI_R = np.array(ext[3:12], dtype=float).reshape(3, 3)

        # ---- non-rep registration (replaces Super-LIO point-to-plane) ------
        self.gicp_submap_radius = float(gp("gicp_submap_radius", 25.0).value)
        self.gicp_voxel_size = float(gp("gicp_voxel_size", 0.2).value)
        self.gicp_max_corr_distance = float(gp("gicp_max_corr_distance", 2.0).value)
        self.gicp_max_iterations = int(gp("gicp_max_iterations", 50).value)
        self.gicp_min_conf = float(gp("gicp_min_conf", 0.25).value)
        self.gicp_cov_scale = float(gp("gicp_cov_scale", 1.0).value)
        self.meas_noise_pos = float(gp("ieskf_meas_noise_pos", 0.05).value)
        self.meas_noise_rot = float(gp("ieskf_meas_noise_rot", 0.01).value)
        self.chi2_threshold = float(gp("gicp_chi2_threshold", 50.0).value)
        self.submap_min_cells = int(gp("submap_min_cells", 50).value)
        self.imu_base_weight = float(gp("imu_base_weight", 0.3).value)
        self.nonrep_base_weight = float(gp("nonrep_base_weight", 0.7).value)
        self.use_pctools_gicp = bool(gp("use_pctools_gicp", True).value)
        self.force_z_zero = bool(gp("force_z_zero", False).value)

        # ---- NDT registration (used by the *_ndt variants) ----------------
        self.ndt_resolution = float(gp("ndt_resolution", 1.0).value)   # voxel size [m]
        self.ndt_max_iter = int(gp("ndt_max_iter", 15).value)
        self.ndt_min_voxel_pts = int(gp("ndt_min_voxel_pts", 6).value)

        # ---- degeneracy guard (skip registration near walls / tight spaces) -
        # Stricter defaults => the guard trips more readily, so more marginal
        # scans are treated as degenerate and their registration is skipped.
        self.degen_enable = bool(gp("degen_skip_enable", True).value)
        # near-planar/linear if smallest PCA eigenvalue is < ratio·largest
        self.degen_planarity_ratio = float(gp("degen_planarity_ratio", 0.03).value)
        # whole scan "concentrated in a small area" if its largest spatial
        # spread (std-dev along the dominant axis, m) is below this
        self.degen_min_extent = float(gp("degen_min_extent", 1.0).value)
        self.degen_min_points = int(gp("degen_min_points", 80).value)
        # log the PCA metrics for every scan so the thresholds can be tuned
        self.degen_debug = bool(gp("degen_debug", False).value)

        # ---- filter / map --------------------------------------------------
        self.kf = ESKF(self.gravity_norm)
        self.kf.num_iterations = self.kf_max_iterations
        self.kf.quit_eps = self.kf_quit_eps
        self.ivox = OctVoxMapPy(self.vox_resolution, self.hash_capacity)

        # ---- non-repetitive processor + GICP backend (per-variant) --------
        self.processor = (NonRepetitiveLiDARProcessor(force_z_zero=self.force_z_zero)
                          if self.USE_NONREP else None)
        if self.USE_GICP:
            self._gicp_func, self._gicp_backend = self._resolve_gicp()
        else:
            self._gicp_func, self._gicp_backend = None, "none"

        # ---- measurement buffers (super_lio sync_measure) ------------------
        self._lidar_buf: "collections.deque" = collections.deque()
        self._imu_buf: "collections.deque" = collections.deque()
        self._last_imu_stamp = -1.0
        self._last_lidar_stamp = -1.0
        self._lidar_pushed = False
        self._cur_lidar = None

        # ---- state machine -------------------------------------------------
        self._state = "wait_kf"
        self._kf_imu_count = 0
        self._kf_mean_gyro = np.zeros(3)
        self._kf_mean_acce = np.zeros(3)
        self._frame_num = 0
        self._sys_init_R = np.eye(3)
        self._sys_init_t = np.zeros(3)
        self._scan_counter = 0

        # downsampled body points of the current scan (reused by UpdateMap)
        self._pts_body = np.zeros((0, 3))
        self._pts_len = np.zeros(0)
        self.last_conf = 0.0
        self.last_chi2 = 0.0
        self.last_accepted = 0
        self.last_method = "none"
        self._n_p2p_fallback = 0
        self._n_lidar_recv = 0            # DIAG: scans delivered to cb_lidar
        self._n_lidar_parse_none = 0      # DIAG: scans rejected by _parse_cloud
        self._degen_active = False        # currently in a degenerate stretch
        self._skip_map = False            # pause map insertion for this scan
        self._n_degen_skip = 0
        self._proc_ms = []                # DIAG: per-scan processing time [ms]

        # ---- ROS I/O -------------------------------------------------------
        # IMU intake and LiDAR processing live in SEPARATE callback groups so a
        # MultiThreadedExecutor (see run_node) services them on different
        # threads: the heavy GICP/registration run from cb_lidar never blocks
        # IMU buffering.  A lock guards the cross-thread sensor buffers.
        self._buf_lock = threading.Lock()
        self._data_cv = threading.Condition(self._buf_lock)
        self._stop_worker_flag = False
        self._worker: Optional[threading.Thread] = None
        self._imu_cbg = MutuallyExclusiveCallbackGroup()
        self._lidar_cbg = MutuallyExclusiveCallbackGroup()
        # RELIABLE + deep queues so scans are never silently dropped and the bag
        # back-pressures to this node's processing speed (bag rate stops affecting
        # correctness — same approach as the external SOTA nodes / converter).
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=5000)
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=2000)
        self.create_subscription(Imu, self.imu_topic, self.cb_imu, imu_qos,
                                 callback_group=self._imu_cbg)
        self.create_subscription(PointCloud2, self.lidar_topic, self.cb_lidar,
                                 lidar_qos, callback_group=self._lidar_cbg)
        self.pub_odom = self.create_publisher(Odometry, "/lio/odom", 100)
        self.pub_cloud = self.create_publisher(PointCloud2, "/lio/cloud_world", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self._csv_fh = None
        if self.debug_csv:
            self._csv_fh = open(self.debug_csv, "w")
            self._csv_fh.write("scan,stamp,x,y,z,qx,qy,qz,qw\n")
            self._csv_fh.flush()

        self.get_logger().info(f"=== {self.NODE_NAME}  [{self.VARIANT_DESC}] ===")
        self.get_logger().info(f"  lidar={self.lidar_topic}  imu={self.imu_topic}")
        self.get_logger().info(f"  gravity_norm={self.gravity_norm}  voxel_filter={self.voxel_filter_size}")
        self.get_logger().info(f"  vox_resolution={self.vox_resolution}  kf_iters={self.kf_max_iterations}")
        self.get_logger().info(f"  use_nonrep={self.USE_NONREP}  use_gicp={self.USE_GICP}  "
                               f"gicp_backend={self._gicp_backend}")
        if self.USE_GICP:
            self.get_logger().info(f"  gicp voxel={self.gicp_voxel_size}  submap_r={self.gicp_submap_radius}  "
                                   f"chi2={self.chi2_threshold}")
            self.get_logger().info(f"  init-guess weights: imu={self.imu_base_weight} "
                                   f"nonrep={self.nonrep_base_weight}  force_z_zero={self.force_z_zero}")
        if self.debug_csv:
            self.get_logger().info(f"  debug_csv → {self.debug_csv}")

    # ------------------------------------------------------------------
    # GICP backend: prefer Pctools small_gicp (returns Hessian), else Open3D
    # ------------------------------------------------------------------
    def _resolve_gicp(self):
        if self.use_pctools_gicp:
            try:
                from Pctools import apply_gicp_with_init_full
                return apply_gicp_with_init_full, "small_gicp"
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"Pctools small_gicp unavailable ({e}); using Open3D GICP")
        return None, "open3d"

    def _gicp(self, scan_o3d, submap_o3d, init_abs):
        """Register scan(body) → submap(world).  Returns (T_world(4x4), H or None).
        T_world maps the scan body frame into world (= the absolute scan pose)."""
        if self._gicp_backend == "small_gicp":
            # apply_gicp_with_init_full(source=submap, target=scan): small_gicp
            # aligns its 'source' onto 'target' and returns T_target_source =
            # scan(body) → submap(world).  init_T is the absolute pose guess.
            T, H = self._gicp_func(submap_o3d, scan_o3d, init_T=init_abs,
                                   voxel_size=self.gicp_voxel_size)
            return T, H
        # Open3D GICP: registration_generalized_icp(src=scan, tgt=submap, init)
        # returns the transform mapping scan → submap = body → world.
        T = apply_gicp_open3d(scan_o3d, submap_o3d, init_T=init_abs,
                              voxel_size=0.0,  # scan already downsampled
                              max_corr_distance=self.gicp_max_corr_distance,
                              max_iterations=self.gicp_max_iterations)
        return T, None

    # ------------------------------------------------------------------
    # Callbacks — buffer messages, then drain via sync_measure
    # ------------------------------------------------------------------
    def cb_imu(self, msg: Imu):
        secs = ros_time_to_sec(msg.header.stamp)
        acc = np.array([msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z])
        gyr = np.array([msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z])
        # Callbacks ONLY buffer + wake the worker — NO GICP/processing here, so
        # the executor drains the IMU/LiDAR DDS queues promptly and nothing is
        # dropped even when per-scan compute can't keep up with real-time.
        with self._data_cv:
            if secs < self._last_imu_stamp:
                self._imu_buf.clear()
            self._imu_buf.append((acc, gyr, secs))
            self._last_imu_stamp = secs
            self._data_cv.notify()

    def cb_lidar(self, msg: PointCloud2):
        self._n_lidar_recv += 1          # DIAG: how many scans reached the cb
        scan = self._parse_cloud(msg)        # numpy parse — off the IMU thread
        if scan is None:
            self._n_lidar_parse_none += 1
            return
        with self._data_cv:
            self._lidar_buf.append(scan)
            self._data_cv.notify()

    def _parse_cloud(self, msg: PointCloud2):
        """Extract (xyz, offset_time[s], start_time, end_time) — super_lio
        stdMsgHandler LIVOX_PC2 case: filter by tag, blind/maxrange, filter_rate."""
        n = msg.width * msg.height
        if n < 10:
            return None
        fo = {f.name: f for f in msg.fields}
        step = msg.point_step
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, step)

        def col_f32(off):
            return raw[:, off:off + 4].copy().ravel().view(np.float32).astype(np.float64)

        x = col_f32(fo["x"].offset)
        y = col_f32(fo["y"].offset)
        z = col_f32(fo["z"].offset)

        # per-point time → seconds relative to scan start.  Livox Avia/Horizon
        # publish uint32-ns 'offset_time' (already relative to scan start); the
        # Livox mid-360 (iilab /eve/lidar3d) publishes float64-ns absolute
        # 'timestamp' instead — rebase it to the scan's first point.
        if "offset_time" in fo:
            o = fo["offset_time"].offset
            ot = raw[:, o:o + 4].copy().ravel().view(np.uint32).astype(np.float64) * 1e-9
        elif "timestamp" in fo:
            o = fo["timestamp"].offset
            ts = raw[:, o:o + 8].copy().ravel().view(np.float64)
            ot = (ts - ts.min()) * 1e-9
        elif "t" in fo:
            o = fo["t"].offset
            ot = raw[:, o:o + 4].copy().ravel().view(np.uint32).astype(np.float64) * 1e-9
        else:
            return None

        # decimate (super_lio steps by g_filter_rate)
        if self.filter_rate > 1:
            sel = np.arange(0, n, self.filter_rate)
            x, y, z, ot = x[sel], y[sel], z[sel], ot[sel]

        d2 = x * x + y * y + z * z
        good = np.isfinite(d2) & (d2 > self.blind2) & (d2 < self.maxrange2)
        if "tag" in fo:
            tag = raw[:, fo["tag"].offset]
            if self.filter_rate > 1:
                tag = tag[sel]
            tag_ok = ((tag & 0x30) == 0x10) | ((tag & 0x30) == 0x00)
            good = good & tag_ok
        if not np.any(good):
            return None
        xyz = np.column_stack([x[good], y[good], z[good]])
        ot = ot[good]
        start = ros_time_to_sec(msg.header.stamp)
        end = start + float(ot.max())
        return (xyz, ot, start, end)

    # ------------------------------------------------------------------
    # Processing worker — owns ALL heavy work (GICP/ESKF), running on its own
    # thread off the executor.  Because the sensor callbacks only buffer, the
    # DDS queues are drained promptly and scans/IMU are never dropped; the
    # backlog lives in the in-memory deques and is processed in timestamp order.
    # The worker processes every ready scan back-to-back, then sleeps on the CV.
    # ------------------------------------------------------------------
    def _start_worker(self):
        self._stop_worker_flag = False
        self._worker = threading.Thread(target=self._worker_loop,
                                        name=f"{self.NODE_NAME}_worker",
                                        daemon=True)
        self._worker.start()

    def _stop_worker(self):
        with self._data_cv:
            self._stop_worker_flag = True
            self._data_cv.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None

    def _worker_loop(self):
        while True:
            with self._data_cv:
                # Wait (releasing the lock) until a scan+IMU pair is ready, or
                # shutdown.  _sync_measure() runs under the lock (fast); IMU at
                # ~200 Hz keeps notifying so this wakes within ms during play.
                while not self._stop_worker_flag and not self._sync_measure():
                    self._data_cv.wait(timeout=0.5)
                if self._stop_worker_flag:
                    return
            # _sync_measure() populated _meas_imu/_meas_lidar; the heavy step
            # runs WITHOUT the lock so callbacks keep buffering concurrently.
            self._process()

    def _sync_measure(self) -> bool:
        """Port of ROSWrapper::sync_measure: pair the front scan with the IMU
        samples up to its end time."""
        if not self._lidar_buf or not self._imu_buf:
            return False
        if not self._lidar_pushed:
            self._cur_lidar = self._lidar_buf[0]
            self._lidar_pushed = True
        end_time = self._cur_lidar[3]
        if self._last_lidar_stamp > end_time:
            self._lidar_buf.popleft()
            self._lidar_pushed = False
            return False
        if self._last_imu_stamp < end_time:
            return False  # wait for more IMU
        imu_batch = []
        while self._imu_buf and self._imu_buf[0][2] <= end_time:
            imu_batch.append(self._imu_buf.popleft())
        self._meas_imu = imu_batch
        self._meas_lidar = self._cur_lidar
        self._last_lidar_stamp = end_time
        self._lidar_buf.popleft()
        self._lidar_pushed = False
        return True

    # ------------------------------------------------------------------
    # State machine (SuperLIO::process)
    # ------------------------------------------------------------------
    def _process(self):
        if self._state == "wait_kf":
            if self._kf_init():
                self._state = "wait_map"
                self.get_logger().info(" ---> KF init done")
        elif self._state == "wait_map":
            if self._map_init():
                self.kf.init = True
                self._state = "process"
                self.get_logger().info(" ---> Map init done")
        else:
            self._state_process()

    def _kf_init(self) -> bool:
        for acc, gyr, _ in self._meas_imu:
            self._kf_imu_count += 1
            self._kf_mean_gyro += (gyr - self._kf_mean_gyro) / self._kf_imu_count
            self._kf_mean_acce += (acc - self._kf_mean_acce) / self._kf_imu_count
        if self._kf_imu_count < self.kf_init_imu_count:
            return False

        mean_acce = self._kf_mean_acce
        mean_gyro = self._kf_mean_gyro
        anorm = float(np.linalg.norm(mean_acce))
        gravity = -mean_acce * self.gravity_norm / anorm
        ref_gravity = np.array([0.0, 0.0, -self.gravity_norm])
        init_rot = self._from_two_vectors(gravity, ref_gravity)
        n = init_rot[:, 0]
        yaw = float(np.arctan2(n[1], n[0]))
        R_yaw_inv = so3_exp(np.array([0.0, 0.0, -yaw]))
        rot = R_yaw_inv @ init_rot  # g_lidar_robo_yaw = I

        imu_scale = self.gravity_norm / anorm
        self.kf.set_initial_conditions(self.imu_ng, self.imu_na, self.imu_nbg,
                                       self.imu_nba, mean_gyro, imu_scale, ref_gravity)
        t = self._meas_imu[-1][2]
        self.kf.set_x(rot, np.zeros(3), np.zeros(3), t)
        self._sys_init_R = rot.copy()
        self._sys_init_t = np.zeros(3)
        return True

    @staticmethod
    def _from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Quaternion::FromTwoVectors(a,b) → R such that R·a ∝ b."""
        a = a / float(np.linalg.norm(a))
        b = b / float(np.linalg.norm(b))
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = float(np.linalg.norm(v))
        if s < 1e-9:
            if c > 0.0:
                return np.eye(3)
            # 180°: rotate about any axis ⟂ a
            axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
            axis = axis / np.linalg.norm(axis)
            return so3_exp(axis * np.pi)
        return so3_exp((v / s) * np.arctan2(s, c))

    def _map_init(self) -> bool:
        self._frame_num += 1
        xyz = self._meas_lidar[0]
        # transform = sys_init_pose * lidar_imu
        R = self._sys_init_R
        t = self._sys_init_t
        body = (self.TLI_R @ xyz.T).T + self.TLI_t       # lidar → imu
        world = (R @ body.T).T + t                        # imu → world
        self.ivox.insert(world)
        self.kf.last_obs_time = self._meas_lidar[3]
        return self._frame_num > self.map_init_frames

    # ------------------------------------------------------------------
    # Per-scan pipeline (SuperLIO::stateProcess)
    # ------------------------------------------------------------------
    def _state_process(self):
        self._scan_counter += 1
        _t0 = time.perf_counter()   # DIAG: per-scan processing time
        self._skip_map = False      # default: insert this scan into the map
        self._propagation_undistort()
        self._downsample()
        self._register()            # <-- the only variant-specific step
        if self.force_z_zero:       # common 2-D constraint, applied post-update
            self.kf.p[2] = 0.0
            self.kf.v[2] = 0.0
        self._update_map()
        self._output()
        proc_ms = (time.perf_counter() - _t0) * 1000.0
        self._proc_ms.append(proc_ms)
        self.last_proc_ms = proc_ms
        if self._scan_counter % 100 == 0:
            p = self.kf.p
            avg_ms = sum(self._proc_ms) / max(1, len(self._proc_ms))
            self.get_logger().info(
                f"[scan {self._scan_counter}] method={self.last_method} "
                f"conf={self.last_conf:.2f} chi2={self.last_chi2:.1f}  "
                f"proc_ms={proc_ms:.1f}(avg {avg_ms:.1f})  "
                f"p=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}]  cells={len(self.ivox)}  "
                f"p2p_fallbacks={self._n_p2p_fallback}  degen_skips={self._n_degen_skip}  "
                f"lidar_recv={self._n_lidar_recv} (parse_none={self._n_lidar_parse_none}) "
                f"queued={len(self._lidar_buf)}")

    # ---- 1. Propagation + undistortion -------------------------------
    def _propagation_undistort(self):
        kf = self.kf
        kf.current_obs_time = self._meas_lidar[3]
        # propagate_states_: [initial] + state after each IMU predict
        times = [kf.current_time]
        Rs = [kf.R.copy()]
        ps = [kf.p.copy()]
        vs = [kf.v.copy()]
        accs = [np.zeros(3)]
        for acc, gyr, secs in self._meas_imu:
            if kf.predict(acc, gyr, secs):
                t, R, p, v, a = kf._last_dyn
                times.append(t)
                Rs.append(R)
                ps.append(p)
                vs.append(v)
                accs.append(a)

        T_s = np.array(times)
        R_s = np.array(Rs)            # (M,3,3)
        p_s = np.array(ps)            # (M,3)
        v_s = np.array(vs)            # (M,3)
        a_s = np.array(accs)          # (M,3)
        M = T_s.shape[0]

        R_end = kf.R
        R_inv = R_end.T
        p_end = kf.p

        xyz, ot, start, end = self._meas_lidar
        query = start + ot
        # points in imu frame (TLI) once
        imu_pts = (self.TLI_R @ xyz.T).T + self.TLI_t     # (N,3)

        out = np.empty_like(imu_pts)

        if M < 2:
            # no motion info — leave in imu frame
            self._scan_undistort = imu_pts
            return

        # segment index per point: largest s with T_s[s] < query
        seg = np.searchsorted(T_s, query, side="right") - 1
        # points beyond last state → fallback (imu frame, super_lio leaves TLI)
        beyond = query > T_s[-1]
        seg = np.clip(seg, 0, M - 2)

        # precompute per-segment rotation logs (slerp via axis-angle)
        # lv[s] = log(R_s[s]^T R_s[s+1])
        for s in range(M - 1):
            mask = (seg == s) & (~beyond)
            if not np.any(mask):
                continue
            dt = T_s[s + 1] - T_s[s]
            if dt <= 0:
                dt = 1e-9
            tau = query[mask] - T_s[s]
            frac = np.clip(tau / dt, 0.0, 1.0)
            R_h = R_s[s]
            lv = so3_log(R_h.T @ R_s[s + 1])
            ang = float(np.linalg.norm(lv))
            pts = imu_pts[mask]
            if ang < 1e-9:
                rot_pts = (R_h @ pts.T).T
            else:
                axis = lv / ang
                K = _skew(axis)
                th = frac * ang                              # (m,)
                sin_t = np.sin(th)[:, None, None]
                cos_t = np.cos(th)[:, None, None]
                # R_i = R_h @ (I + sinθ K + (1-cosθ)K²)
                Rrel = (np.eye(3)[None] + sin_t * K[None]
                        + (1.0 - cos_t) * (K @ K)[None])      # (m,3,3)
                Ri = R_h[None] @ Rrel                         # (m,3,3)
                rot_pts = np.einsum("mij,mj->mi", Ri, pts)
            p_i = p_s[s] + v_s[s] * tau[:, None] + 0.5 * a_s[s + 1] * (tau[:, None] ** 2)
            world = rot_pts + (p_i - p_end)
            out[mask] = (R_inv @ world.T).T

        if np.any(beyond):
            out[beyond] = imu_pts[beyond]
        self._scan_undistort = out

    # ---- 2. Downsample -----------------------------------------------
    def _downsample(self):
        ds = voxel_downsample(self._scan_undistort, self.voxel_filter_size)
        self._pts_body = ds
        self._pts_len = np.linalg.norm(ds, axis=1)

    # ---- 3. Register — VARIANT-SPECIFIC; subclasses implement this ----
    def _register(self):
        """Update self.kf using the current deskewed scan.  Each benchmark
        variant overrides this with its own registration strategy.  May set
        self.last_method / last_conf / last_chi2 / last_accepted and
        self._skip_map.  The shared building blocks below cover everything a
        variant needs."""
        raise NotImplementedError("subclass must implement _register()")

    # ---- shared registration building blocks -------------------------------
    def _make_scan_cloud(self):
        """Deskewed scan downsampled to gicp_voxel_size, as (pts (N,3), o3d).
        Returns (None, None) if too sparse to register."""
        scan_pts = voxel_downsample(self._scan_undistort, self.gicp_voxel_size)
        if scan_pts.shape[0] < 30:
            return None, None
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(scan_pts.astype(np.float64))
        return scan_pts, cloud

    def _nonrep_init_guess(self, scan_o3d):
        """Non-rep adaptive prediction blended with the IMU prior.
        Returns (feat, pred_pose, pred_conf, T_init(4x4))."""
        feat = self.processor.extract_scan_features(scan_o3d)
        pred_pose, pred_conf = self.processor.predict_pose_adaptive(feat)

        R_imu, p_imu = self.kf.R.copy(), self.kf.p.copy()
        T_init = np.eye(4)
        T_init[:3, :3] = R_imu
        T_init[:3, 3] = p_imu
        if pred_pose is not None and self.processor.get_current_state() is not None:
            R_nr = self._yaw_to_R(float(pred_pose[3]))
            p_nr = pred_pose[:3].copy()
            if self.force_z_zero:
                p_nr[2] = p_imu[2]
            wi = self.imu_base_weight
            wn = float(pred_conf) * self.nonrep_base_weight
            tot = wi + wn
            if tot > 1e-9:
                wi, wn = wi / tot, wn / tot
                T_init[:3, 3] = wi * p_imu + wn * p_nr
                aa = wi * so3_log(R_imu) + wn * so3_log(R_nr)
                T_init[:3, :3] = so3_exp(aa)
        return feat, pred_pose, pred_conf, T_init

    def _get_submap(self):
        """Local OctVoxMap submap around the IMU-propagated position, or None."""
        if len(self.ivox) < self.submap_min_cells:
            return None
        return self.ivox.get_submap(self.kf.p.copy(), self.gicp_submap_radius)

    def _gicp_to_pose(self, scan_o3d, submap, T_init):
        """Run GICP scan→submap and turn the result into an absolute-pose
        measurement.  Returns (R_meas, p_meas, reg_conf, R_n)."""
        T_raw, H = self._gicp(scan_o3d, submap, T_init)
        R_meas = T_raw[:3, :3]
        U, _, Vt = np.linalg.svd(R_meas)
        R_meas = U @ Vt
        p_meas = T_raw[:3, 3].copy()
        reg_conf = estimate_registration_confidence(scan_o3d, submap, T_raw)
        R_n = self._gicp_noise(H, reg_conf)
        return R_meas, p_meas, reg_conf, R_n

    # ------------------------------------------------------------------
    # NDT (Normal Distributions Transform) backend — point-to-distribution,
    # vectorised numpy Gauss-Newton on SE(3).  Used by the *_ndt variants in
    # place of small_gicp.  Returns (T_world, H_6x6[rot,pos]) like _gicp.
    # ------------------------------------------------------------------
    def _build_ndt_model(self, submap_pts):
        """Voxelise the submap (world frame) into per-voxel Gaussians.
        Returns (key->row dict, means(M,3), informations(M,3,3))."""
        res = self.ndt_resolution
        keys = np.floor(submap_pts / res).astype(np.int64)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        ks, ps = keys[order], submap_pts[order]
        change = np.any(np.diff(ks, axis=0) != 0, axis=1)
        bounds = np.concatenate(([0], np.nonzero(change)[0] + 1, [len(ks)]))
        eye = np.eye(3)
        kmap, means, infos = {}, [], []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            if b - a < self.ndt_min_voxel_pts:
                continue
            grp = ps[a:b]
            mu = grp.mean(axis=0)
            cov = np.cov((grp - mu).T)
            w, V = np.linalg.eigh(cov)
            w = np.clip(w, 0.0, None)
            wmax = max(float(w[-1]), 1e-6)
            w = np.maximum(w, 0.001 * wmax)             # NDT eigenvalue clamp
            cov_reg = (V * w) @ V.T
            try:
                info = np.linalg.inv(cov_reg + 1e-6 * eye)
            except np.linalg.LinAlgError:
                continue
            kmap[(int(ks[a, 0]), int(ks[a, 1]), int(ks[a, 2]))] = len(means)
            means.append(mu)
            infos.append(info)
        if not means:
            return {}, None, None
        return kmap, np.asarray(means), np.asarray(infos)

    def _ndt(self, scan_o3d, submap_o3d, init_abs):
        """Register scan(body) → submap(world) via NDT.  Returns (T_world, H)."""
        scan = np.asarray(scan_o3d.points)
        submap = np.asarray(submap_o3d.points)
        kmap, means, infos = self._build_ndt_model(submap)
        T = init_abs.copy()
        if not kmap:
            return T, None
        res = self.ndt_resolution
        H = None
        for _ in range(self.ndt_max_iter):
            tp = scan @ T[:3, :3].T + T[:3, 3]                 # scan in world
            keys = np.floor(tp / res).astype(np.int64)
            rows = np.fromiter(
                (kmap.get((int(k[0]), int(k[1]), int(k[2])), -1) for k in keys),
                dtype=np.int64, count=keys.shape[0])
            m = rows >= 0
            if int(m.sum()) < 10:
                break
            P = tp[m]
            mu = means[rows[m]]
            info = infos[rows[m]]                              # (K,3,3)
            r = P - mu                                         # (K,3)
            K = P.shape[0]
            J = np.zeros((K, 3, 6))
            J[:, 0, 1] = P[:, 2];  J[:, 0, 2] = -P[:, 1]       # -skew(P)
            J[:, 1, 0] = -P[:, 2]; J[:, 1, 2] = P[:, 0]
            J[:, 2, 0] = P[:, 1];  J[:, 2, 1] = -P[:, 0]
            J[:, 0, 3] = 1.0; J[:, 1, 4] = 1.0; J[:, 2, 5] = 1.0
            JT = np.transpose(J, (0, 2, 1))                    # (K,6,3)
            tmp = np.einsum("kab,kbc->kac", JT, info)          # (K,6,3)
            Hm = np.einsum("kac,kcd->ad", tmp, J)              # (6,6)
            bb = np.einsum("kac,kc->a", tmp, r)                # (6,)
            try:
                dx = -np.linalg.solve(Hm + 1e-6 * np.eye(6), bb)
            except np.linalg.LinAlgError:
                break
            dR = so3_exp(dx[0:3])
            newT = np.eye(4)
            newT[:3, :3] = dR @ T[:3, :3]
            newT[:3, 3] = dR @ T[:3, 3] + dx[3:6]
            T = newT
            H = Hm
            if np.linalg.norm(dx) < 1e-4:
                break
        return T, H

    def _ndt_to_pose(self, scan_o3d, submap, T_init):
        """NDT scan→submap as an absolute-pose measurement, mirroring
        _gicp_to_pose.  Returns (R_meas, p_meas, reg_conf, R_n)."""
        T_raw, H = self._ndt(scan_o3d, submap, T_init)
        R_meas = T_raw[:3, :3]
        U, _, Vt = np.linalg.svd(R_meas)
        R_meas = U @ Vt
        p_meas = T_raw[:3, 3].copy()
        reg_conf = estimate_registration_confidence(scan_o3d, submap, T_raw)
        R_n = self._gicp_noise(H, reg_conf)
        return R_meas, p_meas, reg_conf, R_n

    def _feed_processor(self, feat, pred_pose, conf):
        """Feed the corrected pose back to the non-rep processor so its motion
        history / feature database / adaptation stay live (no-op if the variant
        has no processor)."""
        if self.processor is None or feat is None:
            return
        yaw = float(np.arctan2(self.kf.R[1, 0], self.kf.R[0, 0]))
        obs_pose = np.array([self.kf.p[0], self.kf.p[1], self.kf.p[2], yaw])
        self.processor.update_with_observation(obs_pose, feat, conf, pred_pose)

    def _scan_degeneracy(self, pts: np.ndarray) -> bool:
        """Very-simple PCA test for an ill-conditioned scan.

        Returns True when the (body-frame) scan is geometrically degenerate for
        registration, i.e. either:
          * concentrated in a small region — the largest spatial spread
            (std-dev along the dominant axis) is below `degen_min_extent`, or
          * collapsed onto a plane / line — the smallest PCA eigenvalue is below
            `degen_planarity_ratio`× the largest (e.g. robot hugging a wall, so
            motion parallel to it is unobservable).
        """
        n = pts.shape[0]
        if n < self.degen_min_points:
            if self.degen_debug:
                self.get_logger().info(
                    f"[degen] scan {self._scan_counter}: n={n} < min_points="
                    f"{self.degen_min_points} -> DEGEN (too few points)")
            return True
        c = pts - pts.mean(axis=0)
        cov = (c.T @ c) / float(n)
        w = np.clip(np.linalg.eigvalsh(cov), 0.0, None)   # ascending λ0≤λ1≤λ2
        lam_min, lam_max = float(w[0]), float(w[2])
        if lam_max <= 1e-12:
            if self.degen_debug:
                self.get_logger().info(
                    f"[degen] scan {self._scan_counter}: n={n} lam_max~0 -> DEGEN")
            return True
        max_extent = np.sqrt(lam_max)                      # dominant-axis std (m)
        ratio = lam_min / lam_max                          # weakest-direction structure
        small_area = max_extent < self.degen_min_extent
        planar = ratio < self.degen_planarity_ratio
        is_degen = small_area or planar

        if self.degen_debug:
            reason = ("small_area" if small_area and not planar else
                      "planar" if planar and not small_area else
                      "small_area+planar" if is_degen else "ok")
            self.get_logger().info(
                f"[degen] scan {self._scan_counter}: n={n} "
                f"extent={max_extent:.3f}m (min={self.degen_min_extent:.3f}) "
                f"ratio={ratio:.4f} (min={self.degen_planarity_ratio:.4f}) "
                f"lambdas=[{w[0]:.4g},{w[1]:.4g},{w[2]:.4g}] "
                f"-> {'DEGEN' if is_degen else 'OK'} [{reason}]")

        return is_degen

    @staticmethod
    def _yaw_to_R(yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _gicp_noise(self, H, reg_conf: float) -> np.ndarray:
        """6x6 measurement covariance R_n in [rot(3), pos(3)] order for the ESKF
        pose update.

        The small_gicp Gauss-Newton Hessian sums Jᵀ Σ⁻¹ J over *thousands* of
        correspondences, so inv(H) claims millimetre / milli-radian certainty —
        far tighter than the true scan-to-submap uncertainty.  Used raw it makes
        the chi² gate reject almost every (otherwise good) GICP update.  So:
          1. use inv(H) only as the covariance *shape*, scaled by gicp_cov_scale;
          2. widen it when the alignment confidence is low;
          3. clamp every diagonal to a physical floor (meas_noise_rot/pos²) so the
             filter can never be more confident in the registration than that —
             this keeps the chi² gate meaningful (real divergence still rejected)
             without nuking good updates."""
        rot_floor = float(self.meas_noise_rot) ** 2
        pos_floor = float(self.meas_noise_pos) ** 2
        inflate = (1.0 / max(float(reg_conf), 0.05)) ** 2   # poor conf -> looser

        R_n = None
        if H is not None and getattr(H, "shape", None) == (6, 6):
            try:
                cov = np.linalg.inv(H) * self.gicp_cov_scale
                if np.all(np.isfinite(cov)) and np.linalg.eigvalsh(cov).min() > 0.0:
                    R_n = cov
            except np.linalg.LinAlgError:
                R_n = None

        if R_n is None:
            R_n = np.zeros((6, 6))
            R_n[0:3, 0:3] = np.eye(3) * rot_floor
            R_n[3:6, 3:6] = np.eye(3) * pos_floor

        R_n = R_n * inflate

        # Per-axis covariance floor: raise any over-confident diagonal up to the
        # floor (keeps correlations, stays positive-definite).
        floor = np.array([rot_floor] * 3 + [pos_floor] * 3)
        deficit = np.clip(floor - np.diag(R_n), 0.0, None)
        if np.any(deficit > 0.0):
            R_n = R_n + np.diag(deficit)
        return R_n

    # ---- 3b. Point-to-plane fallback (Super-LIO Observe) -------------
    def _observe_point_to_plane(self, weight: float = 1000.0,
                                trans_only: bool = False):
        """Super-LIO's point-to-plane iESKF update against the OctVoxMap.
        `weight` scales the measurement information (lower = down-weighted, for
        fusion).  `trans_only=True` zeros the rotation Jacobian so P2P corrects
        translation only (leaving rotation to the gyro).  Returns (n_corr, rms)."""
        pts_body = self._pts_body
        lengths = self._pts_len
        N = pts_body.shape[0]
        if N == 0:
            return 0, float("nan")
        stats = {"ncorr": 0, "rms": float("nan")}
        cache = {"abcd": None, "knn_valid": None}

        def obs_fn(R, p, need_converge):
            world = (R @ pts_body.T).T + p
            if not need_converge:
                dist, nbr = self.ivox.knn5(world)
                knn_valid = np.isfinite(dist).all(axis=1)
                abcd = np.zeros((N, 4))
                if np.any(knn_valid):
                    pl, ok = self._fit_planes(nbr[knn_valid])
                    idxs = np.where(knn_valid)[0]
                    abcd[idxs] = pl
                    knn_valid[idxs] = ok
                cache["abcd"], cache["knn_valid"] = abcd, knn_valid
            abcd, knn_valid = cache["abcd"], cache["knn_valid"]

            HTVH = np.zeros((6, 6))
            HTVr = np.zeros(6)
            if not np.any(knn_valid):
                return HTVH, HTVr
            nvec = abcd[knn_valid, 0:3]
            d = abcd[knn_valid, 3]
            wv = world[knn_valid]
            err = np.einsum("ij,ij->i", nvec, wv) + d
            lv = lengths[knn_valid]
            keep = lv > (81.0 * err * err)
            if not np.any(keep):
                return HTVH, HTVr
            nvec = nvec[keep]
            err = err[keep]
            pb = pts_body[knn_valid][keep]
            nb = nvec @ R
            Jhead = np.zeros_like(nvec) if trans_only else np.cross(pb, nb)
            J = np.concatenate([Jhead, nvec], axis=1)
            W = float(weight)
            HTVH = J.T @ (W * J)
            HTVr = -(J.T @ (W * err))
            stats["ncorr"] = int(err.shape[0])
            stats["rms"] = float(np.sqrt(np.mean(err * err)))
            return HTVH, HTVr

        self.kf.update_observe(obs_fn)
        return stats["ncorr"], stats["rms"]

    @staticmethod
    def _fit_planes(nbr: np.ndarray):
        """Vectorized calc_plane_coeff over (m,5,3): solve A·x=-1, n=x/|x|,
        d=1/|x|; valid if all 5 points within 0.1 m of the plane."""
        m = nbr.shape[0]
        A = nbr
        b = -np.ones((m, 5, 1))
        AtA = np.einsum("mij,mik->mjk", A, A)
        Atb = np.einsum("mij,mik->mjk", A, b)[:, :, 0]
        AtA += np.eye(3)[None] * 1e-9
        try:
            x = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            x = np.zeros((m, 3))
        norm = np.linalg.norm(x, axis=1)
        ok = norm > 1e-6
        norm_safe = np.where(ok, norm, 1.0)
        d = 1.0 / norm_safe
        nvec = x / norm_safe[:, None]
        abcd = np.concatenate([nvec, d[:, None]], axis=1)
        dd = np.einsum("mkj,mj->mk", nbr, nvec) + d[:, None]
        valid = ok & (np.abs(dd) <= 0.1).all(axis=1)
        return abcd, valid

    @staticmethod
    def _p2p_conf(ncorr: int, rms: float) -> float:
        """Heuristic confidence for the processor feed when P2P was used."""
        if ncorr <= 0 or not np.isfinite(rms):
            return 0.0
        return float(np.clip(1.0 - rms / 0.2, 0.1, 1.0))

    # ---- 4. Update map -----------------------------------------------
    def _update_map(self):
        if self._skip_map:          # degenerate geometry — don't pollute the map
            return
        if self._pts_body.shape[0] == 0:
            return
        R, p = self.kf.R, self.kf.p
        world = (R @ self._pts_body.T).T + p
        self.ivox.insert(world)

    # ---- output ------------------------------------------------------
    def _output(self):
        R, p = self.kf.R, self.kf.p
        stamp = self.kf.current_time
        qx, qy, qz, qw = rot_to_quat(R)

        odom = Odometry()
        odom.header.frame_id = "world"
        odom.header.stamp = self._to_ros_time(stamp)
        odom.pose.pose.position.x = float(p[0])
        odom.pose.pose.position.y = float(p[1])
        odom.pose.pose.position.z = float(p[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(self.kf.v[0])
        odom.twist.twist.linear.y = float(self.kf.v[1])
        odom.twist.twist.linear.z = float(self.kf.v[2])
        self.pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = odom.header.stamp
        tf.header.frame_id = "world"
        tf.child_frame_id = "imu"
        tf.transform.translation.x = float(p[0])
        tf.transform.translation.y = float(p[1])
        tf.transform.translation.z = float(p[2])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

        if self._csv_fh is not None:
            self._csv_fh.write(
                f"{self._scan_counter},{stamp:.9f},"
                f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},"
                f"{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f}\n")
            self._csv_fh.flush()

        if (self.publish_cloud and self.pub_cloud.get_subscription_count() > 0
                and (self._scan_counter % max(self.pub_step, 1) == 0)):
            world = (R @ self._scan_undistort.T).T + p
            self.pub_cloud.publish(self._make_cloud(world, stamp))

    @staticmethod
    def _to_ros_time(t_sec: float):
        from builtin_interfaces.msg import Time
        msg = Time()
        msg.sec = int(np.floor(t_sec))
        msg.nanosec = int((t_sec - msg.sec) * 1e9)
        return msg

    def _make_cloud(self, pts: np.ndarray, stamp: float) -> PointCloud2:
        header = Header()
        header.frame_id = "world"
        header.stamp = self._to_ros_time(stamp)
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = pts.shape[0]
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * pts.shape[0]
        msg.is_dense = True
        msg.data = pts.astype(np.float32).tobytes()
        return msg

    def shutdown(self):
        if self._proc_ms:
            a = np.asarray(self._proc_ms)
            self.get_logger().info(
                "[timing] per-scan processing: "
                f"mean={a.mean():.2f} median={np.median(a):.2f} "
                f"p95={np.percentile(a, 95):.2f} max={a.max():.2f} ms "
                f"over {a.size} scans")
        if self._csv_fh is not None:
            self._csv_fh.close()


def run_node(node_cls):
    """Spin a SuperLioBase subclass.  Each variant's main() calls this.

    Uses a MultiThreadedExecutor so the IMU callback (lightweight buffering) and
    the LiDAR callback (heavy GICP/registration) run on separate threads — IMU
    intake is never blocked by GICP, so propagation data is no longer dropped
    when the per-scan compute can't keep up with real-time playback."""
    import signal
    from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
    rclpy.init()
    node = node_cls()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    node._start_worker()                # heavy processing runs off the executor
    # the benchmark stops nodes with SIGTERM; convert it to a clean shutdown so
    # the timing summary is printed and buffers are flushed (default SIGTERM skips
    # the finally block).
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._stop_worker()
        node.shutdown()                 # variant backend (e.g. PGO) runs here
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():                  # SIGINT may have already shut the context
            rclpy.shutdown()


if __name__ == "__main__":
    print("lio_base.py is an abstract base — run one of the variant nodes:\n"
          "  lio_p2p.py, lio_nonrep_gicp.py, lio_nonrep_gicp_p2p.py, "
          "lio_nonrep_fused_degen.py")
