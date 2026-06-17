#!/usr/bin/env python3.10
"""
ros_lio_v3.py  —  faithful Python port of Super-LIO's Livox-Avia mapping.

Unlike ros_lio.py / ros_lio_v2.py (which fuse a non-rep GICP observation into an
IESKF), this node reproduces the **exact Super-LIO pipeline** for the Livox Avia
as implemented in ../../super_lio (C++):

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
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ros_lio import so3_exp, so3_log, _skew, rot_to_quat, ros_time_to_sec  # noqa: E402


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
    """Faithful behavioural port of super_lio's OctVoxMap for the plane fit.

    One representative point per 0.25 m sub-voxel (resolution/2), updated as a
    running mean of points landing within 0.1 m of the stored point (max 20),
    exactly like OctVox::AddPoint.  getTopK is served by a cKDTree rebuilt over
    the representative points (k=5)."""

    MERGE_DIST2 = 0.1 * 0.1
    MAX_PER_SUBVOX = 20

    def __init__(self, resolution: float = 0.5, capacity: int = 2_000_000):
        self.sub_res = resolution / 2.0
        self.inv_sub = 1.0 / self.sub_res
        self.capacity = capacity
        # cell key (i,j,k) -> [mean(3), count]
        self._cells: "collections.OrderedDict" = collections.OrderedDict()
        self._tree: Optional[cKDTree] = None
        self._tree_pts: Optional[np.ndarray] = None
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

    def build_tree(self) -> None:
        if not self._dirty and self._tree is not None:
            return
        if len(self._cells) == 0:
            self._tree, self._tree_pts = None, None
            self._dirty = False
            return
        self._tree_pts = np.array([v[0] for v in self._cells.values()], dtype=float)
        self._tree = cKDTree(self._tree_pts)
        self._dirty = False

    def knn5(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (dist (N,5), neighbor_pts (N,5,3)).  Missing → dist=inf."""
        self.build_tree()
        if self._tree is None or self._tree_pts.shape[0] < 4:
            n = query.shape[0]
            return np.full((n, 5), np.inf), np.zeros((n, 5, 3))
        kk = min(5, self._tree_pts.shape[0])
        dist, idx = self._tree.query(query, k=kk)
        if kk == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        if kk < 5:  # pad to 5 with inf
            pad = 5 - kk
            dist = np.concatenate([dist, np.full((dist.shape[0], pad), np.inf)], axis=1)
            idx = np.concatenate([idx, np.zeros((idx.shape[0], pad), dtype=idx.dtype)], axis=1)
        nbr = self._tree_pts[idx]
        return dist, nbr


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


# =============================================================================
# SuperLIO node
# =============================================================================
class SuperLioV3(Node):
    def __init__(self):
        super().__init__("super_lio_v3")

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

        # ---- filter / map --------------------------------------------------
        self.kf = ESKF(self.gravity_norm)
        self.kf.num_iterations = self.kf_max_iterations
        self.kf.quit_eps = self.kf_quit_eps
        self.ivox = OctVoxMapPy(self.vox_resolution, self.hash_capacity)

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

        # downsampled body points of the current scan (reused by Observe/UpdateMap)
        self._pts_body = np.zeros((0, 3))
        self._pts_len = np.zeros(0)
        self.last_ncorr = 0
        self.last_rms = 0.0

        # ---- ROS I/O -------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=500)
        self.create_subscription(Imu, self.imu_topic, self.cb_imu, sensor_qos)
        self.create_subscription(PointCloud2, self.lidar_topic, self.cb_lidar, sensor_qos)
        self.pub_odom = self.create_publisher(Odometry, "/lio/odom", 100)
        self.pub_cloud = self.create_publisher(PointCloud2, "/lio/cloud_world", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self._csv_fh = None
        if self.debug_csv:
            self._csv_fh = open(self.debug_csv, "w")
            self._csv_fh.write("scan,stamp,x,y,z,qx,qy,qz,qw\n")
            self._csv_fh.flush()

        self.get_logger().info("=== Super-LIO v3 (faithful Livox-Avia port) ===")
        self.get_logger().info(f"  lidar={self.lidar_topic}  imu={self.imu_topic}")
        self.get_logger().info(f"  gravity_norm={self.gravity_norm}  voxel_filter={self.voxel_filter_size}")
        self.get_logger().info(f"  vox_resolution={self.vox_resolution}  kf_iters={self.kf_max_iterations}")
        self.get_logger().info(f"  filter_rate={self.filter_rate}  blind={np.sqrt(self.blind2):.2f}")
        if self.debug_csv:
            self.get_logger().info(f"  debug_csv → {self.debug_csv}")

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
        if secs < self._last_imu_stamp:
            self._imu_buf.clear()
        self._imu_buf.append((acc, gyr, secs))
        self._last_imu_stamp = secs
        self._drain()

    def cb_lidar(self, msg: PointCloud2):
        scan = self._parse_cloud(msg)
        if scan is None:
            return
        self._lidar_buf.append(scan)
        self._drain()

    def _parse_cloud(self, msg: PointCloud2):
        """Extract (xyz, offset_time[s], start_time, end_time) — super_lio
        stdMsgHandler LIVOX_PC2 case: filter by tag, blind/maxrange, filter_rate."""
        n = msg.width * msg.height
        if n < 10:
            return None
        fo = {f.name: f for f in msg.fields}
        if "offset_time" not in fo:
            return None
        step = msg.point_step
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, step)

        def col_f32(off):
            return raw[:, off:off + 4].copy().ravel().view(np.float32).astype(np.float64)

        x = col_f32(fo["x"].offset)
        y = col_f32(fo["y"].offset)
        z = col_f32(fo["z"].offset)
        ot = raw[:, fo["offset_time"].offset:fo["offset_time"].offset + 4].copy().ravel().view(np.uint32)
        ot = ot.astype(np.float64) * 1e-9

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

    def _drain(self):
        while self._sync_measure():
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
        self._propagation_undistort()
        self._downsample()
        self._observe()
        self._update_map()
        self._output()
        if self._scan_counter % 100 == 0:
            p = self.kf.p
            self.get_logger().info(
                f"[scan {self._scan_counter}] corr={self.last_ncorr} "
                f"rms={self.last_rms:.4f}m  p=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}]  "
                f"cells={len(self.ivox)}")

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

    # ---- 3. Observe (point-to-plane iESKF) ---------------------------
    def _observe(self):
        pts_body = self._pts_body
        lengths = self._pts_len
        N = pts_body.shape[0]
        if N == 0:
            return
        self.ivox.build_tree()

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
                cache["abcd"] = abcd
                cache["knn_valid"] = knn_valid
            abcd = cache["abcd"]
            knn_valid = cache["knn_valid"]

            HTVH = np.zeros((6, 6))
            HTVr = np.zeros(6)
            if not np.any(knn_valid):
                return HTVH, HTVr

            nvec = abcd[knn_valid, 0:3]
            d = abcd[knn_valid, 3]
            wv = world[knn_valid]
            err = np.einsum("ij,ij->i", nvec, wv) + d         # n·x + d
            lv = lengths[knn_valid]
            keep = lv > (81.0 * err * err)                    # compute_error gate
            if not np.any(keep):
                return HTVH, HTVr
            nvec = nvec[keep]
            err = err[keep]
            pb = pts_body[knn_valid][keep]

            nb = nvec @ R                                     # (Rᵀn)ᵀ rows
            Jhead = np.cross(pb, nb)                          # p_body × (Rᵀn)
            J = np.concatenate([Jhead, nvec], axis=1)         # (m,6) [rot|pos]
            W = 1000.0
            HTVH = J.T @ (W * J)
            HTVr = -(J.T @ (W * err))
            self.last_ncorr = int(err.shape[0])
            self.last_rms = float(np.sqrt(np.mean(err * err)))
            return HTVH, HTVr

        self.kf.update_observe(obs_fn)

    @staticmethod
    def _fit_planes(nbr: np.ndarray):
        """Vectorized calc_plane_coeff over (m,5,3) neighbor sets.
        Solve A·x = -1, n = x/|x|, d = 1/|x|; valid if all 5 within 0.1 m."""
        m = nbr.shape[0]
        A = nbr                                              # (m,5,3)
        b = -np.ones((m, 5, 1))
        AtA = np.einsum("mij,mik->mjk", A, A)                # (m,3,3)
        Atb = np.einsum("mij,mik->mjk", A, b)[:, :, 0]       # (m,3)
        AtA += np.eye(3)[None] * 1e-9
        try:
            x = np.linalg.solve(AtA, Atb)                    # (m,3)
        except np.linalg.LinAlgError:
            x = np.zeros((m, 3))
        norm = np.linalg.norm(x, axis=1)
        ok = norm > 1e-6
        norm_safe = np.where(ok, norm, 1.0)
        d = 1.0 / norm_safe
        nvec = x / norm_safe[:, None]
        abcd = np.concatenate([nvec, d[:, None]], axis=1)    # (m,4)
        # validity: |n·p + d| <= 0.1 for all 5
        dd = np.einsum("mkj,mj->mk", nbr, nvec) + d[:, None]  # (m,5)
        valid = ok & (np.abs(dd) <= 0.1).all(axis=1)
        return abcd, valid

    # ---- 4. Update map -----------------------------------------------
    def _update_map(self):
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
        if self._csv_fh is not None:
            self._csv_fh.close()


def main():
    rclpy.init()
    node = SuperLioV3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
