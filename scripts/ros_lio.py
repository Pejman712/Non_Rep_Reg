#!/usr/bin/env python3
"""
ROS2 LiDAR-Inertial Odometry (LIO) — nonrep_lio branch

Architecture:
  1. IMU messages are buffered with timestamps.
  2. On each LiDAR scan, buffered IMU is pre-integrated over [t_prev_scan, t_scan]
     to obtain a relative transform T_imu (δR from gyro + δp from accel).
  3. T_imu is supplied as the initial guess to GICP, replacing a cold-start at identity.
  4. The GICP-refined transform is used for the full 6-DOF state update.
  5. Velocity is tracked for the Odometry twist field.

When use_imu: false the node falls back to the same behaviour as ros_non_rep.py.
Gravity is estimated from the first imu_gravity_init_n static IMU samples.

Subscribes:
  lidar_topic   sensor_msgs/PointCloud2
  imu_topic     sensor_msgs/Imu

Publishes:
  odom_topic    nav_msgs/Odometry   (full 6-DOF quaternion + linear velocity)
  map_topic     sensor_msgs/PointCloud2
"""

import collections
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros


# =============================================================================
# Data model (unchanged from ros_non_rep.py)
# =============================================================================
@dataclass
class ScanState:
    pose: np.ndarray        # [x, y, z, yaw]
    uncertainty: np.ndarray # 4×4 covariance
    confidence: float
    scan_features: Dict


# =============================================================================
# NonRepetitiveLiDARProcessor (unchanged — drives adaptive weights + confidence)
# =============================================================================
class NonRepetitiveLiDARProcessor:
    def __init__(self,
                 adaptive_threshold: float = 0.9,
                 feature_weight: float = 0.3,
                 geometric_weight: float = 0.4,
                 temporal_weight: float = 0.3,
                 force_z_zero: bool = False,
                 z_redistribution_method: str = 'prediction'):
        self.adaptive_threshold = adaptive_threshold
        self.feature_weight = feature_weight
        self.geometric_weight = geometric_weight
        self.temporal_weight = temporal_weight
        self.force_z_zero = force_z_zero
        self.z_redistribution_method = z_redistribution_method

        self.scan_states: List[ScanState] = []
        self.feature_database: List[Dict] = []
        self.motion_patterns: List[str] = []

        self.voxel_size = 0.1
        self.normal_radius = 0.5
        self.fpfh_radius = 1.0

    def redistribute_z_component(self, pose: np.ndarray, predicted_pose: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.force_z_zero or abs(pose[2]) < 1e-6:
            return pose.copy()
        out = pose.copy()
        out[2] = 0.0
        return out

    def extract_scan_features(self, cloud: o3d.geometry.PointCloud) -> Dict:
        features: Dict = {}
        try:
            points = np.asarray(cloud.points)
            if len(points) == 0:
                return features

            features['point_count'] = int(len(points))
            features['centroid'] = np.mean(points, axis=0)
            features['std_dev'] = np.std(points, axis=0)
            features['bounding_box'] = {
                'min': np.min(points, axis=0),
                'max': np.max(points, axis=0),
                'extent': np.max(points, axis=0) - np.min(points, axis=0),
            }

            cloud_ds = cloud.voxel_down_sample(self.voxel_size) if len(points) > 1000 else cloud

            if len(cloud_ds.points) > 10:
                cloud_ds.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30)
                )
                normals = np.asarray(cloud_ds.normals)
                if len(normals) > 0:
                    features['normal_distribution'] = {
                        'mean': np.mean(normals, axis=0),
                        'std': np.std(normals, axis=0),
                    }

            if len(cloud_ds.points) > 50 and cloud_ds.has_normals():
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    cloud_ds,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=self.fpfh_radius, max_nn=100),
                )
                features['fpfh_histogram'] = np.asarray(fpfh.data).mean(axis=1)

            if len(cloud_ds.points) > 100:
                plane_model, inliers = cloud_ds.segment_plane(
                    distance_threshold=0.1, ransac_n=3, num_iterations=1000
                )
                if len(inliers) > 50:
                    features['dominant_plane'] = {
                        'normal': plane_model[:3],
                        'distance': plane_model[3],
                        'inlier_ratio': float(len(inliers) / len(cloud_ds.points)),
                    }

            z = points[:, 2]
            features['height_profile'] = {
                'min_height': float(np.min(z)),
                'max_height': float(np.max(z)),
                'mean_height': float(np.mean(z)),
                'height_variance': float(np.var(z)),
            }

            if len(points) > 100:
                sample_idx = np.random.choice(len(points), min(100, len(points)), replace=False)
                sample_pts = points[sample_idx]
                distances = cdist(sample_pts, points)
                k_nearest = np.sort(distances, axis=1)[:, 1:6]
                features['local_density'] = float(np.mean(k_nearest))

            if len(points) > 20:
                pca = PCA(n_components=3)
                pca.fit(points)
                features['shape_complexity'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'linearity': float(pca.explained_variance_ratio_[0]),
                    'planarity': float(pca.explained_variance_ratio_[1]),
                    'sphericity': float(pca.explained_variance_ratio_[2]),
                }

        except Exception as e:
            features['extraction_error'] = str(e)

        return features

    def compute_feature_similarity(self, f1: Dict, f2: Dict) -> float:
        if not f1 or not f2:
            return 0.0
        sims = []
        try:
            if 'point_count' in f1 and 'point_count' in f2:
                a, b = float(f1['point_count']), float(f2['point_count'])
                sims.append(min(a, b) / max(a, b, 1e-12))
            if 'centroid' in f1 and 'centroid' in f2:
                d = float(np.linalg.norm(f1['centroid'][:2] - f2['centroid'][:2])
                          if self.force_z_zero
                          else np.linalg.norm(f1['centroid'] - f2['centroid']))
                sims.append(max(0.0, 1.0 - d / 50.0))
            if 'bounding_box' in f1 and 'bounding_box' in f2:
                bb1, bb2 = f1['bounding_box'], f2['bounding_box']
                e1 = bb1['extent'][:2] if self.force_z_zero else bb1['extent']
                e2 = bb2['extent'][:2] if self.force_z_zero else bb2['extent']
                denom = float(np.prod(np.maximum(e1, e2)))
                if denom > 0:
                    sims.append(float(np.prod(np.minimum(e1, e2)) / denom))
            if 'fpfh_histogram' in f1 and 'fpfh_histogram' in f2:
                h1, h2 = f1['fpfh_histogram'], f2['fpfh_histogram']
                if len(h1) == len(h2):
                    dot = float(np.dot(h1, h2))
                    norm = float(np.linalg.norm(h1) * np.linalg.norm(h2))
                    if norm > 1e-12:
                        sims.append(max(0.0, dot / norm))
            if 'height_profile' in f1 and 'height_profile' in f2:
                hp1, hp2 = f1['height_profile'], f2['height_profile']
                if not self.force_z_zero:
                    r1 = float(hp1['max_height'] - hp1['min_height'])
                    r2 = float(hp2['max_height'] - hp2['min_height'])
                    sims.append(min(r1, r2) / max(r1, r2, 1e-12))
                else:
                    v1, v2 = float(hp1['height_variance']), float(hp2['height_variance'])
                    sims.append(min(v1, v2) / max(v1, v2, 1e-12))
            if 'local_density' in f1 and 'local_density' in f2:
                d1, d2 = float(f1['local_density']), float(f2['local_density'])
                sims.append(min(d1, d2) / max(d1, d2, 1e-12))
        except Exception as e:
            print(f"Error computing similarity: {e}")
        return float(np.mean(sims)) if sims else 0.0

    def update_with_observation(self,
                                observed_pose: np.ndarray,
                                scan_features: Dict,
                                registration_confidence: float = 1.0,
                                predicted_pose: Optional[np.ndarray] = None):
        final_pose = self.redistribute_z_component(observed_pose, predicted_pose)
        base_unc = 0.1
        unc = np.eye(4) * (base_unc / max(registration_confidence, 1e-6)) ** 2
        self.scan_states.append(ScanState(
            pose=final_pose.copy(),
            uncertainty=unc,
            confidence=float(registration_confidence),
            scan_features=scan_features,
        ))
        self.feature_database.append(scan_features)
        max_hist = 20
        if len(self.scan_states) > max_hist:
            self.scan_states.pop(0)
            self.feature_database.pop(0)
        self._analyze_motion_patterns()

    def _analyze_motion_patterns(self):
        if len(self.scan_states) < 5:
            return
        recent = [s.pose for s in self.scan_states[-5:]]
        moves = []
        for i in range(1, len(recent)):
            moves.append(float(np.linalg.norm(recent[i][:2] - recent[i - 1][:2])
                               if self.force_z_zero
                               else np.linalg.norm(recent[i][:3] - recent[i - 1][:3])))
        avg = float(np.mean(moves))
        std = float(np.std(moves))
        if avg < 0.1:
            pattern = "stationary"
        elif std / (avg + 1e-6) < 0.3:
            pattern = "smooth"
        elif std / (avg + 1e-6) > 1.0:
            pattern = "erratic"
        else:
            pattern = "variable"
        self.motion_patterns.append(pattern)
        if len(self.motion_patterns) > 10:
            self.motion_patterns.pop(0)
        if pattern == "erratic":
            self.feature_weight, self.geometric_weight, self.temporal_weight = 0.5, 0.2, 0.3
        elif pattern == "smooth":
            self.feature_weight, self.geometric_weight, self.temporal_weight = 0.2, 0.5, 0.3
        else:
            self.feature_weight, self.geometric_weight, self.temporal_weight = 0.35, 0.35, 0.3

    def get_current_state(self) -> Optional[ScanState]:
        return self.scan_states[-1] if self.scan_states else None


# =============================================================================
# IMU Pre-integrator
# =============================================================================
def so3_exp(omega_dt: np.ndarray) -> np.ndarray:
    """Axis-angle → rotation matrix via Rodrigues' formula."""
    angle = float(np.linalg.norm(omega_dt))
    K = np.array([[0.0, -omega_dt[2], omega_dt[1]],
                  [omega_dt[2], 0.0, -omega_dt[0]],
                  [-omega_dt[1], omega_dt[0], 0.0]])
    if angle < 1e-9:
        return np.eye(3) + K
    axis_K = K / angle
    return np.eye(3) + np.sin(angle) * axis_K + (1.0 - np.cos(angle)) * (axis_K @ axis_K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle vector (inverse of so3_exp)."""
    cos_angle = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if abs(angle) < 1e-9:
        return np.zeros(3)
    return (angle / (2.0 * np.sin(angle))) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


class ImuPreintegrator:
    """
    Simple IMU pre-integration between consecutive LiDAR timestamps.

    Gravity estimation:
      The first `gravity_init_n` IMU samples (during a static period) are averaged
      in the world frame to estimate the effective gravity vector g_eff.
      Thereafter: a_robot_world = R_world @ a_body - g_eff

    Integration state (reset between scans):
      delta_R    : 3×3  body rotation from t1 to current (B1→Bk convention)
      dp_world   : 3    position change p2−p1 in world frame
      dv_world   : 3    velocity change in world frame (used for dp integration)
    """

    def __init__(self, gravity_mag: float = 9.81, gravity_init_n: int = 100):
        # Gravity represented as R_world @ a_imu_static (≈ +9.81 on z-up IMU)
        self.g_eff = np.array([0.0, 0.0, gravity_mag])  # default until estimated
        self.gravity_init_n = gravity_init_n
        self.gravity_initialized = False
        self._gravity_samples: List[np.ndarray] = []

        # Biases (can be configured externally)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

        # Pre-integrated values (reset per scan interval)
        self.delta_R = np.eye(3)
        self.dp_world = np.zeros(3)
        self.dv_world = np.zeros(3)
        self.dt_total = 0.0

    def collect_gravity_sample(self, accel_body: np.ndarray, R_world: np.ndarray):
        """Call for each IMU message during the static initialization phase."""
        if self.gravity_initialized:
            return
        a_corrected = accel_body - self.accel_bias
        self._gravity_samples.append(R_world @ a_corrected)
        if len(self._gravity_samples) >= self.gravity_init_n:
            self.g_eff = np.mean(self._gravity_samples, axis=0)
            self.gravity_initialized = True

    def reset(self):
        """Reset integration state for a new scan interval."""
        self.delta_R = np.eye(3)
        self.dp_world = np.zeros(3)
        self.dv_world = np.zeros(3)
        self.dt_total = 0.0

    def integrate(self, omega: np.ndarray, accel: np.ndarray, dt: float,
                  R_world_at_t1: np.ndarray, use_accel: bool = True):
        """
        Integrate one IMU measurement.

        omega          : angular velocity in body frame [rad/s]
        accel          : linear acceleration in body frame [m/s²]
        dt             : time step [s]
        R_world_at_t1  : world rotation at the *start* of the scan interval (B1→W)
        use_accel      : if False only gyro is integrated (gyro-only initial guess)
        """
        if dt <= 0.0 or dt > 1.0:
            return

        omega_c = omega - self.gyro_bias
        dR = so3_exp(omega_c * dt)

        if use_accel and self.gravity_initialized:
            accel_c = accel - self.accel_bias
            # Rotation of the body at this integration step
            R_body_k = R_world_at_t1 @ self.delta_R  # R_W_Bk
            a_world = R_body_k @ accel_c - self.g_eff
            self.dp_world += self.dv_world * dt + 0.5 * a_world * dt * dt
            self.dv_world += a_world * dt

        self.delta_R = self.delta_R @ dR
        self.dt_total += dt

    def get_delta_transform(self, R_world_at_t1: np.ndarray,
                            use_translation: bool = True) -> np.ndarray:
        """
        Build 4×4 T_B1_B2 (initial guess for GICP).

        T_B1_B2 maps points from the current scan frame (B2) into the previous
        scan frame (B1), which is the convention expected by apply_gicp_direct /
        apply_gicp_with_init (T_target_source where target=prev, source=current).

        delta_R   = R_B1_B2  (gyro integration)
        dp_body   = R_W_B1^T @ dp_world  (position change expressed in B1)
        """
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.delta_R
        if use_translation and self.gravity_initialized:
            T[:3, 3] = R_world_at_t1.T @ self.dp_world
        return T

    @property
    def is_gyro_ready(self) -> bool:
        return True  # gyro always available (no init needed)

    @property
    def is_accel_ready(self) -> bool:
        return self.gravity_initialized


# =============================================================================
# Helper functions
# =============================================================================
def ros_time_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """3×3 rotation matrix → quaternion (x, y, z, w) via Shepperd's method."""
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def estimate_registration_confidence(cloud1, cloud2, transformation, sample_size: int = 100) -> float:
    try:
        from scipy.spatial import cKDTree
        pts1 = np.asarray(cloud1.points)
        pts2 = np.asarray(cloud2.points)
        if len(pts1) == 0 or len(pts2) == 0:
            return 0.1
        idx = np.random.choice(len(pts1), min(sample_size, len(pts1)), replace=False)
        sp_h = np.column_stack([pts1[idx], np.ones(len(idx))])
        tp = (transformation @ sp_h.T).T[:, :3]
        d, _ = cKDTree(pts2).query(tp)
        return float(max(0.1, min(1.0, 1.0 - float(np.mean(d)) / 2.0)))
    except Exception:
        return 0.5


def _normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    if intensity.size == 0:
        return intensity.astype(np.float32, copy=False)
    inten = intensity.astype(np.float32, copy=False)
    m = float(np.nanmax(inten)) if np.isfinite(inten).any() else 0.0
    if m <= 0.0:
        return np.zeros_like(inten, dtype=np.float32)
    if m <= 1.5:
        return np.clip(inten, 0.0, 1.0).astype(np.float32, copy=False)
    return np.clip(inten / m, 0.0, 1.0).astype(np.float32, copy=False)


# =============================================================================
# ROS2 ↔ Open3D conversions (unchanged)
# =============================================================================
def pointcloud2_to_xyz_i(msg: PointCloud2) -> Tuple[np.ndarray, np.ndarray]:
    field_names = [f.name for f in msg.fields]
    has_intensity = "intensity" in field_names
    pts, intens = [], []
    if has_intensity:
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
            intens.append(p[3])
    else:
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pts.append([p[0], p[1], p[2]])
        intens = [0.0] * len(pts)
    if not pts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32), np.asarray(intens, dtype=np.float32)


def xyzi_to_open3d_cloud(xyz: np.ndarray, intensity: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    if xyz.size == 0:
        return cloud
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    inten01 = _normalize_intensity(intensity)
    if inten01.size == xyz.shape[0]:
        colors = np.stack([inten01, inten01, inten01], axis=1).astype(np.float64, copy=False)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def open3d_cloud_to_pointcloud2_xyzi(cloud: o3d.geometry.PointCloud, header: Header) -> PointCloud2:
    pts = np.asarray(cloud.points)
    fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    if pts.size == 0:
        return pc2.create_cloud(header, fields, [])
    intensity = (np.asarray(cloud.colors)[:, 0].astype(np.float32, copy=False)
                 if cloud.has_colors() else np.zeros(pts.shape[0], dtype=np.float32))
    pts32 = pts.astype(np.float32, copy=False)
    data = [(float(p[0]), float(p[1]), float(p[2]), float(i)) for p, i in zip(pts32, intensity)]
    return pc2.create_cloud(header, fields, data)


# =============================================================================
# Open3D visualizer (unchanged)
# =============================================================================
class LiveOpen3D:
    def __init__(self, window_name: str = "LIO Map", width: int = 1400, height: int = 900):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=width, height=height)
        self.latest = o3d.geometry.PointCloud()
        self.map = o3d.geometry.PointCloud()
        self._latest_added = False
        self._map_added = False

    def update(self, latest_cloud: Optional[o3d.geometry.PointCloud],
               map_cloud: Optional[o3d.geometry.PointCloud]):
        if latest_cloud is not None:
            self.latest = latest_cloud
            if not self._latest_added:
                self.vis.add_geometry(self.latest)
                self._latest_added = True
            else:
                self.vis.update_geometry(self.latest)
        if map_cloud is not None:
            self.map = map_cloud
            if not self._map_added:
                self.vis.add_geometry(self.map)
                self._map_added = True
            else:
                self.vis.update_geometry(self.map)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


# =============================================================================
# GICP (Open3D fallback with init_T support)
# =============================================================================
def apply_gicp_open3d(source: o3d.geometry.PointCloud,
                      target: o3d.geometry.PointCloud,
                      init_T: Optional[np.ndarray] = None,
                      voxel_size: float = 0.2,
                      max_corr_distance: float = 2.0,
                      max_iterations: int = 50) -> np.ndarray:
    if len(source.points) < 30 or len(target.points) < 30:
        return np.eye(4, dtype=float)
    if init_T is None:
        init_T = np.eye(4, dtype=float)
    src = source.voxel_down_sample(float(voxel_size)) if voxel_size > 0 else source
    tgt = target.voxel_down_sample(float(voxel_size)) if voxel_size > 0 else target
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    res = o3d.pipelines.registration.registration_generalized_icp(
        src, tgt,
        max_correspondence_distance=float(max_corr_distance),
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iterations)),
    )
    return res.transformation


# =============================================================================
# LIO ROS2 Node
# =============================================================================
class LioNode(Node):
    def __init__(self):
        super().__init__("lio_node")

        def p(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # ---- Input / ROS
        self.lidar_topic = str(p("lidar_topic", "/livox/points"))
        self.queue_size = int(p("queue_size", 10))

        # ---- IMU parameters
        self.use_imu = bool(p("use_imu", True))
        self.imu_topic = str(p("imu_topic", "/imu/data"))
        self.imu_gravity_mag = float(p("imu_gravity_mag", 9.81))
        self.imu_gravity_init_n = int(p("imu_gravity_init_n", 100))
        self.imu_use_accel = bool(p("imu_use_accel", True))
        self.imu_timeout_sec = float(p("imu_timeout_sec", 0.5))
        gyro_bias_raw = p("imu_gyro_bias", [0.0, 0.0, 0.0])
        accel_bias_raw = p("imu_accel_bias", [0.0, 0.0, 0.0])

        # ---- Publishing
        self.publish_odom = bool(p("publish_odom", True))
        self.odom_topic = str(p("odom_topic", "/lio/odom"))
        self.publish_map = bool(p("publish_map", True))
        self.map_topic = str(p("map_topic", "/lio/map"))
        self.publish_tf = bool(p("publish_tf", False))
        self.map_frame = str(p("map_frame", "map"))
        self.base_frame = str(p("base_frame", "base_link"))
        self.map_publish_voxel = float(p("map_publish_voxel", 0.15))
        self.map_publish_max_points = int(p("map_publish_max_points", 800_000))
        self.map_publish_every_n_scans = max(1, int(p("map_publish_every_n_scans", 1)))

        # ---- Scan processing
        self.step_decimation = max(1, int(p("step_decimation", 1)))
        max_scans = int(p("max_scans", -1))
        self.max_scans: Optional[int] = None if max_scans < 0 else max_scans
        self.accumulate_between_decimation = bool(p("accumulate_between_decimation", False))
        acc_vox = float(p("accumulate_voxel", 0.1))
        self.accumulate_voxel: Optional[float] = None if acc_vox < 0 else acc_vox
        self.accumulate_max_points = int(p("accumulate_max_points", 1_500_000))

        # ---- Z handling
        self.force_z_zero = bool(p("force_z_zero", False))

        # ---- Visualization / map
        self.visualize = bool(p("visualize", True))
        self.map_voxel = float(p("map_voxel", 0.15))

        # ---- GICP
        self.use_pctools_gicp = bool(p("use_pctools_gicp", True))
        self.gicp_max_corr_distance = float(p("gicp_max_corr_distance", 2.0))
        self.gicp_voxel_size = float(p("gicp_voxel_size", 0.2))
        self.gicp_max_iterations = int(p("gicp_max_iterations", 50))

        # ---- Fusion weights (initial-guess blending)
        # Final weight = base_weight × confidence_score, then normalised.
        self.imu_base_weight = float(p("imu_base_weight", 0.7))
        self.nonrep_base_weight = float(p("nonrep_base_weight", 0.3))

        # ---- Processor (non-rep adaptive predictor + confidence tracking)
        self.processor = NonRepetitiveLiDARProcessor(force_z_zero=self.force_z_zero)

        # ---- IMU pre-integrator
        self.integrator = ImuPreintegrator(
            gravity_mag=self.imu_gravity_mag,
            gravity_init_n=self.imu_gravity_init_n,
        )
        self.integrator.gyro_bias = np.asarray(gyro_bias_raw, dtype=float)
        self.integrator.accel_bias = np.asarray(accel_bias_raw, dtype=float)

        # IMU message buffer: (stamp_sec, omega, accel)
        self.imu_buffer: Deque[Tuple[float, np.ndarray, np.ndarray]] = collections.deque(maxlen=2000)
        self.last_imu_stamp_sec: Optional[float] = None

        # ---- 6-DOF world state
        self.R_world = np.eye(3, dtype=float)  # rotation: body → world
        self.p_world = np.zeros(3, dtype=float) # position in world frame
        self.v_world = np.zeros(3, dtype=float) # velocity in world frame

        # ---- Scan processing state
        self.prev_cloud: Optional[o3d.geometry.PointCloud] = None
        self.map_cloud = o3d.geometry.PointCloud()
        self._buffer_cloud = o3d.geometry.PointCloud()
        self.msg_counter = 0
        self.scan_counter = 0
        self.last_scan_stamp_sec: Optional[float] = None

        # ---- GICP function (with init_T support)
        self.apply_gicp_func = self._resolve_gicp()

        # ---- Visualizer
        self.viewer = LiveOpen3D(
            window_name="LIO Map (IMU + GICP)"
        ) if self.visualize else None

        # ---- Publishers
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10) if self.publish_odom else None
        self.map_pub = self.create_publisher(PointCloud2, self.map_topic, 1) if self.publish_map else None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.publish_tf else None

        # ---- Subscribers
        self.lidar_sub = self.create_subscription(
            PointCloud2, self.lidar_topic, self.cb_cloud, self.queue_size)
        if self.use_imu:
            self.imu_sub = self.create_subscription(
                Imu, self.imu_topic, self.cb_imu, 200)

        self.get_logger().info("=== LIO Node (Non-Rep prediction + IMU pre-integration + GICP) ===")
        self.get_logger().info(f"lidar={self.lidar_topic}  use_imu={self.use_imu}")
        self.get_logger().info(
            f"fusion weights — imu_base={self.imu_base_weight}  "
            f"nonrep_base={self.nonrep_base_weight}"
        )
        if self.use_imu:
            self.get_logger().info(
                f"imu={self.imu_topic}  use_accel={self.imu_use_accel}  "
                f"gravity_init_n={self.imu_gravity_init_n}"
            )

    # -------------------------------------------------------------------------
    # GICP resolver
    # -------------------------------------------------------------------------
    def _resolve_gicp(self):
        """Return a callable: (src, tgt, init_T) → 4×4 transform."""
        if self.use_pctools_gicp:
            try:
                from Pctools import apply_gicp_with_init
                self.get_logger().info("Using Pctools.apply_gicp_with_init")
                return apply_gicp_with_init
            except Exception as e:
                self.get_logger().warn(f"Pctools.apply_gicp_with_init unavailable: {e}")
                try:
                    from Pctools import apply_gicp_direct
                    # Wrap to accept but ignore init_T (no init support)
                    def _wrap(src, tgt, init_T=None):
                        return apply_gicp_direct(src, tgt)
                    self.get_logger().warn("Falling back to apply_gicp_direct (no init_T)")
                    return _wrap
                except Exception as e2:
                    self.get_logger().warn(f"Pctools import failed: {e2}. Using Open3D fallback.")

        def _o3d_gicp(src, tgt, init_T=None):
            return apply_gicp_open3d(
                src, tgt,
                init_T=init_T,
                voxel_size=self.gicp_voxel_size,
                max_corr_distance=self.gicp_max_corr_distance,
                max_iterations=self.gicp_max_iterations,
            )
        self.get_logger().info("Using Open3D GICP")
        return _o3d_gicp

    # -------------------------------------------------------------------------
    # IMU callbacks + pre-integration helpers
    # -------------------------------------------------------------------------
    def cb_imu(self, msg: Imu):
        stamp_sec = ros_time_to_sec(msg.header.stamp)
        omega = np.array([msg.angular_velocity.x,
                          msg.angular_velocity.y,
                          msg.angular_velocity.z], dtype=float)
        accel = np.array([msg.linear_acceleration.x,
                          msg.linear_acceleration.y,
                          msg.linear_acceleration.z], dtype=float)

        # Gravity initialization from static period (before first LiDAR scan)
        if not self.integrator.gravity_initialized:
            self.integrator.collect_gravity_sample(accel, self.R_world)
            if self.integrator.gravity_initialized:
                self.get_logger().info(
                    f"Gravity estimated: {self.integrator.g_eff}  "
                    f"norm={np.linalg.norm(self.integrator.g_eff):.3f} m/s²"
                )

        self.imu_buffer.append((stamp_sec, omega, accel))
        self.last_imu_stamp_sec = stamp_sec

    def _imu_is_fresh(self, stamp_sec: float) -> bool:
        if self.last_imu_stamp_sec is None:
            return False
        return abs(stamp_sec - self.last_imu_stamp_sec) <= self.imu_timeout_sec

    def _pop_imu_for_interval(self, t_start: float, t_end: float
                               ) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """Extract and remove IMU messages with t_start ≤ t ≤ t_end from buffer."""
        in_interval: List[Tuple[float, np.ndarray, np.ndarray]] = []
        remaining: Deque = collections.deque(maxlen=self.imu_buffer.maxlen)
        for item in self.imu_buffer:
            t = item[0]
            if t_start <= t <= t_end:
                in_interval.append(item)
            elif t > t_start:
                remaining.append(item)
        self.imu_buffer = remaining
        return in_interval

    def _build_imu_init_guess(self, imu_msgs: List, R_world_at_t1: np.ndarray) -> np.ndarray:
        """Pre-integrate IMU and return 4×4 initial transform T_B1_B2."""
        self.integrator.reset()
        if not imu_msgs:
            return np.eye(4, dtype=float)

        prev_t: Optional[float] = None
        for t, omega, accel in imu_msgs:
            if prev_t is not None:
                dt = t - prev_t
                self.integrator.integrate(omega, accel, dt, R_world_at_t1,
                                          use_accel=self.imu_use_accel)
            prev_t = t

        use_trans = self.imu_use_accel and self.integrator.is_accel_ready
        return self.integrator.get_delta_transform(R_world_at_t1, use_translation=use_trans)

    def _imu_confidence(self, current_stamp_sec: float) -> float:
        """Confidence score [0,1] for the IMU initial guess."""
        if not self.use_imu or not self._imu_is_fresh(current_stamp_sec):
            return 0.0
        if self.integrator.is_accel_ready:
            return 0.9   # full 6-DOF pre-integration (rotation + translation)
        return 0.6       # gyro-only (rotation good, translation = 0)

    def _build_nonrep_init_guess(self, pred_pose: np.ndarray) -> np.ndarray:
        """
        Convert the non-rep predictor's absolute predicted pose [x,y,z,yaw]
        into a relative 4×4 T_B1_B2 suitable as a GICP initial guess.

        The non-rep processor tracks poses with yaw-only rotation, so the
        delta rotation is Rz(Δyaw). The translation delta is expressed in
        the previous body frame using the LIO's full R_world.
        """
        st = self.processor.get_current_state()
        if st is None:
            return np.eye(4, dtype=float)

        prev = st.pose  # [x1, y1, z1, yaw1]

        delta_p_world = pred_pose[:3] - prev[:3]
        delta_yaw = float(np.arctan2(
            np.sin(pred_pose[3] - prev[3]),
            np.cos(pred_pose[3] - prev[3]),
        ))

        cy, sy = np.cos(delta_yaw), np.sin(delta_yaw)
        dR = np.array([[cy, -sy, 0.0],
                       [sy,  cy, 0.0],
                       [0.0, 0.0, 1.0]], dtype=float)

        # Express translation in prev body frame via the LIO's full rotation
        dp_body = self.R_world.T @ delta_p_world

        T = np.eye(4, dtype=float)
        T[:3, :3] = dR
        T[:3, 3] = dp_body
        return T

    def _merge_init_guesses(self,
                             T_imu: np.ndarray, w_imu: float,
                             T_nonrep: np.ndarray, w_nonrep: float) -> np.ndarray:
        """
        Blend two relative transforms in tangent space (axis-angle + translation).

        Rotations are averaged in so3 via weighted axis-angle interpolation.
        Translations are blended linearly.
        Falls back to identity when both weights are zero.
        """
        total = w_imu + w_nonrep
        if total < 1e-9:
            return np.eye(4, dtype=float)

        wi = w_imu / total
        wn = w_nonrep / total

        # Blend translations
        dp = wi * T_imu[:3, 3] + wn * T_nonrep[:3, 3]

        # Blend rotations in axis-angle space
        aa = wi * so3_log(T_imu[:3, :3]) + wn * so3_log(T_nonrep[:3, :3])
        dR = so3_exp(aa)

        T = np.eye(4, dtype=float)
        T[:3, :3] = dR
        T[:3, 3] = dp
        return T

    # -------------------------------------------------------------------------
    # Scan buffering / decimation (unchanged logic)
    # -------------------------------------------------------------------------
    def _flush_or_buffer(self, cloud_raw: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
        if not self.accumulate_between_decimation:
            if (self.msg_counter - 1) % self.step_decimation != 0:
                return None
            return cloud_raw

        if len(cloud_raw.points) > 0:
            self._buffer_cloud += cloud_raw
            if len(self._buffer_cloud.points) > self.accumulate_max_points:
                vx = self.accumulate_voxel if self.accumulate_voxel is not None else 0.1
                self._buffer_cloud = self._buffer_cloud.voxel_down_sample(float(vx))

        if (self.msg_counter - 1) % self.step_decimation != 0:
            return None

        merged = self._buffer_cloud
        if self.accumulate_voxel is not None and len(merged.points) > 0:
            merged = merged.voxel_down_sample(float(self.accumulate_voxel))
        self._buffer_cloud = o3d.geometry.PointCloud()
        return merged

    # -------------------------------------------------------------------------
    # Publishers
    # -------------------------------------------------------------------------
    def _publish_odom_and_tf(self, stamp,
                              p_world: np.ndarray, R_world: np.ndarray,
                              v_world: np.ndarray, st: Optional[ScanState]):
        x, y, z = float(p_world[0]), float(p_world[1]), float(p_world[2])
        qx, qy, qz, qw = rot_to_quat(R_world)

        if self.odom_pub is not None:
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = self.map_frame
            odom.child_frame_id = self.base_frame
            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.position.z = z
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw
            # Velocity in body frame
            v_body = R_world.T @ v_world
            odom.twist.twist.linear.x = float(v_body[0])
            odom.twist.twist.linear.y = float(v_body[1])
            odom.twist.twist.linear.z = float(v_body[2])
            # Covariance from processor confidence
            cov6 = np.zeros((6, 6), dtype=float)
            if st is not None and st.uncertainty.shape == (4, 4):
                cov6[0, 0] = float(st.uncertainty[0, 0])
                cov6[1, 1] = float(st.uncertainty[1, 1])
                cov6[2, 2] = float(st.uncertainty[2, 2])
                cov6[5, 5] = float(st.uncertainty[3, 3])
            odom.pose.covariance = cov6.reshape(-1).tolist()
            self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.map_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)

    def _publish_map_cloud(self, stamp):
        if self.map_pub is None:
            return
        if self.scan_counter % self.map_publish_every_n_scans != 0:
            return
        cloud_to_pub = self.map_cloud
        if len(cloud_to_pub.points) > 0 and self.map_publish_voxel > 0:
            cloud_to_pub = cloud_to_pub.voxel_down_sample(float(self.map_publish_voxel))
        if self.map_publish_max_points > 0 and len(cloud_to_pub.points) > self.map_publish_max_points:
            pts = np.asarray(cloud_to_pub.points)
            cols = np.asarray(cloud_to_pub.colors) if cloud_to_pub.has_colors() else None
            idx = np.random.choice(len(pts), self.map_publish_max_points, replace=False)
            tmp = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(pts[idx].astype(np.float64, copy=False))
            if cols is not None and len(cols) == len(pts):
                tmp.colors = o3d.utility.Vector3dVector(cols[idx].astype(np.float64, copy=False))
            cloud_to_pub = tmp
        header = Header()
        header.stamp = stamp
        header.frame_id = self.map_frame
        self.map_pub.publish(open3d_cloud_to_pointcloud2_xyzi(cloud_to_pub, header))

    # -------------------------------------------------------------------------
    # LiDAR callback — core LIO pipeline
    # -------------------------------------------------------------------------
    def cb_cloud(self, msg: PointCloud2):
        self.msg_counter += 1
        xyz, intensity = pointcloud2_to_xyz_i(msg)
        cloud_raw = xyzi_to_open3d_cloud(xyz, intensity)
        cloud = self._flush_or_buffer(cloud_raw)
        if cloud is None or len(cloud.points) == 0:
            return
        if self.max_scans is not None and self.scan_counter >= self.max_scans:
            return

        stamp = msg.header.stamp
        current_stamp_sec = ros_time_to_sec(stamp)

        try:
            feat = self.processor.extract_scan_features(cloud)

            # ------ Non-rep adaptive prediction -----------------------
            # predict_pose_adaptive uses feature similarity, geometric
            # consistency, and temporal extrapolation from the processor's
            # scan-state history.
            pred_pose, pred_conf = self.processor.predict_pose_adaptive(feat)

            # ------ IMU pre-integration --------------------------------
            T_imu = np.eye(4, dtype=float)
            w_imu = 0.0
            if self.use_imu and self.last_scan_stamp_sec is not None:
                if self._imu_is_fresh(current_stamp_sec):
                    imu_msgs = self._pop_imu_for_interval(
                        self.last_scan_stamp_sec, current_stamp_sec)
                    T_imu = self._build_imu_init_guess(imu_msgs, self.R_world.copy())
                    w_imu = self._imu_confidence(current_stamp_sec) * self.imu_base_weight
                else:
                    self.get_logger().warn("IMU stale — falling back to non-rep only")

            # ------ Non-rep → relative transform -----------------------
            T_nonrep = np.eye(4, dtype=float)
            w_nonrep = 0.0
            if pred_pose is not None and self.processor.get_current_state() is not None:
                T_nonrep = self._build_nonrep_init_guess(pred_pose)
                w_nonrep = float(pred_conf) * self.nonrep_base_weight

            # ------ Weighted merge → GICP initial guess ----------------
            T_init = self._merge_init_guesses(T_imu, w_imu, T_nonrep, w_nonrep)

            # ------ GICP registration ----------------------------------
            if self.prev_cloud is not None and len(self.prev_cloud.points) > 0:
                T = self.apply_gicp_func(self.prev_cloud, cloud, T_init)

                # Update 6-DOF world state
                # T = T_B1_B2 : maps current (B2) to previous (B1)
                # → R2 = R1 @ dR,  p2 = p1 + R1 @ dp
                dR = T[:3, :3]
                dp_body = T[:3, 3]

                dt_scan = max(1e-6,
                              current_stamp_sec - self.last_scan_stamp_sec
                              if self.last_scan_stamp_sec else 1.0)
                p_prev = self.p_world.copy()
                self.p_world = self.p_world + self.R_world @ dp_body
                self.R_world = self.R_world @ dR

                # Re-orthogonalize rotation (prevent numerical drift)
                U, _, Vt = np.linalg.svd(self.R_world)
                self.R_world = U @ Vt

                # Velocity estimate from consecutive positions
                self.v_world = (self.p_world - p_prev) / dt_scan

                if self.force_z_zero:
                    self.p_world[2] = 0.0

                reg_conf = estimate_registration_confidence(self.prev_cloud, cloud, T)

                # Feed processor for adaptive tracking
                yaw = float(np.arctan2(self.R_world[1, 0], self.R_world[0, 0]))
                obs_pose = np.array([self.p_world[0], self.p_world[1], self.p_world[2], yaw])
                self.processor.update_with_observation(obs_pose, feat, reg_conf, pred_pose)
                st = self.processor.get_current_state()

            else:
                # First scan — initialize at origin
                self.processor.update_with_observation(
                    np.array([0.0, 0.0, 0.0, 0.0]), feat, 0.3)
                st = self.processor.get_current_state()

            # ------ Publish odometry -----------------------------------
            self._publish_odom_and_tf(stamp, self.p_world, self.R_world, self.v_world, st)

            # ------ Map accumulation -----------------------------------
            Tmap = np.eye(4, dtype=float)
            Tmap[:3, :3] = self.R_world
            Tmap[:3, 3] = self.p_world
            cur_in_map = o3d.geometry.PointCloud(cloud)
            cur_in_map.transform(Tmap)
            self.map_cloud += cur_in_map
            if self.map_voxel > 0 and len(self.map_cloud.points) > 2_000_000:
                self.map_cloud = self.map_cloud.voxel_down_sample(float(self.map_voxel))
            self._publish_map_cloud(stamp)

            if self.viewer is not None:
                self.viewer.update(latest_cloud=cloud, map_cloud=self.map_cloud)

        except Exception as e:
            import traceback
            self.get_logger().error(f"Scan {self.scan_counter} error: {e}\n{traceback.format_exc()}")

        self.prev_cloud = cloud
        self.last_scan_stamp_sec = current_stamp_sec
        self.scan_counter += 1

        if self.max_scans is not None and self.scan_counter >= self.max_scans:
            self.get_logger().info("Max scans reached. Shutting down.")
            rclpy.shutdown()

    def shutdown(self):
        if self.viewer is not None:
            self.viewer.close()


# =============================================================================
# Entry point
# =============================================================================
def main():
    rclpy.init()
    node = LioNode()
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
