#!/usr/bin/env python3.10
"""
ROS2 LiDAR-Inertial Odometry (LIO) — Point-LIO-style iESKF

Architecture:
  1. Each IMU message immediately propagates a 15-DOF error-state Kalman filter
     (p, v, SO(3) rotation, gyro bias, accel bias) — continuous, no batching.
  2. On each LiDAR scan the non-rep predictor + IESKF relative prediction are
     blended as the GICP initial guess (same logic as before).
  3. GICP refines the initial guess to T_B1_B2.
  4. T_B1_B2 feeds the IESKF as a pose measurement: iterated Kalman update on
     the SO(3) manifold replaces the old direct state assignment.
  5. Filter state (p, v, R, online-estimated biases) is published.

Non-rep logic fully preserved:
  NonRepetitiveLiDARProcessor, adaptive blending, feature DB, GICP pipeline.
"""

import collections
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.signal import butter, sosfilt, sosfilt_zi
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
# Data model (unchanged)
# =============================================================================
@dataclass
class ScanState:
    pose: np.ndarray        # [x, y, z, yaw]
    uncertainty: np.ndarray # 4×4 covariance
    confidence: float
    scan_features: Dict


# =============================================================================
# NonRepetitiveLiDARProcessor (completely unchanged)
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
        """Fast lightweight feature extraction (no FPFH, no RANSAC, no normals).

        Caps input to 500 random points before any computation so that all ops
        are O(500) regardless of scan density.  Removed: FPFH O(N·K), RANSAC
        O(N·iters), normal estimation O(N log N).  Retained: centroid, bbox,
        height profile, PCA shape, nearest-neighbour density on 50-pt sample.
        compute_feature_similarity gracefully skips absent keys (fpfh_histogram,
        dominant_plane) so nothing downstream breaks.
        """
        features: Dict = {}
        try:
            points = np.asarray(cloud.points)
            if len(points) == 0:
                return features

            # Cap for speed — all subsequent ops are O(n_sample)
            n_sample = min(500, len(points))
            if len(points) > n_sample:
                idx = np.random.choice(len(points), n_sample, replace=False)
                pts = points[idx]
            else:
                pts = points

            features['point_count'] = int(len(points))
            features['centroid'] = np.mean(pts, axis=0)
            features['std_dev'] = np.std(pts, axis=0)
            features['bounding_box'] = {
                'min': np.min(pts, axis=0),
                'max': np.max(pts, axis=0),
                'extent': np.max(pts, axis=0) - np.min(pts, axis=0),
            }

            z = pts[:, 2]
            features['height_profile'] = {
                'min_height': float(np.min(z)),
                'max_height': float(np.max(z)),
                'mean_height': float(np.mean(z)),
                'height_variance': float(np.var(z)),
            }

            if len(pts) >= 3:
                pca = PCA(n_components=min(3, pts.shape[1]))
                pca.fit(pts)
                evr = pca.explained_variance_ratio_
                # Pad to length 3 if cloud is planar (rank-2)
                while len(evr) < 3:
                    evr = np.append(evr, 0.0)
                features['shape_complexity'] = {
                    'explained_variance_ratio': evr,
                    'linearity': float(evr[0]),
                    'planarity': float(evr[1]),
                    'sphericity': float(evr[2]),
                }

            # Nearest-neighbour density on tiny 50-pt sub-sample (O(50²))
            if len(pts) >= 10:
                tiny = pts[:min(50, len(pts))]
                diff = tiny[:, np.newaxis, :] - tiny[np.newaxis, :, :]  # (N, N, 3)
                dists = np.sqrt(np.sum(diff ** 2, axis=-1))
                np.fill_diagonal(dists, np.inf)
                features['local_density'] = float(np.mean(np.min(dists, axis=1)))

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

    def predict_pose_feature_based(self, current_features: Dict) -> Tuple[Optional[np.ndarray], float]:
        if len(self.feature_database) < 2:
            return None, 0.0
        sims = []
        for i, (feat, st) in enumerate(zip(self.feature_database, self.scan_states)):
            sims.append((self.compute_feature_similarity(current_features, feat), i, st))
        sims.sort(key=lambda x: x[0], reverse=True)
        if len(sims) < 2:
            return None, 0.0
        best_sim, _, best_state = sims[0]
        sec_sim, _, sec_state = sims[1]
        if best_sim < 0.3:
            return None, 0.0
        w1 = best_sim / (best_sim + sec_sim + 1e-12)
        w2 = sec_sim / (best_sim + sec_sim + 1e-12)
        pred = w1 * best_state.pose + w2 * sec_state.pose
        conf = float((best_sim + sec_sim) / 2.0)
        return pred, conf

    def predict_pose_geometric_consistency(self) -> Tuple[Optional[np.ndarray], float]:
        if len(self.scan_states) < 3:
            return None, 0.0
        recent = self.scan_states[-min(5, len(self.scan_states)):]
        poses = [s.pose for s in recent]
        if len(poses) < 3:
            return None, 0.0
        pred = np.zeros(4, dtype=float)
        confs = []
        dims = [0, 1, 3] if self.force_z_zero else [0, 1, 2, 3]
        for dim in dims:
            vals = np.array([p[dim] for p in poses], dtype=float)
            x = np.arange(len(vals), dtype=float)
            deg = min(2, len(vals) - 1)
            coeffs = np.polyfit(x, vals, deg)
            pred[dim] = float(np.polyval(coeffs, float(len(vals))))
            fitted = np.polyval(coeffs, x)
            mse = float(np.mean((vals - fitted) ** 2))
            confs.append(max(0.0, 1.0 - mse))
        if self.force_z_zero:
            pred[2] = 0.0
        pred[3] = float(np.arctan2(np.sin(pred[3]), np.cos(pred[3])))
        return pred, float(np.mean(confs)) if confs else 0.0

    def predict_pose_adaptive(self, current_features: Dict) -> Tuple[Optional[np.ndarray], float]:
        preds = []
        fp, fc = self.predict_pose_feature_based(current_features)
        if fp is not None:
            preds.append((fp, fc, "feature"))
        gp, gc = self.predict_pose_geometric_consistency()
        if gp is not None:
            preds.append((gp, gc, "geometric"))
        if len(self.scan_states) >= 2:
            last = self.scan_states[-1].pose
            prev = self.scan_states[-2].pose
            extrap = last + 0.3 * (last - prev)
            extrap[3] = np.arctan2(np.sin(extrap[3]), np.cos(extrap[3]))
            if self.force_z_zero:
                extrap[2] = 0.0
            preds.append((extrap, 0.4, "extrapolation"))
        if not preds:
            return None, 0.0
        if len(preds) == 1:
            return preds[0][0], float(preds[0][1])
        total_w = 0.0
        weighted = np.zeros(4, dtype=float)
        for pose, conf, strat in preds:
            strat_w = (self.feature_weight if strat == "feature"
                       else self.geometric_weight if strat == "geometric"
                       else self.temporal_weight)
            w = float(conf) * float(strat_w)
            weighted += w * pose
            total_w += w
        if total_w > 1e-12:
            out = weighted / total_w
            if self.force_z_zero:
                out[2] = 0.0
            return out, float(total_w / len(preds))
        return None, 0.0

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
# VoxelHashMap — bounded O(1)-insert map with radius pruning
# =============================================================================
class VoxelHashMap:
    """
    Replaces the unbounded Open3D PointCloud accumulation.

    Storage model:
      Each occupied voxel stores the centroid of the last point inserted into
      it (one point per cell).  This is equivalent to voxel downsampling at
      insert-time without any periodic batch operation.

    Complexity:
      insert()      O(N)          N = points in current scan
      prune_far()   O(V) numpy   V = occupied voxels (bounded by max_voxels)
      get_submap()  O(V) numpy   V = occupied voxels
      to_open3d()   O(V)

    Memory: max_voxels × (3 float32 centroid + 3 float32 colour) ≈ 14 MB at 400k.
    """

    def __init__(self,
                 voxel_size: float = 0.15,
                 prune_radius: float = 80.0,
                 max_voxels: int = 400_000):
        self._vx = float(voxel_size)
        self._prune_r2 = float(prune_radius) ** 2
        self._max = int(max_voxels)
        # (ix, iy, iz) → centroid np.ndarray shape (3,) float32
        self._cells: Dict[tuple, np.ndarray] = {}
        self._colors: Dict[tuple, np.ndarray] = {}
        # insertion-order list of keys for FIFO eviction when > max_voxels
        self._order: List[tuple] = []

    def insert(self, pts: np.ndarray,
               colors: Optional[np.ndarray] = None) -> None:
        if pts.shape[0] == 0:
            return
        pts32 = pts.astype(np.float32, copy=False)
        keys_arr = (pts32 / self._vx).astype(np.int32)
        for i in range(len(pts32)):
            k = (int(keys_arr[i, 0]), int(keys_arr[i, 1]), int(keys_arr[i, 2]))
            if k not in self._cells:
                self._order.append(k)
            self._cells[k] = pts32[i]
            if colors is not None and i < len(colors):
                self._colors[k] = colors[i].astype(np.float32, copy=False)
        # FIFO eviction when over capacity
        while len(self._cells) > self._max:
            old = self._order.pop(0)
            self._cells.pop(old, None)
            self._colors.pop(old, None)

    def prune_far(self, center: np.ndarray) -> int:
        """Remove voxels beyond prune_radius from center.  Returns count removed."""
        if not self._cells:
            return 0
        keys = list(self._cells.keys())
        pts = np.array([self._cells[k] for k in keys], dtype=np.float32)
        diff = pts - center.astype(np.float32)
        far_mask = np.sum(diff ** 2, axis=1) > self._prune_r2
        del_count = 0
        for i, k in enumerate(keys):
            if far_mask[i]:
                del self._cells[k]
                self._colors.pop(k, None)
                del_count += 1
        # Rebuild insertion-order list (rare operation)
        if del_count > 0:
            valid = set(self._cells.keys())
            self._order = [k for k in self._order if k in valid]
        return del_count

    def get_submap(self, center: np.ndarray, radius: float) -> o3d.geometry.PointCloud:
        """Return an Open3D cloud of all voxels within radius of center."""
        if not self._cells:
            return o3d.geometry.PointCloud()
        pts = np.array(list(self._cells.values()), dtype=np.float32)
        diff = pts - center.astype(np.float32)
        in_mask = np.sum(diff ** 2, axis=1) <= float(radius) ** 2
        pts_in = pts[in_mask]
        cloud = o3d.geometry.PointCloud()
        if len(pts_in) > 0:
            cloud.points = o3d.utility.Vector3dVector(pts_in.astype(np.float64))
        return cloud

    def to_open3d(self, max_points: int = 800_000) -> o3d.geometry.PointCloud:
        """Materialise the full map as an Open3D cloud (for publishing)."""
        if not self._cells:
            return o3d.geometry.PointCloud()
        pts = np.array(list(self._cells.values()), dtype=np.float64)
        has_color = bool(self._colors)
        if has_color:
            cols = np.array(
                [self._colors.get(k, np.array([0.5, 0.5, 0.5], dtype=np.float32))
                 for k in self._cells], dtype=np.float64)
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]
            if has_color:
                cols = cols[idx]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        if has_color:
            cloud.colors = o3d.utility.Vector3dVector(cols)
        return cloud

    def __len__(self) -> int:
        return len(self._cells)


# =============================================================================
# SO(3) helpers
# =============================================================================
def so3_exp(omega_dt: np.ndarray) -> np.ndarray:
    """Axis-angle → rotation matrix (Rodrigues)."""
    angle = float(np.linalg.norm(omega_dt))
    K = np.array([[0.0, -omega_dt[2], omega_dt[1]],
                  [omega_dt[2], 0.0, -omega_dt[0]],
                  [-omega_dt[1], omega_dt[0], 0.0]])
    if angle < 1e-9:
        return np.eye(3) + K
    axis_K = K / angle
    return np.eye(3) + np.sin(angle) * axis_K + (1.0 - np.cos(angle)) * (axis_K @ axis_K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle."""
    cos_angle = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if abs(angle) < 1e-9:
        return np.zeros(3)
    return (angle / (2.0 * np.sin(angle))) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def _skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric matrix."""
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])


# =============================================================================
# IESKF — Iterated Error-State Kalman Filter on SO(3)
# =============================================================================
class IESKF:
    """
    15-DOF iterated error-state Kalman filter for continuous IMU + LiDAR fusion.

    Nominal state  x  = [p(3), v(3), R∈SO(3), bg(3), ba(3)]
    Error state   δx  = [δp,   δv,   δθ,      δbg,   δba ] ∈ R^15

    Gravity convention (same as the old ImuPreintegrator):
        g_world = mean(R_world @ a_static)  ≈ [0, 0, 9.81] for z-up
        true_accel_world = R @ (a_meas - ba) - g_world

    IMU propagation:
        Call propagate() once per IMU message (continuous, not batched).
        Propagation is gated on gravity_initialized AND first scan complete.

    LiDAR measurement update:
        GICP returns T_B1_B2 (relative 4×4).  Call update() to run the
        iterated Kalman step that corrects p, v, R, bg, ba and shrinks P.

    Process noise (noise spectral density convention):
        sigma_gyro  [rad/s/√Hz]     angle random walk
        sigma_accel [m/s²/√Hz]      velocity random walk
        sigma_bg    [rad/s/√Hz]     gyro bias random walk rate
        sigma_ba    [m/s²/√Hz]      accel bias random walk rate
    """

    def __init__(
        self,
        gravity_mag: float = 9.81,
        gravity_init_n: int = 100,
        sigma_gyro: float = 0.005,
        sigma_accel: float = 0.05,
        sigma_bg: float = 1e-4,
        sigma_ba: float = 1e-3,
        init_p_cov: float = 1e-6,
        init_v_cov: float = 1e-4,
        init_rot_cov: float = 1e-6,
        init_bg_cov: float = 1e-4,
        init_ba_cov: float = 1e-4,
        max_iters: int = 3,
        conv_threshold: float = 1e-4,
    ):
        # Gravity
        self.gravity_mag = gravity_mag
        self.g_world = np.array([0.0, 0.0, gravity_mag])
        self.gravity_init_n = gravity_init_n
        self.gravity_initialized = False
        self._gravity_samples: List[np.ndarray] = []
        self._gyro_samples:    List[np.ndarray] = []   # for static gyro bias estimation
        self.gravity_z_down = False  # set True via config if sensor z-axis points down

        # Nominal state
        self.p  = np.zeros(3)
        self.v  = np.zeros(3)
        self.R  = np.eye(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)

        # Error covariance (15×15)
        d = np.concatenate([[init_p_cov]*3, [init_v_cov]*3, [init_rot_cov]*3,
                             [init_bg_cov]*3, [init_ba_cov]*3])
        self.P = np.diag(d.astype(float))

        # Process noise densities
        self.sigma_gyro  = sigma_gyro
        self.sigma_accel = sigma_accel
        self.sigma_bg    = sigma_bg
        self.sigma_ba    = sigma_ba

        # iESKF iteration control
        self.max_iters      = max_iters
        self.conv_threshold = conv_threshold

        # State snapshot saved after each scan update; used to compute
        # get_predicted_relative_transform() for the next GICP initial guess.
        self._p_last = np.zeros(3)
        self._R_last = np.eye(3)

        # Full snapshot for delayed-update repropagate
        self._snap: Optional[dict] = None

    # ------------------------------------------------------------------
    # Static initialisation: gravity + gyro bias (fix 2 + fix 4)
    # ------------------------------------------------------------------
    def collect_static_sample(self, omega: np.ndarray, accel: np.ndarray,
                               R_world: np.ndarray) -> bool:
        """
        Collect one IMU sample during the static initialisation window.
        Returns True the moment initialisation completes.

        Fix 2 — gyro bias: averages gyro readings over the same static window
                 used for gravity, giving an initial bg estimate before the
                 iESKF has had time to estimate it from GICP updates.

        Fix 4 — gravity direction: if the estimated g_world z-component is
                 negative (z-down sensor mounting), the sign is flipped so
                 the propagation formula  a_world = R @ accel_c − g_world
                 correctly removes gravity regardless of sensor orientation.
                 Set imu_gravity_z_down: true in the yaml to force this flip
                 without waiting for auto-detection.
        """
        if self.gravity_initialized:
            return False
        a_c = accel - self.ba
        if abs(float(np.linalg.norm(a_c)) - self.gravity_mag) < 1.5:
            self._gravity_samples.append(R_world @ a_c)
            self._gyro_samples.append(omega.copy())
            if len(self._gravity_samples) >= self.gravity_init_n:
                self.g_world = np.mean(self._gravity_samples, axis=0)
                # Fix 2: bootstrap gyro bias from static window
                # (iESKF will refine it online; this removes the worst of
                #  the yaw drift in the first few seconds of operation)
                self.bg = np.mean(self._gyro_samples, axis=0)
                # Fix 4: normalise gravity direction
                if self.gravity_z_down or self.g_world[2] < 0.0:
                    self.g_world = -self.g_world
                self.gravity_initialized = True
                return True
        return False

    # ------------------------------------------------------------------
    # Continuous IMU propagation (one call per IMU message)
    # ------------------------------------------------------------------
    def propagate(self, omega: np.ndarray, accel: np.ndarray, dt: float,
                  use_accel: bool = True):
        if dt <= 0.0 or dt > 1.0:
            return

        omega_c = omega - self.bg
        R_new   = self.R @ so3_exp(omega_c * dt)

        if use_accel:
            accel_c = accel - self.ba
            a_world = self.R @ accel_c - self.g_world
            p_new   = self.p + self.v * dt + 0.5 * a_world * (dt * dt)
            v_new   = self.v + a_world * dt
        else:
            # Gyro-only: position is held constant; velocity reset to zero so
            # that no accel-integration drift accumulates between GICP updates.
            accel_c = np.zeros(3)
            a_world = np.zeros(3)
            p_new   = self.p
            v_new   = np.zeros(3)

        # ----- Error state Jacobian F (15×15) -----
        F = np.eye(15)
        F[0:3, 3:6]  = np.eye(3) * dt
        if use_accel:
            F[3:6, 6:9]   = -(self.R @ _skew(accel_c)) * dt
            F[3:6, 12:15] = -self.R * dt
        F[6:9, 6:9]  = so3_exp(-omega_c * dt)
        F[6:9, 9:12] = -np.eye(3) * dt

        # ----- Process noise input Jacobian G (15×12) -----
        G = np.zeros((15, 12))
        if use_accel:
            G[3:6, 3:6]   = self.R
            G[12:15, 9:12] = np.eye(3)
        G[6:9, 0:3]  = np.eye(3)
        G[9:12, 6:9] = np.eye(3)

        # ----- Discrete process noise Q (12×12) -----
        Q = np.zeros((12, 12))
        Q[0:3, 0:3] = np.eye(3) * (self.sigma_gyro ** 2 / dt)
        Q[6:9, 6:9] = np.eye(3) * (self.sigma_bg   ** 2 * dt)
        if use_accel:
            Q[3:6,  3:6]  = np.eye(3) * (self.sigma_accel ** 2 / dt)
            Q[9:12, 9:12] = np.eye(3) * (self.sigma_ba    ** 2 * dt)

        self.P = F @ self.P @ F.T + G @ Q @ G.T

        self.p = p_new
        self.v = v_new
        U, _, Vt = np.linalg.svd(R_new)
        self.R = U @ Vt

    # ------------------------------------------------------------------
    # Scan-state snapshot
    # ------------------------------------------------------------------
    def save_scan_state(self):
        """Call after each scan's measurement update."""
        self._p_last = self.p.copy()
        self._R_last = self.R.copy()

    # ------------------------------------------------------------------
    # Full state snapshot for delayed-update repropagate
    # ------------------------------------------------------------------
    def save_full_snapshot(self) -> dict:
        return {
            'p': self.p.copy(), 'v': self.v.copy(), 'R': self.R.copy(),
            'bg': self.bg.copy(), 'ba': self.ba.copy(),
            'P': self.P.copy(),
            '_p_last': self._p_last.copy(), '_R_last': self._R_last.copy(),
        }

    def restore_full_snapshot(self, snap: dict) -> None:
        self.p  = snap['p'].copy()
        self.v  = snap['v'].copy()
        self.R  = snap['R'].copy()
        self.bg = snap['bg'].copy()
        self.ba = snap['ba'].copy()
        self.P  = snap['P'].copy()
        self._p_last = snap['_p_last'].copy()
        self._R_last = snap['_R_last'].copy()

    def get_predicted_relative_transform(self) -> np.ndarray:
        """
        Return T_B1_B2 (4×4) — relative transform from IMU propagation
        since the last save_scan_state() call.  Used as GICP initial guess.
        """
        dR = self._R_last.T @ self.R
        dp_body = self._R_last.T @ (self.p - self._p_last)
        T = np.eye(4, dtype=float)
        T[:3, :3] = dR
        T[:3, 3]  = dp_body
        return T

    # ------------------------------------------------------------------
    # iESKF measurement update
    # ------------------------------------------------------------------
    def update(
        self,
        T_gicp: np.ndarray,
        R_n: np.ndarray,
        chi2_threshold: float = 22.46,
    ) -> Tuple[bool, float]:
        """
        Iterated EKF measurement update using the GICP relative transform.

        T_gicp          4×4 T_B1_B2 (maps current body B2 → previous body B1)
        R_n             6×6 measurement noise covariance [pos(3), rot(3)] order
        chi2_threshold  Mahalanobis gating threshold (chi²(6, 0.999) ≈ 22.46)

        Returns (accepted, chi2_val).
        """
        # Absolute pose measurement from relative GICP transform
        # p_meas = p_B1 + R_B1 * t_{B1←B2}   (SE(3) composition)
        p_meas = self._p_last + self._R_last @ T_gicp[:3, 3]
        R_meas = self._R_last @ T_gicp[:3, :3]

        # Measurement Jacobian H (6×15):  [pos|vel|rot|bg|ba]
        #   pos rows:  dh/dp = I    (identity on position block)
        #   rot rows:  dh/dθ = I    (identity on rotation error block)
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)

        # Innovation covariance and chi-squared gating
        S = H @ self.P @ H.T + R_n

        # Pre-update innovation at nominal state
        z_p0 = p_meas - self.p
        z_R0 = so3_log(self.R.T @ R_meas)
        z0   = np.concatenate([z_p0, z_R0])
        S_inv = np.linalg.inv(S)
        chi2  = float(z0 @ S_inv @ z0)

        if chi2 > chi2_threshold:
            return False, chi2   # outlier — skip update, keep propagated state

        K = self.P @ H.T @ S_inv

        # Iterated update: re-linearise on SO(3) around current iterate
        p_k  = self.p.copy()
        v_k  = self.v.copy()
        R_k  = self.R.copy()
        bg_k = self.bg.copy()
        ba_k = self.ba.copy()

        for _ in range(self.max_iters):
            z_p = p_meas - p_k
            z_R = so3_log(R_k.T @ R_meas)   # proper SO(3) residual
            z   = np.concatenate([z_p, z_R])

            dx = K @ z

            p_k  = p_k  + dx[0:3]
            v_k  = v_k  + dx[3:6]
            R_k  = R_k  @ so3_exp(dx[6:9])  # right perturbation on SO(3)
            bg_k = bg_k + dx[9:12]
            ba_k = ba_k + dx[12:15]

            if float(np.linalg.norm(dx)) < self.conv_threshold:
                break

        # Commit
        self.p  = p_k
        self.v  = v_k
        U, _, Vt = np.linalg.svd(R_k)
        self.R  = U @ Vt                    # project onto SO(3)
        self.bg = bg_k
        self.ba = ba_k

        # Joseph-form covariance update (numerically stable)
        I_KH   = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_n @ K.T

        return True, chi2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ZUPT — zero-velocity pseudo-measurement
    # ------------------------------------------------------------------
    def zupt_update(self, sigma_v: float = 0.01) -> None:
        """Inject v = 0 as a pseudo-measurement when the robot is stationary.

        H selects the velocity block; the Kalman gain propagates corrections
        back to accel bias (ba) through P[v, ba] cross-covariance accumulated
        during propagation, bounding drift during GICP-failure windows.
        """
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        R_n = np.eye(3) * (sigma_v ** 2)
        S   = H @ self.P @ H.T + R_n
        K   = self.P @ H.T @ np.linalg.inv(S)
        dx  = K @ (-self.v)                     # innovation: 0 - v_predicted
        self.p  += dx[0:3]
        self.v  += dx[3:6]
        self.R   = self.R @ so3_exp(dx[6:9])
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        I_KH   = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_n @ K.T

    # ------------------------------------------------------------------
    # Soft floor constraint — z pseudo-measurement
    # ------------------------------------------------------------------
    def soft_z_update(self, sigma_z: float = 0.05) -> None:
        """Soft floor constraint: p_z = 0 with uncertainty sigma_z [m].

        Unlike hard zeroing this lets a z-residual exist, which drives
        ba_z correction through the P[p_z, ba_z] cross-covariance and
        makes the z-accel bias observable.  sigma_z controls the trade-off:
        tight (0.02 m) → near-rigid floor; loose (0.2 m) → gentle nudge.
        """
        H = np.zeros((1, 15))
        H[0, 2] = 1.0                           # observe p_z
        R_n = np.array([[sigma_z ** 2]])
        S   = H @ self.P @ H.T + R_n
        K   = self.P @ H.T @ np.linalg.inv(S)
        dx  = K @ np.array([0.0 - self.p[2]])   # innovation: 0 - p_z
        self.p  += dx[0:3]
        self.v  += dx[3:6]
        self.R   = self.R @ so3_exp(dx[6:9])
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        I_KH   = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_n @ K.T

    @property
    def is_gyro_ready(self) -> bool:
        return True

    @property
    def is_accel_ready(self) -> bool:
        return self.gravity_initialized

    @property
    def pose_covariance_6x6(self) -> np.ndarray:
        """Extract [pos, rot] cross-covariance block from P for ROS publishing."""
        cov = np.zeros((6, 6))
        cov[0:3, 0:3] = self.P[0:3, 0:3]
        cov[3:6, 3:6] = self.P[6:9, 6:9]
        cov[0:3, 3:6] = self.P[0:3, 6:9]
        cov[3:6, 0:3] = self.P[6:9, 0:3]
        return cov


# =============================================================================
# Helper functions (unchanged)
# =============================================================================
def ros_time_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return 0.0, 0.0, float(np.sin(half)), float(np.cos(half))


def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
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
    xyz, intensity, _ = pointcloud2_to_xyz_i_stamps(msg)
    return xyz, intensity


def pointcloud2_to_xyz_i_stamps(
    msg: PointCloud2,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract xyz, intensity, and optional per-point timestamps from a PointCloud2.

    Per-point timestamps are returned as float64 seconds if the message has a
    'timestamp' (uint64 ns) or 't' field; otherwise the third return is None.
    """
    n_pts = msg.width * msg.height
    if n_pts == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                None)
    field_map  = {f.name: f for f in msg.fields}
    field_offsets = {f.name: f.offset for f in msg.fields}
    step = msg.point_step
    raw  = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_pts, step)

    def _col_f32(off: int) -> np.ndarray:
        return raw[:, off:off + 4].copy().ravel().view(np.float32)

    xyz = np.column_stack([
        _col_f32(field_offsets['x']),
        _col_f32(field_offsets['y']),
        _col_f32(field_offsets['z']),
    ])
    intensity = (_col_f32(field_offsets['intensity']) if 'intensity' in field_offsets
                 else np.zeros(n_pts, dtype=np.float32))

    # Per-point timestamp: Livox driver writes uint64 nanoseconds in 'timestamp' or 't'
    stamps_sec: Optional[np.ndarray] = None
    ts_field = 'timestamp' if 'timestamp' in field_map else ('t' if 't' in field_map else None)
    if ts_field is not None:
        off = field_offsets[ts_field]
        datatype = field_map[ts_field].datatype
        if datatype == 8:   # UINT64
            stamps_ns = raw[:, off:off + 8].copy().ravel().view(np.uint64)
            stamps_sec = stamps_ns.astype(np.float64) * 1e-9
        elif datatype == 7: # FLOAT64
            stamps_sec = raw[:, off:off + 8].copy().ravel().view(np.float64).copy()
        elif datatype == 6: # FLOAT32 (rare)
            stamps_sec = raw[:, off:off + 4].copy().ravel().view(np.float32).astype(np.float64)

    valid = np.isfinite(xyz).all(axis=1)
    stamps_out = stamps_sec[valid] if stamps_sec is not None else None
    return xyz[valid].astype(np.float32), intensity[valid], stamps_out


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
        PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
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
# GICP (Open3D fallback with init_T support, unchanged)
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
        self.queue_size  = int(p("queue_size", 10))

        # ---- IMU parameters
        self.use_imu           = bool(p("use_imu", True))
        self.imu_topic         = str(p("imu_topic", "/imu/data"))
        self.imu_gravity_mag   = float(p("imu_gravity_mag", 9.81))
        self.imu_gravity_init_n = int(p("imu_gravity_init_n", 100))
        self.imu_use_accel     = bool(p("imu_use_accel", True))
        self.imu_timeout_sec    = float(p("imu_timeout_sec", 0.5))
        self.imu_gravity_z_down = bool(p("imu_gravity_z_down", False))
        # Scale factor for raw accelerometer readings → m/s².
        # Set to 9.81 when the IMU publishes in g-units (e.g. Livox Horizon bags).
        self.imu_accel_scale   = float(p("imu_accel_scale", 1.0))
        gyro_bias_raw  = p("imu_gyro_bias",  [0.0, 0.0, 0.0])
        accel_bias_raw = p("imu_accel_bias", [0.0, 0.0, 0.0])

        # ---- iESKF noise parameters
        self.ieskf_sigma_gyro     = float(p("ieskf_sigma_gyro",     0.005))
        self.ieskf_sigma_accel    = float(p("ieskf_sigma_accel",    0.05))
        self.ieskf_sigma_bg       = float(p("ieskf_sigma_bg",       1e-4))
        self.ieskf_sigma_ba       = float(p("ieskf_sigma_ba",       1e-3))
        self.ieskf_init_p_cov     = float(p("ieskf_init_p_cov",     1.0))
        self.ieskf_init_bg_cov    = float(p("ieskf_init_bg_cov",    1e-4))
        self.ieskf_init_ba_cov    = float(p("ieskf_init_ba_cov",    1e-4))
        self.ieskf_max_iters      = int(p("ieskf_max_iters",        3))
        self.ieskf_meas_noise_pos = float(p("ieskf_meas_noise_pos", 0.05))
        self.ieskf_meas_noise_rot = float(p("ieskf_meas_noise_rot", 0.01))

        # ---- Publishing
        self.publish_odom           = bool(p("publish_odom", True))
        self.odom_topic             = str(p("odom_topic", "/lio/odom"))
        self.publish_map            = bool(p("publish_map", True))
        self.map_topic              = str(p("map_topic", "/lio/map"))
        self.publish_tf             = bool(p("publish_tf", False))
        self.map_frame              = str(p("map_frame", "map"))
        self.odom_frame             = str(p("odom_frame", "odom"))
        self.base_frame             = str(p("base_frame", "base_link"))
        self.map_publish_voxel      = float(p("map_publish_voxel", 0.15))
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
        self.force_z_zero  = bool(p("force_z_zero", False))
        # Soft floor constraint: replaces hard zeroing when > 0.
        # sigma_z [m] — how tightly to constrain p_z to 0.
        # 0.05 m is a good indoor default; set 0 to keep legacy hard-zero.
        self.soft_z_sigma  = float(p("soft_z_sigma", 0.05))

        # ---- ZUPT (zero-velocity update)
        # Applied every IMU tick when the robot appears stationary.
        self.zupt_omega_thresh = float(p("zupt_omega_thresh", 0.05))  # rad/s
        self.zupt_accel_thresh = float(p("zupt_accel_thresh", 0.30))  # m/s²
        self.zupt_sigma_v      = float(p("zupt_sigma_v",      0.01))  # m/s

        # ---- Motion-adaptive GICP noise
        # Thresholds for classifying motion state from the inter-scan IMU window.
        self.motion_omega_stationary      = float(p("motion_omega_stationary",      0.05))  # rad/s
        self.motion_omega_rotating        = float(p("motion_omega_rotating",        0.30))  # rad/s
        self.motion_accel_stationary      = float(p("motion_accel_stationary",      0.30))  # m/s²
        self.motion_accel_translating     = float(p("motion_accel_translating",     0.80))  # m/s²
        # When stationary: skip GICP entirely (ZUPT holds position).
        self.motion_skip_stationary       = bool( p("motion_skip_stationary",       True))
        # When rotating fast: inflate GICP rotation noise so gyro propagation holds rotation.
        self.motion_rot_noise_scale       = float(p("motion_rot_noise_scale",       4.0))
        # When translating: deflate GICP position noise so GICP drives translation.
        self.motion_trans_pos_noise_scale = float(p("motion_trans_pos_noise_scale", 0.5))

        # ---- Map
        self.map_voxel  = float(p("map_voxel", 0.15))

        # ---- Debug CSV (per-scan diagnostics; empty string = disabled)
        _debug_csv = str(p("debug_csv", ""))
        self._debug_fh = None
        if _debug_csv:
            import os as _os
            _os.makedirs(_os.path.dirname(_debug_csv) or ".", exist_ok=True)
            self._debug_fh = open(_debug_csv, "w", buffering=1)
            self._debug_fh.write(
                "scan_num,stamp,x,y,z,yaw,"
                "imu_x,imu_y,imu_z,imu_yaw,"
                "gicp_dx,gicp_dy,gicp_dz,gicp_dyaw,"
                "gicp_conf,chi2,accepted,"
                "motion,n_map_vox,n_scan_pts,use_submap\n"
            )
            self.get_logger().info(f"[debug] Writing per-scan CSV → {_debug_csv}")

        # ---- GICP
        self.use_pctools_gicp       = bool(p("use_pctools_gicp", True))
        self.gicp_max_corr_distance = float(p("gicp_max_corr_distance", 2.0))
        self.gicp_voxel_size        = float(p("gicp_voxel_size", 0.2))
        self.gicp_max_iterations    = int(p("gicp_max_iterations", 50))

        # ---- Initial-guess fusion weights (GICP seeding, unchanged role)
        self.imu_base_weight    = float(p("imu_base_weight",    0.7))
        self.nonrep_base_weight = float(p("nonrep_base_weight", 0.3))

        # ---- Degenerate-scan threshold
        self.gicp_min_conf = float(p("gicp_min_conf", 0.25))

        # ---- Non-rep processor
        self.processor = NonRepetitiveLiDARProcessor(force_z_zero=self.force_z_zero)

        # ---- iESKF (replaces ImuPreintegrator)
        self.ieskf = IESKF(
            gravity_mag     = self.imu_gravity_mag,
            gravity_init_n  = self.imu_gravity_init_n,
            sigma_gyro      = self.ieskf_sigma_gyro,
            sigma_accel     = self.ieskf_sigma_accel,
            sigma_bg        = self.ieskf_sigma_bg,
            sigma_ba        = self.ieskf_sigma_ba,
            init_p_cov      = self.ieskf_init_p_cov,
            init_bg_cov     = self.ieskf_init_bg_cov,
            init_ba_cov     = self.ieskf_init_ba_cov,
            max_iters       = self.ieskf_max_iters,
        )
        self.ieskf.gravity_z_down = self.imu_gravity_z_down
        self.ieskf.bg = np.asarray(gyro_bias_raw,  dtype=float)
        self.ieskf.ba = np.asarray(accel_bias_raw, dtype=float)

        # IMU stamp tracking (dt computation for propagation; no buffer needed)
        self.last_imu_stamp_sec: Optional[float] = None
        # Last raw IMU measurements — used to bridge the gap to exact scan time
        self._last_omega: np.ndarray = np.zeros(3)
        self._last_accel: np.ndarray = np.zeros(3)

        # 2nd-order Butterworth IIR — replaces 41-tap FIR.
        # Group delay: ~1/(2*fc) samples ≈ 10 ms at 25 Hz / 200 Hz,
        # vs 100 ms for the old 41-tap Hann FIR.
        _imu_in_hz = 200.0
        _cutoff_hz = float(p("imu_fir_cutoff_hz", 25.0))  # reuse same yaml key
        self._iir_sos: np.ndarray = butter(
            2, _cutoff_hz / (_imu_in_hz / 2.0), btype='low', output='sos'
        ).astype(np.float64)
        # Per-channel initial conditions — shape (n_sections, 2) each
        _zi0 = sosfilt_zi(self._iir_sos)                    # (n_sections, 2)
        self._iir_zi_omega: List[np.ndarray] = [_zi0.copy() for _ in range(3)]
        self._iir_zi_accel: List[np.ndarray] = [_zi0.copy() for _ in range(3)]
        self._iir_initialized = False
        # Group delay is frequency-dependent; approximate constant for dt correction
        self._fir_group_delay_sec = 1.0 / (2.0 * _cutoff_hz)  # ≈ 0.020 s
        # Separate propagation timestamp tracked in effective (delay-compensated) time
        self._last_prop_stamp: Optional[float] = None

        # IMU ring buffer for deskewing and delayed-update repropagate
        # Stores (t_eff, omega_filt, accel_filt) at full 200 Hz for ≈2.5 s
        self._imu_buffer: collections.deque = collections.deque(maxlen=500)

        # GICP → iESKF covariance parameters
        self._gicp_cov_scale  = float(p("gicp_cov_scale", 100.0))
        self._chi2_threshold  = float(p("gicp_chi2_threshold", 22.46))

        # ---- Voxel hash map (replaces unbounded map_cloud accumulation)
        # map_voxel already declared above — reuse self.map_voxel to avoid duplicate
        self.voxel_map = VoxelHashMap(
            voxel_size   = self.map_voxel,
            prune_radius = float(p("voxel_map_prune_radius", 80.0)),
            max_voxels   = int(p("voxel_map_max_voxels", 400_000)),
        )
        # Submap radius for scan-to-submap GICP (m). 0 → keep scan-to-scan.
        self.gicp_submap_radius = float(p("gicp_submap_radius", 25.0))
        # Min voxels in map before switching to submap GICP
        self._submap_min_voxels = 200

        # ---- Scan state
        self.prev_cloud: Optional[o3d.geometry.PointCloud] = None
        self._buffer_cloud = o3d.geometry.PointCloud()
        self.msg_counter   = 0
        self.scan_counter  = 0
        self.last_scan_stamp_sec: Optional[float] = None

        # ---- GICP callable
        self.apply_gicp_func = self._resolve_gicp()

        # ---- Publishers / subscribers
        self.odom_pub     = self.create_publisher(Odometry, self.odom_topic, 10) if self.publish_odom else None
        self.imu_odom_pub = (self.create_publisher(Odometry, self.odom_topic + "_imu_only", 10)
                             if self.publish_odom and self.use_imu else None)
        self.map_pub      = self.create_publisher(PointCloud2, self.map_topic, 1) if self.publish_map else None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.publish_tf else None

        self.lidar_sub = self.create_subscription(
            PointCloud2, self.lidar_topic, self.cb_cloud, self.queue_size)
        if self.use_imu:
            self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.cb_imu, 200)

        self.get_logger().info("=== LIO Node (Non-Rep + iESKF Point-LIO style) ===")
        self.get_logger().info(f"lidar={self.lidar_topic}  use_imu={self.use_imu}")
        if self.use_imu:
            self.get_logger().info(
                f"imu={self.imu_topic}  "
                f"accel_scale={self.imu_accel_scale}  "
                f"σ_gyro={self.ieskf_sigma_gyro}  σ_accel={self.ieskf_sigma_accel}  "
                f"σ_bg={self.ieskf_sigma_bg}  σ_ba={self.ieskf_sigma_ba}"
            )
            self.get_logger().info(
                f"IIR filter: 2nd-order Butterworth LP, "
                f"cutoff={_cutoff_hz:.0f} Hz, "
                f"approx group_delay={self._fir_group_delay_sec*1000:.0f} ms "
                f"(was 100 ms with 41-tap FIR), propagation at full 200 Hz"
            )

    # -------------------------------------------------------------------------
    # GICP resolver — returns func(prev, curr, T_init) -> (T_4x4, H_6x6|None)
    # -------------------------------------------------------------------------
    def _resolve_gicp(self):
        if self.use_pctools_gicp:
            if self.use_imu:
                try:
                    from Pctools import apply_gicp_with_init_full
                    self.get_logger().info("Using Pctools.apply_gicp_with_init_full (H matrix)")
                    return apply_gicp_with_init_full
                except Exception as e:
                    self.get_logger().warn(f"Pctools.apply_gicp_with_init_full unavailable: {e}")
                try:
                    from Pctools import apply_gicp_with_init
                    self.get_logger().info("Using Pctools.apply_gicp_with_init (scalar conf fallback)")
                    def _wrap_init(src, tgt, init_T=None):
                        return apply_gicp_with_init(src, tgt, init_T), None
                    return _wrap_init
                except Exception as e2:
                    self.get_logger().warn(f"Pctools.apply_gicp_with_init unavailable: {e2}")
            try:
                from Pctools import apply_gicp_direct
                def _wrap_direct(src, tgt, init_T=None):
                    return apply_gicp_direct(src, tgt), None
                self.get_logger().info("Using Pctools.apply_gicp_direct (no init_T)")
                return _wrap_direct
            except Exception as e3:
                self.get_logger().warn(f"Pctools import failed: {e3}. Using Open3D fallback.")

        def _o3d_gicp(src, tgt, init_T=None):
            T = apply_gicp_open3d(src, tgt,
                                  init_T=init_T if self.use_imu else None,
                                  voxel_size=self.gicp_voxel_size,
                                  max_corr_distance=self.gicp_max_corr_distance,
                                  max_iterations=self.gicp_max_iterations)
            return T, None
        self.get_logger().info("Using Open3D GICP")
        return _o3d_gicp

    # -------------------------------------------------------------------------
    # Post-GICP observation hook.
    # Called inside cb_cloud right after the GICP/non-rep iESKF update (state is
    # post-update; debug row + odom are published afterwards).  No-op here so v1
    # behaviour is unchanged; LioNodeV2 overrides it to add Super-LIO
    # point-to-plane refinement against the local submap.
    #   scan_cloud  : current scan, body/sensor frame (== self.prev_cloud once set)
    # -------------------------------------------------------------------------
    def _post_gicp_update(self, scan_cloud) -> None:
        return

    # -------------------------------------------------------------------------
    # Scan measurement update (overridable).  Default: apply the non-rep GICP
    # relative-pose update to the iESKF.  Returns (accepted, chi2).
    # LioNodeV2 overrides this to optionally run Super-LIO point-to-plane first
    # and use this GICP update only as a low-P2P-confidence fallback.
    # -------------------------------------------------------------------------
    def _gicp_measurement_update(self, T, R_n, reg_conf, scan_cloud):
        accepted, chi2 = self.ieskf.update(T, R_n, self._chi2_threshold)
        if not accepted:
            self.get_logger().warn(
                f"Scan {self.scan_counter}: GICP update REJECTED "
                f"(χ²={chi2:.1f} > {self._chi2_threshold:.1f}) — "
                f"conf={reg_conf:.2f}, IMU holds"
            )
        elif reg_conf < self.gicp_min_conf:
            self.get_logger().warn(
                f"Scan {self.scan_counter}: low GICP conf "
                f"({reg_conf:.2f}), χ²={chi2:.1f}"
            )
        return accepted, chi2

    # -------------------------------------------------------------------------
    # IIR Butterworth filter helper (runs at full 200 Hz, sample-by-sample)
    # -------------------------------------------------------------------------
    def _fir_filter_imu(self, omega: np.ndarray, accel: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Pass one IMU sample through the 2nd-order Butterworth IIR filter.

        Replaces the old 41-tap FIR. Group delay ≈ 10-20 ms vs 100 ms.
        State zi is maintained across calls for phase continuity.
        On the first call, zi is initialised to DC steady-state so the filter
        output starts at the correct level without a transient.
        """
        if not self._iir_initialized:
            # Warm up zi to DC steady-state of the first sample
            for ch in range(3):
                _, self._iir_zi_omega[ch] = sosfilt(
                    self._iir_sos, [omega[ch]], zi=self._iir_zi_omega[ch] * omega[ch])
                _, self._iir_zi_accel[ch] = sosfilt(
                    self._iir_sos, [accel[ch]], zi=self._iir_zi_accel[ch] * accel[ch])
            self._iir_initialized = True

        y_omega = np.empty(3, dtype=np.float64)
        y_accel = np.empty(3, dtype=np.float64)
        for ch in range(3):
            out_o, self._iir_zi_omega[ch] = sosfilt(
                self._iir_sos, [omega[ch]], zi=self._iir_zi_omega[ch])
            out_a, self._iir_zi_accel[ch] = sosfilt(
                self._iir_sos, [accel[ch]], zi=self._iir_zi_accel[ch])
            y_omega[ch] = out_o[0]
            y_accel[ch] = out_a[0]
        return y_omega, y_accel

    # -------------------------------------------------------------------------
    # IMU callback — FIR at 200 Hz, propagate at full 200 Hz (no decimation)
    # -------------------------------------------------------------------------
    def cb_imu(self, msg: Imu):
        stamp_hw = ros_time_to_sec(msg.header.stamp)    # hardware timestamp
        omega = np.array([msg.angular_velocity.x,
                          msg.angular_velocity.y,
                          msg.angular_velocity.z], dtype=float)
        accel = np.array([msg.linear_acceleration.x,
                          msg.linear_acceleration.y,
                          msg.linear_acceleration.z], dtype=float) * self.imu_accel_scale

        # FIR runs at full 200 Hz — smoothing only, no decimation
        omega, accel = self._fir_filter_imu(omega, accel)

        # Effective timestamp: shift back by FIR group delay so that
        # propagation dt is computed at the time the signal was actually measured.
        # last_imu_stamp_sec keeps the hardware time for bridge/freshness checks.
        stamp_eff = stamp_hw - self._fir_group_delay_sec

        self.last_imu_stamp_sec = stamp_hw
        self._last_omega = omega
        self._last_accel = accel

        # Buffer for deskewing and delayed-update repropagate
        self._imu_buffer.append((stamp_eff, omega.copy(), accel.copy()))

        # Gravity / bias static init (every sample)
        if self.imu_use_accel and not self.ieskf.gravity_initialized:
            just_initialized = self.ieskf.collect_static_sample(
                omega, accel, self.ieskf.R)
            if just_initialized:
                g  = self.ieskf.g_world
                bg = self.ieskf.bg
                self.get_logger().info(
                    f"Static init complete — "
                    f"g_world=[{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}] "
                    f"norm={np.linalg.norm(g):.3f} m/s²  |  "
                    f"gyro_bias=[{bg[0]:.4f}, {bg[1]:.4f}, {bg[2]:.4f}] rad/s"
                )
                if g[2] < 0.0:
                    self.get_logger().warn(
                        "g_world z-component is negative after flip — "
                        "sensor may be z-down mounted. "
                        "Set imu_gravity_z_down: true in lio.yaml to suppress this warning."
                    )

        # Propagate at full 200 Hz using effective (delay-compensated) timestamps.
        # Does not wait for gravity_initialized — default g_world=[0,0,9.81] is
        # valid for z-up sensors and keeps P growing to prevent gain collapse.
        if (self.last_scan_stamp_sec is not None
                and self._last_prop_stamp is not None):
            dt = stamp_eff - self._last_prop_stamp
            if 0.0 < dt < 0.05:   # sanity: normal IMU interval ≤ 50 ms
                self.ieskf.propagate(omega, accel, dt,
                                     use_accel=self.imu_use_accel)

            # --- ZUPT: inject v=0 when robot is stationary -------------------
            # Checked every IMU tick so drift is bounded during GICP-failure
            # windows.  Uses bias-corrected values to avoid false triggers.
            if self.imu_use_accel:
                omega_c = omega - self.ieskf.bg
                accel_c = accel - self.ieskf.ba
                a_world_residual = float(np.linalg.norm(
                    self.ieskf.R @ accel_c - self.ieskf.g_world))
                if (float(np.linalg.norm(omega_c)) < self.zupt_omega_thresh
                        and a_world_residual < self.zupt_accel_thresh):
                    self.ieskf.zupt_update(self.zupt_sigma_v)

            # --- Soft z-constraint: floor pseudo-measurement -----------------
            # Applied every IMU tick so z-accel bias remains observable even
            # when GICP is rejecting updates.  Replaces hard zeroing.
            if self.force_z_zero and self.soft_z_sigma > 0.0:
                self.ieskf.soft_z_update(self.soft_z_sigma)

        self._last_prop_stamp = stamp_eff

    # -------------------------------------------------------------------------
    # IMU freshness check (unchanged logic)
    # -------------------------------------------------------------------------
    def _imu_is_fresh(self, stamp_sec: float) -> bool:
        if self.last_imu_stamp_sec is None:
            return False
        return abs(stamp_sec - self.last_imu_stamp_sec) <= self.imu_timeout_sec

    # -------------------------------------------------------------------------
    # GICP initial-guess helpers (unchanged logic, updated references)
    # -------------------------------------------------------------------------
    def _imu_confidence(self, current_stamp_sec: float) -> float:
        if not self.use_imu or not self._imu_is_fresh(current_stamp_sec):
            return 0.0
        if not self.imu_use_accel:
            return 0.6  # gyro-only: good rotation hint, no position prediction
        # Full IMU: 0.9 once gravity is refined, 0.6 while still on default g_world
        return 0.9 if self.ieskf.gravity_initialized else 0.6

    def _classify_motion_at_scan(self) -> Tuple[str, float, float]:
        """Classify motion state at scan time from inter-scan IMU buffer.

        Returns (state, omega_rms_rad_s, accel_rms_m_s2) where state is one of:
          'stationary' — skip GICP; ZUPT holds position
          'rotating'   — inflate GICP rotation noise; gyro propagation holds rotation
          'translating'— deflate GICP position noise; GICP drives translation
          'combined'   — both; partial adjustments
          'mild'       — low motion below thresholds; use normal noise
          'unknown'    — IMU unavailable; use normal noise
        """
        if not self.use_imu or not self.imu_use_accel or len(self._imu_buffer) == 0:
            return 'unknown', 0.0, 0.0

        t_since = self.last_scan_stamp_sec if self.last_scan_stamp_sec is not None else -1.0
        samples = [(om, ac) for (t, om, ac) in self._imu_buffer if t > t_since]
        if not samples:
            _, om, ac = self._imu_buffer[-1]
            samples = [(om, ac)]

        bg = self.ieskf.bg
        ba = self.ieskf.ba
        R  = self.ieskf.R
        g  = self.ieskf.g_world

        omega_sq  = [float(np.dot(om - bg, om - bg)) for om, _ in samples]
        omega_rms = float(np.sqrt(np.mean(omega_sq)))

        accel_res = [float(np.linalg.norm(R @ (ac - ba) - g)) for _, ac in samples]
        accel_rms = float(np.mean(accel_res))

        if (omega_rms < self.motion_omega_stationary
                and accel_rms < self.motion_accel_stationary):
            return 'stationary', omega_rms, accel_rms

        is_rot   = omega_rms > self.motion_omega_rotating
        is_trans = accel_rms > self.motion_accel_translating

        if is_rot and is_trans:
            return 'combined', omega_rms, accel_rms
        if is_rot:
            return 'rotating', omega_rms, accel_rms
        if is_trans:
            return 'translating', omega_rms, accel_rms
        return 'mild', omega_rms, accel_rms

    # -------------------------------------------------------------------------
    # LiDAR scan deskewing (rotation-only using IMU buffer)
    # -------------------------------------------------------------------------
    def _deskew_cloud(self, xyz: np.ndarray,
                      stamps_sec: Optional[np.ndarray],
                      t_scan_sec: float) -> np.ndarray:
        """Rotate each point from its capture time into the scan reference frame
        (t_scan_sec) using gyro integration from the IMU buffer.

        Returns xyz unchanged if no per-point timestamps or IMU buffer is empty.
        """
        if stamps_sec is None or xyz.shape[0] == 0 or len(self._imu_buffer) == 0:
            return xyz

        t_min = float(stamps_sec.min())
        if t_scan_sec - t_min < 1e-4:
            return xyz   # all points at same time — nothing to do

        # Collect relevant IMU samples covering the scan window
        buf = [(t, w, a) for (t, w, a) in self._imu_buffer if t >= t_min - 0.005]
        if len(buf) < 2:
            return xyz

        if buf[-1][0] < t_scan_sec:
            buf.append((t_scan_sec, self._last_omega.copy(), self._last_accel.copy()))

        # Forward-integrate rotations from t_min to each IMU step (gyro only)
        # R_accum[i] = R from t_min to buf[i][0]
        R_cur = np.eye(3)
        timeline: List[Tuple[float, np.ndarray]] = [(buf[0][0], R_cur.copy())]
        for i in range(1, len(buf)):
            dt = buf[i][0] - buf[i - 1][0]
            if not (0.0 < dt < 0.05):
                timeline.append((buf[i][0], R_cur.copy()))
                continue
            omega_c = buf[i - 1][1] - self.ieskf.bg
            R_cur = R_cur @ so3_exp(omega_c * dt)
            timeline.append((buf[i][0], R_cur.copy()))

        # R at the scan reference time
        R_scan = timeline[-1][1]   # R_{t_min → t_scan}

        # For each point: R_{t_i → t_scan} = R_scan.T @ R_{t_min → t_i}
        xyz_out = xyz.copy()
        for i in range(len(timeline) - 1):
            t_lo, R_lo = timeline[i]
            t_hi, _    = timeline[i + 1]
            mask = (stamps_sec >= t_lo) & (stamps_sec < t_hi)
            if not mask.any():
                continue
            R_i_to_scan = R_scan.T @ R_lo
            xyz_out[mask] = (R_i_to_scan @ xyz[mask].T).T

        return xyz_out

    # -------------------------------------------------------------------------
    # Build 6×6 measurement noise covariance from small_gicp H matrix
    # -------------------------------------------------------------------------
    def _build_gicp_noise_cov(self,
                               H_gicp: Optional[np.ndarray],
                               reg_conf: float,
                               pos_scale: float = 1.0,
                               rot_scale: float = 1.0) -> np.ndarray:
        """Convert small_gicp Hessian H (order [rot(3), trans(3)]) to
        measurement noise cov R_n in [pos(3), rot(3)] order for iESKF.

        pos_scale / rot_scale: motion-adaptive multipliers applied via
        diagonal similarity transform S·R_n·S so positive-definiteness is
        preserved regardless of which path (Hessian or scalar) built R_n.

        Falls back to scalar diagonal when H is unavailable or ill-conditioned.
        """
        if H_gicp is not None and H_gicp.shape == (6, 6):
            # small_gicp tangent order: [rot(3), trans(3)]
            # Reorder to our [pos(3)=trans, rot(3)] convention
            idx = [3, 4, 5, 0, 1, 2]
            H_reordered = H_gicp[np.ix_(idx, idx)]
            try:
                cov = np.linalg.inv(H_reordered) * self._gicp_cov_scale
                # Ensure positive definiteness
                eigvals = np.linalg.eigvalsh(cov)
                if eigvals.min() > 0.0:
                    R_n = cov
                    # Fall through to apply motion scales below
                else:
                    R_n = None
            except np.linalg.LinAlgError:
                R_n = None
        else:
            R_n = None

        if R_n is None:
            # Scalar fallback: inflate by inverse confidence
            noise_scale = max(1.0, 1.0 / max(reg_conf, 0.05))
            R_n = np.zeros((6, 6))
            R_n[0:3, 0:3] = np.eye(3) * (self.ieskf_meas_noise_pos * noise_scale) ** 2
            R_n[3:6, 3:6] = np.eye(3) * (self.ieskf_meas_noise_rot * noise_scale) ** 2

        # Apply motion-adaptive diagonal scaling: R_n_scaled = S @ R_n @ S
        # where S = diag(pos_scale×I_3, rot_scale×I_3).
        if pos_scale != 1.0 or rot_scale != 1.0:
            s = np.concatenate([np.full(3, pos_scale), np.full(3, rot_scale)])
            R_n = (s[:, None] * R_n) * s[None, :]

        return R_n

    # -------------------------------------------------------------------------
    # IMU-only odometry publisher
    # -------------------------------------------------------------------------
    def _publish_imu_only_odom(self, stamp, p: np.ndarray, R: np.ndarray,
                                v: np.ndarray) -> None:
        if self.imu_odom_pub is None:
            return
        qx, qy, qz, qw = rot_to_quat(R)
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = self.map_frame
        odom.child_frame_id  = self.base_frame + "_imu"
        odom.pose.pose.position.x    = float(p[0])
        odom.pose.pose.position.y    = float(p[1])
        odom.pose.pose.position.z    = float(p[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        v_body = R.T @ v
        odom.twist.twist.linear.x = float(v_body[0])
        odom.twist.twist.linear.y = float(v_body[1])
        odom.twist.twist.linear.z = float(v_body[2])
        self.imu_odom_pub.publish(odom)

    def _build_nonrep_init_guess(self, pred_pose: np.ndarray) -> np.ndarray:
        """Convert non-rep absolute predicted pose [x,y,z,yaw] → relative T_B1_B2."""
        st = self.processor.get_current_state()
        if st is None:
            return np.eye(4, dtype=float)

        prev = st.pose
        delta_p_world = pred_pose[:3] - prev[:3]
        delta_yaw = float(np.arctan2(
            np.sin(pred_pose[3] - prev[3]),
            np.cos(pred_pose[3] - prev[3]),
        ))
        cy, sy = np.cos(delta_yaw), np.sin(delta_yaw)
        dR = np.array([[cy, -sy, 0.0],
                       [sy,  cy, 0.0],
                       [0.0, 0.0, 1.0]], dtype=float)
        # Express translation in previous body frame (use last-scan rotation)
        dp_body = self.ieskf._R_last.T @ delta_p_world

        T = np.eye(4, dtype=float)
        T[:3, :3] = dR
        T[:3, 3]  = dp_body
        return T

    def _merge_init_guesses(self,
                             T_imu: np.ndarray, w_imu: float,
                             T_nonrep: np.ndarray, w_nonrep: float) -> np.ndarray:
        """Blend two relative transforms in tangent space."""
        total = w_imu + w_nonrep
        if total < 1e-9:
            return np.eye(4, dtype=float)
        wi = w_imu / total
        wn = w_nonrep / total
        dp  = wi * T_imu[:3, 3] + wn * T_nonrep[:3, 3]
        aa  = wi * so3_log(T_imu[:3, :3]) + wn * so3_log(T_nonrep[:3, :3])
        T   = np.eye(4, dtype=float)
        T[:3, :3] = so3_exp(aa)
        T[:3, 3]  = dp
        return T

    # -------------------------------------------------------------------------
    # Scan buffering / decimation (unchanged)
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
    # Publishers (updated to use iESKF covariance)
    # -------------------------------------------------------------------------
    def _publish_odom_and_tf(self, stamp,
                              p_world: np.ndarray, R_world: np.ndarray,
                              v_world: np.ndarray):
        x, y, z = float(p_world[0]), float(p_world[1]), float(p_world[2])

        if self.use_imu:
            qx, qy, qz, qw = rot_to_quat(R_world)
            parent_frame = self.map_frame
        else:
            yaw = float(np.arctan2(R_world[1, 0], R_world[0, 0]))
            qx, qy, qz, qw = yaw_to_quat(yaw)
            parent_frame = self.odom_frame

        if self.odom_pub is not None:
            odom = Odometry()
            odom.header.stamp      = stamp
            odom.header.frame_id   = parent_frame
            odom.child_frame_id    = self.base_frame
            odom.pose.pose.position.x    = x
            odom.pose.pose.position.y    = y
            odom.pose.pose.position.z    = z
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw
            v_body = R_world.T @ v_world
            odom.twist.twist.linear.x = float(v_body[0])
            odom.twist.twist.linear.y = float(v_body[1])
            odom.twist.twist.linear.z = float(v_body[2])
            cov6 = self.ieskf.pose_covariance_6x6
            odom.pose.covariance = cov6.reshape(-1).tolist()
            self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp    = stamp
            t.header.frame_id = parent_frame
            t.child_frame_id  = self.base_frame
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)

            if not self.use_imu:
                t2 = TransformStamped()
                t2.header.stamp    = stamp
                t2.header.frame_id = self.map_frame
                t2.child_frame_id  = self.odom_frame
                t2.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t2)

    def _publish_map_cloud(self, stamp):
        if self.map_pub is None:
            return
        if self.scan_counter % self.map_publish_every_n_scans != 0:
            return
        # VoxelHashMap already stores one point per voxel (= implicit DS at map_voxel
        # resolution). to_open3d() caps at map_publish_max_points via random sampling.
        cloud_to_pub = self.voxel_map.to_open3d(
            max_points=self.map_publish_max_points if self.map_publish_max_points > 0
            else 800_000)
        header = Header()
        header.stamp    = stamp
        header.frame_id = self.map_frame
        self.map_pub.publish(open3d_cloud_to_pointcloud2_xyzi(cloud_to_pub, header))

    # -------------------------------------------------------------------------
    # LiDAR callback — core LIO pipeline
    # -------------------------------------------------------------------------
    def cb_cloud(self, msg: PointCloud2):
        self.msg_counter += 1

        # Extract points with per-point timestamps (if available for deskewing)
        xyz, intensity, pt_stamps = pointcloud2_to_xyz_i_stamps(msg)

        # Deskew using IMU before buffering/accumulation
        stamp             = msg.header.stamp
        current_stamp_sec = ros_time_to_sec(stamp)
        if self.use_imu and len(self._imu_buffer) > 0:
            xyz = self._deskew_cloud(xyz, pt_stamps, current_stamp_sec)

        cloud_raw = xyzi_to_open3d_cloud(xyz, intensity)
        cloud = self._flush_or_buffer(cloud_raw)
        if cloud is None or len(cloud.points) == 0:
            return
        if self.max_scans is not None and self.scan_counter >= self.max_scans:
            return

        # Bridge the gap between the last filtered-IMU effective time and scan time.
        # cb_imu propagates to _last_prop_stamp (effective); the bridge covers the
        # remaining interval to current_stamp_sec using a zero-order hold.
        imu_ready = (self.use_imu
                     and self.last_scan_stamp_sec is not None
                     and self._last_prop_stamp is not None)
        if imu_ready:
            dt_gap = current_stamp_sec - self._last_prop_stamp
            if 0.0 < dt_gap < 0.15:   # allow up to ~FIR_delay + one IMU period
                self.ieskf.propagate(self._last_omega, self._last_accel, dt_gap,
                                     use_accel=self.imu_use_accel)

        # Save full state snapshot at scan time — used to roll back if repropagate needed
        state_snap_at_scan = self.ieskf.save_full_snapshot()
        last_prop_at_scan  = self._last_prop_stamp

        # Classify motion state from inter-scan IMU window (requires accel to be on)
        motion_state, _omega_rms, _accel_rms = self._classify_motion_at_scan()

        # Debug sentinels — overwritten below as each step completes
        _dbg_imu_p   = self.ieskf.p.copy()
        _dbg_imu_R   = self.ieskf.R.copy()
        _dbg_T       = None
        _dbg_conf    = float('nan')
        _dbg_chi2    = float('nan')
        _dbg_acc     = -1        # -1 = GICP did not run
        _dbg_sub     = False

        try:
            feat = self.processor.extract_scan_features(cloud)

            # ------ Non-rep adaptive prediction ---------------------------
            pred_pose, pred_conf = self.processor.predict_pose_adaptive(feat)

            # ------ GICP initial guess: iESKF prediction + non-rep --------
            T_imu   = np.eye(4, dtype=float)
            w_imu   = 0.0
            if self.use_imu and self._imu_is_fresh(current_stamp_sec):
                T_imu = self.ieskf.get_predicted_relative_transform()
                w_imu = self._imu_confidence(current_stamp_sec) * self.imu_base_weight

            T_nonrep = np.eye(4, dtype=float)
            w_nonrep = 0.0
            if self.use_imu and pred_pose is not None and self.processor.get_current_state() is not None:
                T_nonrep = self._build_nonrep_init_guess(pred_pose)
                w_nonrep = float(pred_conf) * self.nonrep_base_weight

            T_init = self._merge_init_guesses(T_imu, w_imu, T_nonrep, w_nonrep)

            # ------ Capture IMU-only state BEFORE GICP corrects it --------
            # This is the pure propagated (dead-reckoning) pose at scan time.
            imu_only_p = self.ieskf.p.copy()
            imu_only_R = self.ieskf.R.copy()
            imu_only_v = self.ieskf.v.copy()
            _dbg_imu_p = imu_only_p
            _dbg_imu_R = imu_only_R

            # ------ GICP registration -------------------------------------
            if self.prev_cloud is not None and len(self.prev_cloud.points) > 0:

                # Stationary: robot not moving — skip GICP entirely.
                # IMU/ZUPT already hold position; no wasted registration compute.
                if (self.use_imu and self.motion_skip_stationary
                        and motion_state == 'stationary'):
                    yaw      = float(np.arctan2(self.ieskf.R[1, 0], self.ieskf.R[0, 0]))
                    obs_pose = np.array([self.ieskf.p[0], self.ieskf.p[1],
                                         self.ieskf.p[2], yaw])
                    self.processor.update_with_observation(
                        obs_pose, feat, 0.0, pred_pose)

                else:
                    # Motion-adaptive noise scales:
                    #   rotating    → inflate GICP rot noise; gyro propagation holds rotation
                    #   translating → deflate GICP pos noise; GICP owns translation
                    #   combined    → partial rotation inflation
                    _pos_scale, _rot_scale = 1.0, 1.0
                    if motion_state == 'rotating':
                        _rot_scale = self.motion_rot_noise_scale
                    elif motion_state == 'translating':
                        _pos_scale = self.motion_trans_pos_noise_scale
                    elif motion_state == 'combined':
                        _rot_scale = max(1.0, self.motion_rot_noise_scale * 0.5)

                    # --- Choose GICP reference: local submap or previous scan ---
                    # Scan-to-submap GICP: extract bounded local cloud from the
                    # voxel hash map instead of matching against the full history.
                    # Falls back to scan-to-scan when the map is still being built.
                    # Submap GICP requires a valid init_T to bridge sensor→world frames.
                    # Without IMU there is no init_T, so scan-to-scan is used instead.
                    use_submap = (self.use_imu
                                  and self.gicp_submap_radius > 0
                                  and len(self.voxel_map) >= self._submap_min_voxels)
                    if use_submap:
                        gicp_ref = self.voxel_map.get_submap(
                            self.ieskf.p, self.gicp_submap_radius)
                        if len(gicp_ref.points) < 30:
                            gicp_ref = self.prev_cloud
                            use_submap = False
                    else:
                        gicp_ref = self.prev_cloud

                    # For scan-to-submap the init guess must be in world frame;
                    # for scan-to-scan it stays as the relative T_B1_B2.
                    if use_submap:
                        T_world_prev = np.eye(4, dtype=float)
                        T_world_prev[:3, :3] = self.ieskf._R_last
                        T_world_prev[:3, 3]  = self.ieskf._p_last
                        T_init_gicp = T_world_prev @ T_init   # relative → absolute
                    else:
                        T_init_gicp = T_init

                    T_raw, H_gicp = self.apply_gicp_func(gicp_ref, cloud, T_init_gicp)

                    # Convert GICP result back to relative T_B1_B2 for iESKF.update
                    if use_submap:
                        T = np.linalg.inv(T_world_prev) @ T_raw
                    else:
                        T = T_raw
                    _dbg_T   = T
                    _dbg_sub = use_submap

                    # Repropagate: if new IMU arrived during GICP, roll back to scan
                    # time, apply update, then replay the buffered samples.
                    # (In single-threaded mode this is a no-op; matters for live use.)
                    new_imu_since_scan = [
                        s for s in self._imu_buffer
                        if s[0] > (last_prop_at_scan or -1.0)
                    ]
                    if self.use_imu and len(new_imu_since_scan) > 0:
                        self.ieskf.restore_full_snapshot(state_snap_at_scan)
                        self._last_prop_stamp = last_prop_at_scan

                    reg_conf = estimate_registration_confidence(self.prev_cloud, cloud, T)
                    _dbg_conf = reg_conf

                    # Build 6×6 measurement noise with motion-adaptive pos/rot scales
                    R_n = self._build_gicp_noise_cov(
                        H_gicp, reg_conf, _pos_scale, _rot_scale)

                    if self.use_imu:
                        # Scan measurement update (overridable).  v1: GICP pose
                        # update.  v2 may run point-to-plane first and gate this
                        # non-rep GICP update on P2P confidence.
                        accepted, chi2 = self._gicp_measurement_update(
                            T, R_n, reg_conf, cloud)
                        _dbg_chi2 = chi2
                        _dbg_acc  = int(accepted)

                        # Repropagate any buffered IMU samples that arrived during GICP
                        if len(new_imu_since_scan) > 0:
                            for (t_s, om, ac) in new_imu_since_scan:
                                if self._last_prop_stamp is not None:
                                    dt_r = t_s - self._last_prop_stamp
                                    if 0.0 < dt_r < 0.05:
                                        self.ieskf.propagate(om, ac, dt_r,
                                                             use_accel=self.imu_use_accel)
                                self._last_prop_stamp = t_s
                    else:
                        # IMU disabled — direct state integration from GICP
                        dR      = T[:3, :3]
                        dp_body = T[:3, 3]
                        dt_scan = max(1e-6, current_stamp_sec - self.last_scan_stamp_sec
                                      if self.last_scan_stamp_sec else 1.0)
                        p_prev  = self.ieskf.p.copy()
                        self.ieskf.p = self.ieskf.p + self.ieskf.R @ dp_body
                        self.ieskf.R = self.ieskf.R @ dR
                        U, _, Vt = np.linalg.svd(self.ieskf.R)
                        self.ieskf.R = U @ Vt
                        self.ieskf.v = (self.ieskf.p - p_prev) / dt_scan

                    # Super-LIO point-to-plane refinement against the local
                    # submap (no-op in v1; LioNodeV2 overrides).  Runs after the
                    # GICP/non-rep update so both observations correct this scan.
                    self._post_gicp_update(cloud)

                    if self.force_z_zero:
                        if self.soft_z_sigma > 0.0:
                            # Soft constraint: correct p_z toward 0, let ba_z adjust
                            self.ieskf.soft_z_update(self.soft_z_sigma)
                        else:
                            # Legacy hard zero (used when soft_z_sigma: 0 in config)
                            self.ieskf.p[2] = 0.0
                            self.ieskf.v[2] = 0.0

                    # Feed non-rep processor
                    yaw      = float(np.arctan2(self.ieskf.R[1, 0], self.ieskf.R[0, 0]))
                    obs_pose = np.array([self.ieskf.p[0], self.ieskf.p[1],
                                         self.ieskf.p[2], yaw])
                    self.processor.update_with_observation(
                        obs_pose, feat, reg_conf, pred_pose)

            else:
                # First scan — initialise at origin
                self.processor.update_with_observation(
                    np.array([0.0, 0.0, 0.0, 0.0]), feat, 0.3, pred_pose)

            # Snapshot state for next interval's relative-transform query
            self.ieskf.save_scan_state()

            # ------ Publish IMU-only odometry (dead-reckoning, no LiDAR) --
            self._publish_imu_only_odom(stamp, imu_only_p, imu_only_R, imu_only_v)

            # ------ Publish fused odometry --------------------------------
            self._publish_odom_and_tf(stamp, self.ieskf.p, self.ieskf.R, self.ieskf.v)

            # ------ Voxel hash map insert (O(N) bounded, no periodic DS) --
            Tmap = np.eye(4, dtype=float)
            if self.use_imu:
                Tmap[:3, :3] = self.ieskf.R
            else:
                yaw_map = float(np.arctan2(self.ieskf.R[1, 0], self.ieskf.R[0, 0]))
                Tmap[0, 0] =  np.cos(yaw_map); Tmap[0, 1] = -np.sin(yaw_map)
                Tmap[1, 0] =  np.sin(yaw_map); Tmap[1, 1] =  np.cos(yaw_map)
            Tmap[:3, 3] = self.ieskf.p
            cur_pts = np.asarray(cloud.points, dtype=np.float64)
            if len(cur_pts) > 0:
                cur_pts_map = (Tmap[:3, :3] @ cur_pts.T + Tmap[:3, 3:4]).T
                cur_cols = (np.asarray(cloud.colors, dtype=np.float64)
                            if cloud.has_colors() else None)
                self.voxel_map.insert(cur_pts_map.astype(np.float32), cur_cols)

            # Range-based pruning every 20 scans — removes voxels that drifted
            # beyond prune_radius from the current position (no batch DS needed).
            if self.scan_counter % 20 == 0 and len(self.voxel_map) > 0:
                n_pruned = self.voxel_map.prune_far(self.ieskf.p)
                if n_pruned > 0:
                    self.get_logger().debug(
                        f"Pruned {n_pruned} voxels — map: {len(self.voxel_map)}")

            self._publish_map_cloud(stamp)

        except Exception as e:
            import traceback
            self.get_logger().error(
                f"Scan {self.scan_counter} error: {e}\n{traceback.format_exc()}")

        self.prev_cloud        = cloud
        self.last_scan_stamp_sec = current_stamp_sec

        # ---- Write debug CSV row (one line per scan) ----
        if self._debug_fh is not None:
            _yf   = float(np.arctan2(self.ieskf.R[1, 0], self.ieskf.R[0, 0]))
            _yimu = float(np.arctan2(_dbg_imu_R[1, 0], _dbg_imu_R[0, 0]))
            if _dbg_T is not None:
                _tdx  = float(_dbg_T[0, 3])
                _tdy  = float(_dbg_T[1, 3])
                _tdz  = float(_dbg_T[2, 3])
                _tdyw = float(np.arctan2(_dbg_T[1, 0], _dbg_T[0, 0]))
            else:
                _tdx = _tdy = _tdz = _tdyw = float('nan')
            self._debug_fh.write(
                f"{self.scan_counter},{current_stamp_sec:.6f},"
                f"{self.ieskf.p[0]:.6f},{self.ieskf.p[1]:.6f},{self.ieskf.p[2]:.6f},{_yf:.6f},"
                f"{_dbg_imu_p[0]:.6f},{_dbg_imu_p[1]:.6f},{_dbg_imu_p[2]:.6f},{_yimu:.6f},"
                f"{_tdx},{_tdy},{_tdz},{_tdyw},"
                f"{_dbg_conf},{_dbg_chi2},{_dbg_acc},"
                f"{motion_state},{len(self.voxel_map)},{len(cloud.points)},{int(_dbg_sub)}\n"
            )

        self.scan_counter      += 1

        if self.max_scans is not None and self.scan_counter >= self.max_scans:
            self.get_logger().info("Max scans reached. Shutting down.")
            rclpy.shutdown()

    def shutdown(self):
        if self._debug_fh is not None:
            self._debug_fh.flush()
            self._debug_fh.close()
            self._debug_fh = None


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
