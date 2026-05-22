#!/usr/bin/env python3
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
        if abs(float(np.linalg.norm(a_c)) - self.gravity_mag) < 0.5:
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
        meas_noise_pos: float,
        meas_noise_rot: float,
    ) -> None:
        """
        Iterated EKF measurement update using the GICP relative transform.

        T_gicp          4×4 T_B1_B2 from GICP (maps current B2 → previous B1)
        meas_noise_pos  position measurement std [m]  — scaled by GICP confidence
        meas_noise_rot  rotation measurement std [rad] — scaled by GICP confidence
        """
        # Convert relative GICP transform to absolute pose measurement
        p_meas = self._p_last + self._R_last @ T_gicp[:3, 3]
        R_meas = self._R_last @ T_gicp[:3, :3]

        # Measurement noise covariance R_n (6×6)
        R_n = np.zeros((6, 6))
        R_n[0:3, 0:3] = np.eye(3) * (meas_noise_pos ** 2)
        R_n[3:6, 3:6] = np.eye(3) * (meas_noise_rot ** 2)

        # Measurement Jacobian H (6×15):  H = [I 0 0 0 0 ; 0 0 I 0 0]
        # H is constant (doesn't depend on state), so K is computed once.
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)   # position row: measurement = p + δp
        H[3:6, 6:9] = np.eye(3)   # rotation row: measurement = θ + δθ

        S = H @ self.P @ H.T + R_n
        K = self.P @ H.T @ np.linalg.inv(S)

        # Iterate: re-evaluate the SO(3) innovation around the current estimate
        p_k  = self.p.copy()
        v_k  = self.v.copy()
        R_k  = self.R.copy()
        bg_k = self.bg.copy()
        ba_k = self.ba.copy()

        for _ in range(self.max_iters):
            z_p = p_meas - p_k
            z_R = so3_log(R_k.T @ R_meas)      # rotation error on manifold
            z   = np.concatenate([z_p, z_R])

            dx = K @ z

            p_k  = p_k  + dx[0:3]
            v_k  = v_k  + dx[3:6]
            R_k  = R_k  @ so3_exp(dx[6:9])
            bg_k = bg_k + dx[9:12]
            ba_k = ba_k + dx[12:15]

            if float(np.linalg.norm(dx)) < self.conv_threshold:
                break

        # Commit nominal state
        self.p  = p_k
        self.v  = v_k
        U, _, Vt = np.linalg.svd(R_k)
        self.R  = U @ Vt
        self.bg = bg_k
        self.ba = ba_k

        # Joseph-form covariance update (numerically stable)
        I_KH   = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_n @ K.T

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
    n_pts = msg.width * msg.height
    if n_pts == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    field_offsets = {f.name: f.offset for f in msg.fields}
    step = msg.point_step
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_pts, step)

    def _col(off: int) -> np.ndarray:
        return raw[:, off:off + 4].copy().ravel().view(np.float32)

    xyz = np.column_stack([
        _col(field_offsets['x']),
        _col(field_offsets['y']),
        _col(field_offsets['z']),
    ])
    intensity = (_col(field_offsets['intensity']) if 'intensity' in field_offsets
                 else np.zeros(n_pts, dtype=np.float32))
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid], intensity[valid]


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
        gyro_bias_raw  = p("imu_gyro_bias",  [0.0, 0.0, 0.0])
        accel_bias_raw = p("imu_accel_bias", [0.0, 0.0, 0.0])

        # ---- iESKF noise parameters
        self.ieskf_sigma_gyro     = float(p("ieskf_sigma_gyro",     0.005))
        self.ieskf_sigma_accel    = float(p("ieskf_sigma_accel",    0.05))
        self.ieskf_sigma_bg       = float(p("ieskf_sigma_bg",       1e-4))
        self.ieskf_sigma_ba       = float(p("ieskf_sigma_ba",       1e-3))
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
        self.force_z_zero = bool(p("force_z_zero", False))

        # ---- Visualization / map
        self.visualize  = bool(p("visualize", True))
        self.map_voxel  = float(p("map_voxel", 0.15))

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

        # ---- Scan state
        self.prev_cloud: Optional[o3d.geometry.PointCloud] = None
        self.map_cloud = o3d.geometry.PointCloud()
        self._buffer_cloud = o3d.geometry.PointCloud()
        self.msg_counter   = 0
        self.scan_counter  = 0
        self.last_scan_stamp_sec: Optional[float] = None

        # ---- GICP callable
        self.apply_gicp_func = self._resolve_gicp()

        # ---- Visualizer
        self.viewer = LiveOpen3D(window_name="LIO Map (iESKF)") if self.visualize else None

        # ---- Publishers / subscribers
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10) if self.publish_odom else None
        self.map_pub  = self.create_publisher(PointCloud2, self.map_topic, 1) if self.publish_map  else None
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
                f"σ_gyro={self.ieskf_sigma_gyro}  σ_accel={self.ieskf_sigma_accel}  "
                f"σ_bg={self.ieskf_sigma_bg}  σ_ba={self.ieskf_sigma_ba}"
            )

    # -------------------------------------------------------------------------
    # GICP resolver (unchanged)
    # -------------------------------------------------------------------------
    def _resolve_gicp(self):
        if self.use_pctools_gicp:
            if self.use_imu:
                try:
                    from Pctools import apply_gicp_with_init
                    self.get_logger().info("Using Pctools.apply_gicp_with_init")
                    return apply_gicp_with_init
                except Exception as e:
                    self.get_logger().warn(f"Pctools.apply_gicp_with_init unavailable: {e}")
            # use_imu=False (or apply_gicp_with_init unavailable): use apply_gicp_direct
            # like ros_non_rep.py — cold-start GICP, no init_T
            try:
                from Pctools import apply_gicp_direct
                def _wrap(src, tgt, init_T=None):
                    return apply_gicp_direct(src, tgt)
                self.get_logger().info("Using Pctools.apply_gicp_direct (no init_T)")
                return _wrap
            except Exception as e2:
                self.get_logger().warn(f"Pctools import failed: {e2}. Using Open3D fallback.")

        def _o3d_gicp(src, tgt, init_T=None):
            return apply_gicp_open3d(src, tgt,
                                     init_T=init_T if self.use_imu else None,
                                     voxel_size=self.gicp_voxel_size,
                                     max_corr_distance=self.gicp_max_corr_distance,
                                     max_iterations=self.gicp_max_iterations)
        self.get_logger().info("Using Open3D GICP")
        return _o3d_gicp

    # -------------------------------------------------------------------------
    # IMU callback — continuous iESKF propagation
    # -------------------------------------------------------------------------
    def cb_imu(self, msg: Imu):
        stamp_sec = ros_time_to_sec(msg.header.stamp)
        omega = np.array([msg.angular_velocity.x,
                          msg.angular_velocity.y,
                          msg.angular_velocity.z], dtype=float)
        accel = np.array([msg.linear_acceleration.x,
                          msg.linear_acceleration.y,
                          msg.linear_acceleration.z], dtype=float)

        if not self.imu_use_accel:
            # Gyro-only mode: propagate rotation immediately without waiting
            # for gravity initialisation (no accel integration, no drift).
            if (self.last_scan_stamp_sec is not None
                    and self.last_imu_stamp_sec is not None):
                dt = stamp_sec - self.last_imu_stamp_sec
                if 0.0 < dt < 1.0:
                    self.ieskf.propagate(omega, accel, dt, use_accel=False)
        elif not self.ieskf.gravity_initialized:
            just_initialized = self.ieskf.collect_static_sample(omega, accel, self.ieskf.R)
            if just_initialized:
                g = self.ieskf.g_world
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
        elif self.last_scan_stamp_sec is not None:
            if self.last_imu_stamp_sec is not None:
                dt = stamp_sec - self.last_imu_stamp_sec
                if 0.0 < dt < 1.0:
                    self.ieskf.propagate(omega, accel, dt, use_accel=True)

        self.last_imu_stamp_sec = stamp_sec
        self._last_omega = omega
        self._last_accel = accel

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
        return 0.9 if self.ieskf.is_accel_ready else 0.6

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
        cloud_to_pub = self.map_cloud
        if len(cloud_to_pub.points) > 0 and self.map_publish_voxel > 0:
            cloud_to_pub = cloud_to_pub.voxel_down_sample(float(self.map_publish_voxel))
        if self.map_publish_max_points > 0 and len(cloud_to_pub.points) > self.map_publish_max_points:
            pts  = np.asarray(cloud_to_pub.points)
            cols = np.asarray(cloud_to_pub.colors) if cloud_to_pub.has_colors() else None
            idx  = np.random.choice(len(pts), self.map_publish_max_points, replace=False)
            tmp  = o3d.geometry.PointCloud()
            tmp.points = o3d.utility.Vector3dVector(pts[idx].astype(np.float64, copy=False))
            if cols is not None and len(cols) == len(pts):
                tmp.colors = o3d.utility.Vector3dVector(cols[idx].astype(np.float64, copy=False))
            cloud_to_pub = tmp
        header = Header()
        header.stamp    = stamp
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

        stamp             = msg.header.stamp
        current_stamp_sec = ros_time_to_sec(stamp)

        # Bridge the gap between the last IMU message and the exact scan
        # timestamp.  cb_imu propagates up to last_imu_stamp_sec; any
        # remaining dt is covered here using the most recent IMU sample
        # (zero-order hold), so the IESKF state aligns to scan time.
        imu_ready = (self.use_imu
                     and self.last_scan_stamp_sec is not None
                     and self.last_imu_stamp_sec is not None
                     and (not self.imu_use_accel or self.ieskf.gravity_initialized))
        if imu_ready:
            dt_gap = current_stamp_sec - self.last_imu_stamp_sec
            if 0.0 < dt_gap < 0.1:
                self.ieskf.propagate(self._last_omega, self._last_accel, dt_gap,
                                     use_accel=self.imu_use_accel)

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

            T_nonrep  = np.eye(4, dtype=float)
            w_nonrep  = 0.0
            # Only blend non-rep init guess when IMU is active; with use_imu=False
            # GICP starts from identity, matching ros_non_rep.py behaviour.
            if self.use_imu and pred_pose is not None and self.processor.get_current_state() is not None:
                T_nonrep = self._build_nonrep_init_guess(pred_pose)
                w_nonrep = float(pred_conf) * self.nonrep_base_weight

            T_init = self._merge_init_guesses(T_imu, w_imu, T_nonrep, w_nonrep)

            # ------ GICP registration -------------------------------------
            if self.prev_cloud is not None and len(self.prev_cloud.points) > 0:
                T        = self.apply_gicp_func(self.prev_cloud, cloud, T_init)
                reg_conf = estimate_registration_confidence(self.prev_cloud, cloud, T)

                # Scale measurement noise by inverse of GICP confidence;
                # low confidence → larger noise → filter trusts IMU more.
                noise_scale       = max(1.0, 1.0 / max(reg_conf, 0.05))
                meas_noise_pos    = self.ieskf_meas_noise_pos * noise_scale
                meas_noise_rot    = self.ieskf_meas_noise_rot * noise_scale

                if reg_conf < self.gicp_min_conf and w_imu > 0.0:
                    self.get_logger().warn(
                        f"Scan {self.scan_counter}: low GICP conf "
                        f"({reg_conf:.2f}) — boosting measurement noise ×20"
                    )
                    meas_noise_pos *= 20.0
                    meas_noise_rot *= 20.0

                if self.use_imu:
                    # iESKF measurement update — replaces direct state assignment
                    self.ieskf.update(T, meas_noise_pos, meas_noise_rot)
                else:
                    # Fallback when IMU disabled: direct state update (original)
                    dR       = T[:3, :3]
                    dp_body  = T[:3, 3]
                    dt_scan  = max(1e-6,
                                   current_stamp_sec - self.last_scan_stamp_sec
                                   if self.last_scan_stamp_sec else 1.0)
                    p_prev   = self.ieskf.p.copy()
                    self.ieskf.p = self.ieskf.p + self.ieskf.R @ dp_body
                    self.ieskf.R = self.ieskf.R @ dR
                    U, _, Vt = np.linalg.svd(self.ieskf.R)
                    self.ieskf.R = U @ Vt
                    self.ieskf.v = (self.ieskf.p - p_prev) / dt_scan

                if self.force_z_zero:
                    self.ieskf.p[2] = 0.0

                # Feed non-rep processor (unchanged)
                yaw      = float(np.arctan2(self.ieskf.R[1, 0], self.ieskf.R[0, 0]))
                obs_pose = np.array([self.ieskf.p[0], self.ieskf.p[1],
                                     self.ieskf.p[2], yaw])
                self.processor.update_with_observation(obs_pose, feat, reg_conf, pred_pose)

            else:
                # First scan — initialise at origin
                self.processor.update_with_observation(
                    np.array([0.0, 0.0, 0.0, 0.0]), feat, 0.3, pred_pose)

            # Snapshot state for next interval's relative-transform query
            self.ieskf.save_scan_state()

            # ------ Publish odometry --------------------------------------
            self._publish_odom_and_tf(stamp, self.ieskf.p, self.ieskf.R, self.ieskf.v)

            # ------ Map accumulation --------------------------------------
            Tmap = np.eye(4, dtype=float)
            if self.use_imu:
                Tmap[:3, :3] = self.ieskf.R
            else:
                yaw_map = float(np.arctan2(self.ieskf.R[1, 0], self.ieskf.R[0, 0]))
                Tmap[0, 0] =  np.cos(yaw_map); Tmap[0, 1] = -np.sin(yaw_map)
                Tmap[1, 0] =  np.sin(yaw_map); Tmap[1, 1] =  np.cos(yaw_map)
            Tmap[:3, 3]  = self.ieskf.p
            cur_in_map   = o3d.geometry.PointCloud(cloud)
            cur_in_map.transform(Tmap)
            self.map_cloud += cur_in_map
            if self.map_voxel > 0 and len(self.map_cloud.points) > 2_000_000:
                self.map_cloud = self.map_cloud.voxel_down_sample(float(self.map_voxel))
            self._publish_map_cloud(stamp)

            if self.viewer is not None:
                self.viewer.update(latest_cloud=cloud, map_cloud=self.map_cloud)

        except Exception as e:
            import traceback
            self.get_logger().error(
                f"Scan {self.scan_counter} error: {e}\n{traceback.format_exc()}")

        self.prev_cloud        = cloud
        self.last_scan_stamp_sec = current_stamp_sec
        self.scan_counter      += 1

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
