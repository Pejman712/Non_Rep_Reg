#!/usr/bin/env python3.10
"""
ros_lio_v2.py  —  LIO node with 18-DOF IESKF (gravity in state, information-form update).

Upgrades over ros_lio.py  (ros_lio.py is UNCHANGED):
  1. 18-DOF error state: [δp, δv, δθ, δbg, δba, δg]
       Gravity g is estimated online and constrained to ||g|| = gravity_mag.
  2. Information-form iterated update (Super-LIO style):
         A  = P_k⁻¹ + Hᵀ V⁻¹ H
         Q_k = A⁻¹
         dx  = Q_k · b + (Q_k·A − I) · dx_prior
       Numerically stable for degenerate non-rep scans where some DOFs are
       unobservable.
  3. Right-Jacobian correction in the propagation Jacobian F:
         F[6:9, 9:12] = −J_r(ω·dt)·dt   (was −I·dt)
       More accurate for fast-rotating platforms.
  4. Midpoint IMU integration (Super-LIO Predict):
         ω, a  =  0.5·(sample_k + sample_{k−1})
  5. IMU accelerometer auto-scale (Super-LIO kf_init):
         scale = gravity_mag / ‖mean static accel‖   applied as  a·scale − ba
  6. Leveled, yaw-removed initial orientation (Super-LIO kf_init):
       At gravity init the whole state is rotated into a gravity-aligned world
       frame (yaw removed) and g is pinned to the canonical [0,0,gravity_mag].
       The voxel map and prev_cloud are restarted in the new frame (Super-LIO
       runs kf_init strictly before map_init for the same reason).
  7. Full per-point undistortion (Super-LIO Propagation_Undistort):
       rotation AND translation compensation from propagated filter states
         x' = R_endᵀ·(R_i·x + p_i − p_end),   p_i = p + v·τ + ½·a·τ²
       (v1 deskew is rotation-only).  Degrades gracefully to rotation-only when
       accel integration is off.

Registration is deliberately UNCHANGED — non-rep GICP scan-to-submap remains
the observation source (Super-LIO's point-to-plane residuals are NOT ported).

Everything else is unchanged from ros_lio.py:
  - NonRepetitiveLiDARProcessor (imported)
  - VoxelHashMap (imported)
  - GICP pipeline, scan-to-submap, ZUPT, soft-z (inherited)
  - ROS publishers / TF / benchmark tooling (inherited)

Run:
  ros2 launch regnonrep lio_v2.launch.py
"""

import sys
import os

# Both scripts land in lib/regnonrep/ after colcon build, so ros_lio is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import collections

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2
from nav_msgs.msg import Odometry

# Import all shared building blocks from the original node (unchanged).
from ros_lio import (
    ScanState,
    NonRepetitiveLiDARProcessor,
    VoxelHashMap,
    so3_exp,
    so3_log,
    _skew,
    rot_to_quat,
    yaw_to_quat,
    ros_time_to_sec,
    estimate_registration_confidence,
    pointcloud2_to_xyz_i_stamps,
    xyzi_to_open3d_cloud,
    open3d_cloud_to_pointcloud2_xyzi,
    apply_gicp_open3d,
    LioNode,
)


# =============================================================================
# IESKF18  —  18-DOF information-form iESKF with gravity in state
# =============================================================================
class IESKF18:
    """
    18-DOF Iterated Error-State Kalman Filter on SO(3).

    Nominal state:   x  = [p(3), v(3), R∈SO(3), bg(3), ba(3), g(3)]
    Error state:    δx  ∈ R^18  = [δp, δv, δθ, δbg, δba, δg]

    Key differences from IESKF (15-DOF in ros_lio.py):
      • g is part of the state and estimated online (constrained ||g||=gravity_mag).
      • Update uses the information form: A = P⁻¹ + HᵀV⁻¹H, Q = A⁻¹.
      • Propagation Jacobian uses Right Jacobian J_r instead of -I·dt for the
        gyro-bias block and the gyro-noise input.
      • g_world is a property alias for g so all LioNode callback code that
        reads self.ieskf.g_world automatically gets the current gravity estimate.
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
        init_g_cov: float = 0.01,
        max_iters: int = 3,
        conv_threshold: float = 1e-6,
    ):
        self.gravity_mag = gravity_mag
        self.gravity_init_n = gravity_init_n
        self.gravity_initialized = False
        self.gravity_z_down = False
        self._gravity_samples: List[np.ndarray] = []
        self._gyro_samples:    List[np.ndarray] = []

        # ---- Super-LIO style init/integration options --------------------
        # level_at_init: rotate state into a gravity-aligned, yaw-removed world
        # frame at static-init completion (Super-LIO kf_init).
        self.level_at_init = False
        # auto_scale: estimate accel scale = gravity_mag/||mean static accel||
        # (Super-LIO imu_scale); applied multiplicatively before bias.
        self.auto_scale  = False
        self.accel_scale = 1.0
        # Set to the leveling rotation W at init so the node can restart the map.
        self.last_level_W: Optional[np.ndarray] = None
        # Midpoint-integration memory (Super-LIO: 0.5*(imu + last_imu))
        self._prev_omega: Optional[np.ndarray] = None
        self._prev_accel: Optional[np.ndarray] = None
        # Last world-frame acceleration (for full undistortion state snapshots)
        self.last_a_world = np.zeros(3)
        # Gyro-only position random walk [m/s/√Hz]: unmodeled motion prior so
        # GICP retains authority over translation when accel is off (walking
        # speed scale).  Unused when accel integration is active.
        self.sigma_p_rw = 1.0

        # Nominal state
        self.p  = np.zeros(3)
        self.v  = np.zeros(3)
        self.R  = np.eye(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self._g = np.array([0.0, 0.0, gravity_mag], dtype=float)  # gravity state

        # Covariance (18×18)
        d = np.concatenate([
            [init_p_cov]   * 3,
            [init_v_cov]   * 3,
            [init_rot_cov] * 3,
            [init_bg_cov]  * 3,
            [init_ba_cov]  * 3,
            [init_g_cov]   * 3,
        ])
        self.P = np.diag(d.astype(float))

        # Process noise densities
        self.sigma_gyro  = sigma_gyro
        self.sigma_accel = sigma_accel
        self.sigma_bg    = sigma_bg
        self.sigma_ba    = sigma_ba

        # Iteration control
        self.max_iters      = max_iters
        self.conv_threshold = conv_threshold

        # Scan-state snapshots (used by get_predicted_relative_transform)
        self._p_last = np.zeros(3)
        self._R_last = np.eye(3)

    # ------------------------------------------------------------------
    # g / g_world property — keeps backward compat with LioNode callbacks
    # ------------------------------------------------------------------
    @property
    def g(self) -> np.ndarray:
        return self._g

    @g.setter
    def g(self, val: np.ndarray):
        self._g = np.asarray(val, dtype=float)

    @property
    def g_world(self) -> np.ndarray:
        """Alias: LioNode callbacks read self.ieskf.g_world; return live estimate."""
        return self._g

    @g_world.setter
    def g_world(self, val: np.ndarray):
        self._g = np.asarray(val, dtype=float)

    # ------------------------------------------------------------------
    # Static initialisation (gravity + gyro bias)
    # ------------------------------------------------------------------
    def collect_static_sample(self, omega: np.ndarray, accel: np.ndarray,
                               R_world: np.ndarray) -> bool:
        if self.gravity_initialized:
            return False
        a_c = accel - self.ba
        if abs(float(np.linalg.norm(a_c)) - self.gravity_mag) < 1.5:
            self._gravity_samples.append(R_world @ a_c)
            self._gyro_samples.append(omega.copy())
            if len(self._gravity_samples) >= self.gravity_init_n:
                g_est = np.mean(self._gravity_samples, axis=0)
                self.bg = np.mean(self._gyro_samples, axis=0)
                if self.gravity_z_down or g_est[2] < 0.0:
                    g_est = -g_est
                g_norm = float(np.linalg.norm(g_est))

                # Super-LIO imu_scale: measured static accel should equal g
                if self.auto_scale and g_norm > 1e-3:
                    self.accel_scale = self.gravity_mag / g_norm

                if self.level_at_init and g_norm > 1e-3:
                    # Super-LIO kf_init: rotate the whole state into a
                    # gravity-aligned world frame with yaw removed, then pin
                    # g to the canonical vector.
                    W = self._level_rotation(g_est)
                    self.p  = W @ self.p
                    self.v  = W @ self.v
                    self.R  = W @ self.R
                    self._p_last = W @ self._p_last
                    self._R_last = W @ self._R_last
                    self._g = np.array([0.0, 0.0, self.gravity_mag])
                    self.last_level_W = W
                else:
                    self._g = g_est * self.accel_scale
                self.gravity_initialized = True
                return True
        return False

    # ------------------------------------------------------------------
    # Leveling rotation (Super-LIO kf_init): W·g_est ∝ +z, yaw removed
    # ------------------------------------------------------------------
    @staticmethod
    def _level_rotation(g_est: np.ndarray) -> np.ndarray:
        a = g_est / float(np.linalg.norm(g_est))
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(a, z)
        s = float(np.linalg.norm(v))
        c = float(np.dot(a, z))
        if s < 1e-9:
            W0 = np.eye(3) if c > 0.0 else so3_exp(np.array([np.pi, 0.0, 0.0]))
        else:
            W0 = so3_exp((v / s) * np.arctan2(s, c))
        # Remove yaw (Super-LIO: yaw of the x-axis of the alignment rotation)
        yaw = float(np.arctan2(W0[1, 0], W0[0, 0]))
        return so3_exp(np.array([0.0, 0.0, -yaw])) @ W0

    # ------------------------------------------------------------------
    # Right Jacobian of SO(3): J_r(φ) = I - a·K + b·K²
    # ------------------------------------------------------------------
    @staticmethod
    def _right_jacobian(phi: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(phi))
        if angle < 1e-8:
            return np.eye(3)
        K = _skew(phi / angle)
        a = (1.0 - np.cos(angle)) / angle
        b = 1.0 - np.sin(angle) / angle
        return np.eye(3) - a * K + b * (K @ K)

    # ------------------------------------------------------------------
    # IMU propagation  (18-DOF, right-Jacobian, gravity from state)
    # ------------------------------------------------------------------
    def propagate(self, omega: np.ndarray, accel: np.ndarray, dt: float,
                  use_accel: bool = True):
        if dt <= 0.0 or dt > 1.0:
            return

        # Never integrate acceleration until gravity has been initialised;
        # before that self._g is a placeholder and would cause position explosion.
        effective_use_accel = use_accel and self.gravity_initialized

        # Super-LIO imu_scale: multiplicative accel correction before bias
        accel_s = accel * self.accel_scale

        # Super-LIO midpoint integration: 0.5·(sample_k + sample_{k−1})
        if self._prev_omega is not None:
            omega_m = 0.5 * (omega + self._prev_omega)
            accel_m = 0.5 * (accel_s + self._prev_accel)
        else:
            omega_m = omega
            accel_m = accel_s
        self._prev_omega = omega.copy()
        self._prev_accel = accel_s.copy()

        omega_c = omega_m - self.bg
        Jr      = self._right_jacobian(omega_c * dt)   # 3×3 right Jacobian
        Jr_dt   = Jr * dt

        R_new = self.R @ so3_exp(omega_c * dt)

        if effective_use_accel:
            accel_c = accel_m - self.ba
            a_world = self.R @ accel_c - self._g          # gravity from state
            p_new   = self.p + self.v * dt + 0.5 * a_world * dt * dt
            v_new   = self.v + a_world * dt
        else:
            accel_c = np.zeros(3)
            a_world = np.zeros(3)
            p_new   = self.p
            v_new   = np.zeros(3)
        self.last_a_world = a_world   # snapshot for full undistortion

        # ----- Error-state Jacobian F (18×18) -----
        F = np.eye(18)
        F[0:3,   3:6]  = np.eye(3) * dt                    # ∂δp/∂δv
        if effective_use_accel:
            F[3:6,  6:9]  = -(self.R @ _skew(accel_c)) * dt  # ∂δv/∂δθ
            F[3:6, 12:15] = -self.R * dt                    # ∂δv/∂δba
            F[3:6, 15:18] = np.eye(3) * dt                  # ∂δv/∂δg  [NEW]
        F[6:9,   6:9]  = so3_exp(-omega_c * dt)             # ∂δθ/∂δθ
        F[6:9,   9:12] = -Jr_dt                             # ∂δθ/∂δbg  [RIGHT JACOBIAN]

        # ----- Process-noise input Jacobian G (18×12) -----
        G = np.zeros((18, 12))
        G[6:9,   0:3]  = -Jr                               # δθ ← gyro noise  [RIGHT JACOBIAN]
        if effective_use_accel:
            G[3:6,  3:6]  = self.R                         # δv ← accel noise
            G[12:15, 9:12] = np.eye(3)                     # δba drift
        G[9:12,  6:9]  = np.eye(3)                         # δbg drift

        # ----- Discrete process noise Q (12×12) -----
        Q = np.zeros((12, 12))
        Q[0:3, 0:3]   = np.eye(3) * (self.sigma_gyro  ** 2 / dt)
        Q[6:9, 6:9]   = np.eye(3) * (self.sigma_bg    ** 2 * dt)
        if effective_use_accel:
            Q[3:6,  3:6]  = np.eye(3) * (self.sigma_accel ** 2 / dt)
            Q[9:12, 9:12] = np.eye(3) * (self.sigma_ba    ** 2 * dt)

        self.P = F @ self.P @ F.T + G @ Q @ G.T

        # Gyro-only: position is frozen during propagation, so without process
        # noise P_pp would stay at its (tiny) init value and the information-
        # form update would give GICP position measurements ~zero gain — the
        # filter never moves.  Model the unmodeled body motion as a position
        # random walk so GICP has full authority over translation.
        if not effective_use_accel:
            self.P[0:3, 0:3] += np.eye(3) * (self.sigma_p_rw ** 2 * dt)

        self.p = p_new
        self.v = v_new
        U, _, Vt = np.linalg.svd(R_new)
        self.R   = U @ Vt

    # ------------------------------------------------------------------
    # Scan-state snapshots
    # ------------------------------------------------------------------
    def save_scan_state(self):
        self._p_last = self.p.copy()
        self._R_last = self.R.copy()

    def save_full_snapshot(self) -> dict:
        return {
            'p':  self.p.copy(),  'v': self.v.copy(),   'R': self.R.copy(),
            'bg': self.bg.copy(), 'ba': self.ba.copy(),  'g': self._g.copy(),
            'P':  self.P.copy(),
            '_p_last': self._p_last.copy(), '_R_last': self._R_last.copy(),
            '_prev_omega': None if self._prev_omega is None else self._prev_omega.copy(),
            '_prev_accel': None if self._prev_accel is None else self._prev_accel.copy(),
        }

    def restore_full_snapshot(self, snap: dict) -> None:
        self.p  = snap['p'].copy();  self.v  = snap['v'].copy()
        self.R  = snap['R'].copy();  self.bg = snap['bg'].copy()
        self.ba = snap['ba'].copy(); self._g = snap['g'].copy()
        self.P  = snap['P'].copy()
        self._p_last = snap['_p_last'].copy()
        self._R_last = snap['_R_last'].copy()
        po = snap.get('_prev_omega');  pa = snap.get('_prev_accel')
        self._prev_omega = None if po is None else po.copy()
        self._prev_accel = None if pa is None else pa.copy()

    def get_predicted_relative_transform(self) -> np.ndarray:
        dR = self._R_last.T @ self.R
        dp_body = self._R_last.T @ (self.p - self._p_last)
        T = np.eye(4, dtype=float)
        T[:3, :3] = dR
        T[:3, 3]  = dp_body
        return T

    # ------------------------------------------------------------------
    # Information-form iESKF update  (Super-LIO style)
    # ------------------------------------------------------------------
    def update(
        self,
        T_gicp: np.ndarray,
        R_n: np.ndarray,
        chi2_threshold: float = 22.46,
    ) -> Tuple[bool, float]:
        """
        Information-form iterated update using the GICP relative transform.

        Measurement model (H is 6×18, sparse):
            H[0:3, 0:3] = I   (position)
            H[3:6, 6:9] = I   (rotation)

        Information form:
            A    = P_k⁻¹ + Hᵀ V⁻¹ H
            Q_k  = A⁻¹
            dx   = Q_k · b + (Q_k · A − I) · dx_prior

        Gravity columns in H are zero, so gravity is not directly observed by
        GICP but is indirectly observable through its effect on velocity via
        the accelerometer during propagation.
        """
        p_meas = self._p_last + self._R_last @ T_gicp[:3, 3]
        R_meas = self._R_last @ T_gicp[:3, :3]

        # --- Chi-squared gating at nominal state -------------------------
        z_p0 = p_meas - self.p
        z_R0 = so3_log(self.R.T @ R_meas)
        z0   = np.concatenate([z_p0, z_R0])

        # Marginal S from the p and θ blocks of P
        S = np.block([
            [self.P[0:3, 0:3] + R_n[0:3, 0:3],  self.P[0:3, 6:9] + R_n[0:3, 3:6]],
            [self.P[6:9, 0:3] + R_n[3:6, 0:3],  self.P[6:9, 6:9] + R_n[3:6, 3:6]],
        ])
        try:
            chi2 = float(z0 @ np.linalg.inv(S) @ z0)
        except np.linalg.LinAlgError:
            chi2 = 0.0
        if chi2 > chi2_threshold:
            return False, chi2

        # --- Pre-compute V⁻¹ and the sparse 18×18 Hᵀ V⁻¹ H ---------------
        try:
            V_inv = np.linalg.inv(R_n)
        except np.linalg.LinAlgError:
            return False, chi2

        HTVH = np.zeros((18, 18))
        HTVH[0:3, 0:3] = V_inv[0:3, 0:3]
        HTVH[0:3, 6:9] = V_inv[0:3, 3:6]
        HTVH[6:9, 0:3] = V_inv[3:6, 0:3]
        HTVH[6:9, 6:9] = V_inv[3:6, 3:6]

        # --- Save prediction state ----------------------------------------
        p_pred  = self.p.copy();   v_pred  = self.v.copy()
        R_pred  = self.R.copy();   bg_pred = self.bg.copy()
        ba_pred = self.ba.copy();  g_pred  = self._g.copy()
        P_pred  = self.P.copy()

        Q_k = np.eye(18)
        dx  = np.zeros(18)

        for it in range(self.max_iters):
            # Innovation at current iterate
            z_p = p_meas - self.p
            z_R = so3_log(self.R.T @ R_meas)
            z   = np.concatenate([z_p, z_R])

            # Hᵀ V⁻¹ r  (sparse, only p and θ rows non-zero)
            HTVr = np.zeros(18)
            HTVr[0:3] = V_inv[0:3, :] @ z
            HTVr[6:9] = V_inv[3:6, :] @ z

            # Prior deviation from prediction
            dx_prior = np.zeros(18)
            dx_prior[0:3]   = self.p  - p_pred
            dx_prior[3:6]   = self.v  - v_pred
            dx_prior[6:9]   = so3_log(R_pred.T @ self.R)
            dx_prior[9:12]  = self.bg - bg_pred
            dx_prior[12:15] = self.ba - ba_pred
            dx_prior[15:18] = self._g  - g_pred

            # Left-Jacobian correction for the SO(3) prior term
            J_p = np.eye(3) - 0.5 * _skew(dx_prior[6:9])
            G_p = np.eye(18);  G_p[6:9, 6:9] = J_p
            P_k = G_p @ P_pred @ G_p.T
            dx_prior_c = G_p @ dx_prior

            # Information-form solve
            try:
                A   = np.linalg.inv(P_k) + HTVH
                Q_k = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                return False, chi2

            K_x = Q_k @ HTVH
            dx  = Q_k @ HTVr + (K_x - np.eye(18)) @ dx_prior_c

            # Apply correction to nominal state
            self.p   = self.p  + dx[0:3]
            self.v   = self.v  + dx[3:6]
            self.R   = self.R  @ so3_exp(dx[6:9])
            U, _, Vt = np.linalg.svd(self.R);  self.R = U @ Vt
            self.bg  = self.bg + dx[9:12]
            self.ba  = self.ba + dx[12:15]
            self._g  = self._g + dx[15:18]

            # Constrain gravity norm
            g_n = float(np.linalg.norm(self._g))
            if g_n > 1e-3:
                self._g = self.gravity_mag * self._g / g_n

            # Super-LIO quit check: ∞-norm, never on the first iteration
            if it > 0 and float(np.max(np.abs(dx))) < self.conv_threshold:
                break

        # --- Covariance update (information form + SO(3) reset) ----------
        self.P = Q_k

        J_r = np.eye(3) - 0.5 * _skew(dx[6:9])
        G_r = np.eye(18);  G_r[6:9, 6:9] = J_r
        self.P = G_r @ self.P @ G_r.T
        self.P = 0.5 * (self.P + self.P.T)   # enforce symmetry

        return True, chi2

    # ------------------------------------------------------------------
    # ZUPT — extended to 18-DOF
    # ------------------------------------------------------------------
    def zupt_update(self, sigma_v: float = 0.01) -> None:
        H = np.zeros((3, 18));  H[:, 3:6] = np.eye(3)
        R_n = np.eye(3) * (sigma_v ** 2)
        S   = H @ self.P @ H.T + R_n
        K   = self.P @ H.T @ np.linalg.inv(S)
        dx  = K @ (-self.v)
        self.p  += dx[0:3];   self.v  += dx[3:6]
        self.R   = self.R @ so3_exp(dx[6:9])
        self.bg += dx[9:12];  self.ba += dx[12:15]
        self._g += dx[15:18]
        g_n = float(np.linalg.norm(self._g))
        if g_n > 1e-3:
            self._g = self.gravity_mag * self._g / g_n
        I_KH = np.eye(18) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_n @ K.T

    # ------------------------------------------------------------------
    # Soft floor constraint — extended to 18-DOF
    # ------------------------------------------------------------------
    def soft_z_update(self, sigma_z: float = 0.05) -> None:
        H = np.zeros((1, 18));  H[0, 2] = 1.0
        R_n = np.array([[sigma_z ** 2]])
        S   = H @ self.P @ H.T + R_n
        K   = self.P @ H.T @ np.linalg.inv(S)
        dx  = K @ np.array([0.0 - self.p[2]])
        self.p  += dx[0:3];   self.v  += dx[3:6]
        self.R   = self.R @ so3_exp(dx[6:9])
        self.bg += dx[9:12];  self.ba += dx[12:15]
        self._g += dx[15:18]
        g_n = float(np.linalg.norm(self._g))
        if g_n > 1e-3:
            self._g = self.gravity_mag * self._g / g_n
        I_KH = np.eye(18) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_n @ K.T

    # ------------------------------------------------------------------
    # Properties expected by LioNode callbacks
    # ------------------------------------------------------------------
    @property
    def is_gyro_ready(self) -> bool:
        return True

    @property
    def is_accel_ready(self) -> bool:
        return self.gravity_initialized

    @property
    def pose_covariance_6x6(self) -> np.ndarray:
        cov = np.zeros((6, 6))
        cov[0:3, 0:3] = self.P[0:3, 0:3]
        cov[3:6, 3:6] = self.P[6:9, 6:9]
        cov[0:3, 3:6] = self.P[0:3, 6:9]
        cov[3:6, 0:3] = self.P[6:9, 0:3]
        return cov


# =============================================================================
# LioNodeV2  —  LioNode with IESKF18 swapped in
# =============================================================================
class LioNodeV2(LioNode):
    """
    Identical to LioNode except the 15-DOF IESKF is replaced by IESKF18.

    The constructor calls LioNode.__init__() which initialises the 15-DOF IESKF
    and all ROS subscriptions/publishers, then immediately replaces self.ieskf
    with IESKF18.  All callback methods (cb_imu, cb_cloud, …) are inherited
    unchanged — they access self.ieskf which now points to the IESKF18 instance.

    The ROS node name is kept as "lio_node" internally but overridden to
    "lio_node_v2" by the launch file (--ros-args -r __node:=lio_node_v2).
    """

    def __init__(self):
        # Full LioNode setup: parameters, subscriptions, publishers, 15-DOF IESKF
        super().__init__()

        # Declare extra parameters not present in LioNode
        self.declare_parameter("ieskf_init_g_cov", 0.01)
        init_g_cov = float(self.get_parameter("ieskf_init_g_cov").value)
        # Super-LIO ports (all default on; only active when imu_use_accel=true
        # except undistort_translation which degrades to rotation-only anyway)
        self.declare_parameter("level_init", True)
        self.declare_parameter("imu_auto_scale", True)
        self.declare_parameter("undistort_translation", True)
        level_init      = bool(self.get_parameter("level_init").value)
        imu_auto_scale  = bool(self.get_parameter("imu_auto_scale").value)
        self.undistort_translation = bool(
            self.get_parameter("undistort_translation").value)

        # Propagated dynamic-state snapshots (Super-LIO propagate_states_):
        # (t_eff, R, p, v, a_world) appended after every IMU propagation,
        # consumed by the full undistortion in _deskew_cloud.  ~3 s at 200 Hz.
        self._prop_states: collections.deque = collections.deque(maxlen=600)

        # ---- Deep IMU queue ------------------------------------------------
        # rclpy executes callbacks single-threaded: while cb_cloud is inside a
        # long GICP call (hundreds of ms at corners), IMU messages queue up.
        # LioNode's depth of 200 (1 s at 200 Hz) silently DROPS samples during
        # long stalls — the rotation they carried is lost forever, which makes
        # the propagated yaw under-rotate and poisons the map at every corner.
        # 4000 ≈ 20 s of buffer; messages are replayed with their original
        # stamps so propagation stays correct, just delayed.
        if self.use_imu:
            self.destroy_subscription(self.imu_sub)
            self.imu_sub = self.create_subscription(
                Imu, self.imu_topic, self.cb_imu, 4000)

        # Preserve any bias priors that LioNode.__init__ may have set from YAML
        old_bg = self.ieskf.bg.copy()
        old_ba = self.ieskf.ba.copy()

        # ---- Replace 15-DOF IESKF with 18-DOF IESKF18 -------------------
        self.ieskf = IESKF18(
            gravity_mag     = self.imu_gravity_mag,
            gravity_init_n  = self.imu_gravity_init_n,
            sigma_gyro      = self.ieskf_sigma_gyro,
            sigma_accel     = self.ieskf_sigma_accel,
            sigma_bg        = self.ieskf_sigma_bg,
            sigma_ba        = self.ieskf_sigma_ba,
            init_p_cov      = self.ieskf_init_p_cov,
            init_bg_cov     = self.ieskf_init_bg_cov,
            init_ba_cov     = self.ieskf_init_ba_cov,
            init_g_cov      = init_g_cov,
            max_iters       = self.ieskf_max_iters,
        )
        self.ieskf.gravity_z_down = self.imu_gravity_z_down
        self.ieskf.bg = old_bg
        self.ieskf.ba = old_ba
        self.ieskf.level_at_init = level_init
        self.ieskf.auto_scale    = imu_auto_scale

        self.get_logger().info("=== LIO Node V2 (18-DOF IESKF18, Super-LIO ports) ===")
        self.get_logger().info("  Gravity in state:           online estimation, norm-constrained")
        self.get_logger().info("  Update form:                information-form (A = P⁻¹ + HᵀV⁻¹H)")
        self.get_logger().info("  Propagation:                right-Jacobian J_r, midpoint integration")
        self.get_logger().info(f"  level_init:                 {level_init}  (gravity-aligned world frame)")
        self.get_logger().info(f"  imu_auto_scale:             {imu_auto_scale}")
        self.get_logger().info(f"  undistort_translation:      {self.undistort_translation}")
        self.get_logger().info(f"  init_g_cov:                 {init_g_cov}")
        self.get_logger().info(f"  ieskf_sigma_gyro/accel:     {self.ieskf_sigma_gyro} / {self.ieskf_sigma_accel}")

    # -------------------------------------------------------------------------
    # Scan callback gate (Super-LIO stateWaitKFInit): no lidar processing
    # until gravity/bias init completes.  Pre-init scans were processed with
    # an uninitialised filter — they produced the noisy pose cluster at the
    # trajectory start and seeded GICP with junk.  Gravity init needs only
    # IMU samples, so dropping these scans costs ~1 s of (stationary) data.
    # -------------------------------------------------------------------------
    def cb_cloud(self, msg: PointCloud2):
        if (self.use_imu and self.imu_use_accel
                and not self.ieskf.gravity_initialized):
            return
        super().cb_cloud(msg)

    # -------------------------------------------------------------------------
    # IMU callback override — record propagated states (Super-LIO
    # propagate_states_) and restart the map when the world frame is leveled.
    # -------------------------------------------------------------------------
    def cb_imu(self, msg: Imu):
        was_init = self.ieskf.gravity_initialized
        super().cb_imu(msg)

        # Snapshot the propagated state for full undistortion
        if self._last_prop_stamp is not None:
            self._prop_states.append((
                self._last_prop_stamp,
                self.ieskf.R.copy(),
                self.ieskf.p.copy(),
                self.ieskf.v.copy(),
                self.ieskf.last_a_world.copy(),
            ))

        # Gravity init just completed with leveling: the world frame rotated,
        # so everything built in the old frame must be discarded (Super-LIO
        # runs kf_init strictly before map_init — this is the equivalent).
        if (not was_init and self.ieskf.gravity_initialized
                and self.ieskf.last_level_W is not None):
            self.voxel_map._cells.clear()
            self.voxel_map._colors.clear()
            self.voxel_map._order.clear()
            self.prev_cloud = None
            self._prop_states.clear()
            self.get_logger().info(
                f"[level_init] World frame leveled (gravity-aligned, yaw removed); "
                f"accel_scale={self.ieskf.accel_scale:.4f} — map restarted"
            )

    # -------------------------------------------------------------------------
    # Full undistortion override (Super-LIO Propagation_Undistort):
    # rotation AND translation compensation from propagated filter states.
    #   x' = R_endᵀ · (R_i · x + p_i − p_end),   p_i = p + v·τ + ½·a·τ²
    # Rotation is segment-constant at IMU rate (≈5 ms; matches v1 deskew).
    # Falls back to the inherited rotation-only deskew when states are missing.
    # -------------------------------------------------------------------------
    def _deskew_cloud(self, xyz: np.ndarray,
                      stamps_sec: Optional[np.ndarray],
                      t_scan_sec: float) -> np.ndarray:
        if not self.undistort_translation or len(self._prop_states) < 2:
            return super()._deskew_cloud(xyz, stamps_sec, t_scan_sec)
        if stamps_sec is None or xyz.shape[0] == 0:
            return xyz
        t_min = float(stamps_sec.min())
        if t_scan_sec - t_min < 1e-4:
            return xyz   # all points at same time — nothing to do

        states = [s for s in self._prop_states if s[0] >= t_min - 0.005]
        if len(states) < 2:
            return super()._deskew_cloud(xyz, stamps_sec, t_scan_sec)

        # Synthetic end state at t_scan (zero-order-hold extrapolation,
        # mirrors v1 appending the last IMU sample at scan time)
        t_last, R_last, p_last, v_last, a_last = states[-1]
        if t_last < t_scan_sec and self._last_omega is not None:
            dtau  = t_scan_sec - t_last
            R_end = R_last @ so3_exp((self._last_omega - self.ieskf.bg) * dtau)
            p_end = p_last + v_last * dtau + 0.5 * a_last * dtau * dtau
            v_end = v_last + a_last * dtau
            states.append((t_scan_sec, R_end, p_end, v_end, a_last))

        _, R_end, p_end, _, _ = states[-1]
        R_end_T = R_end.T

        xyz_out = xyz.copy()
        n_seg   = len(states) - 1
        for i in range(n_seg):
            t_lo, R_lo, p_lo, v_lo, a_lo = states[i]
            t_hi = states[i + 1][0]
            # Clamp out-of-range points to the first/last segment
            if i == 0:
                mask = stamps_sec < t_hi
            elif i == n_seg - 1:
                mask = stamps_sec >= t_lo
            else:
                mask = (stamps_sec >= t_lo) & (stamps_sec < t_hi)
            if not mask.any():
                continue
            tau = np.clip(stamps_sec[mask] - t_lo, 0.0, None)[:, None]
            # Const-accel position interp at each point's capture time
            p_i = p_lo + v_lo * tau + 0.5 * a_lo * tau * tau          # (N,3)
            x_w = (R_lo @ xyz[mask].T).T + p_i                         # world
            xyz_out[mask] = (R_end_T @ (x_w - p_end).T).T              # scan-end frame
        return xyz_out


# =============================================================================
# Entry point
# =============================================================================
def main():
    rclpy.init()
    node = LioNodeV2()
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
