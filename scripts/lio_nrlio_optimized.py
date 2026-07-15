#!/usr/bin/env python3.10
"""
lio_nrlio_optimized.py — variant "nrlio_optimized": nrlio with the best
parameters from the Tier convergence campaign (benchmark_results/_campaign2)
baked in, plus two robustness guards that fix the indoor3 divergences.

Campaign-best parameters (mean ATE-SE3 2.75 -> 1.00 m over the 6 Tier sequences,
0 divergences in 18 confirmation runs).  A first campaign chose coarse/robust
settings to STOP divergence; once the guards below made divergence impossible, a
second campaign found that FINER + TIGHTER settings lift the accuracy ceiling:
  * opt_accum_voxel      = 0.08  finer accumulation voxel (was 0.16) — precision
  * opt_voxel_min        = 0.03  finer registration voxel floor (was 0.05)
  * opt_chi2             = 100   tighter acceptance (was 200; 200 admitted conf~0.1 GICP)
  * opt_knee_ms          = 100   1-frame accumulation (less accumulation won)
  * opt_degen_planarity  = 0.01  flag FEWER scans degenerate (was 0.03) — update,
  * opt_degen_extent     = 0.5   don't coast (was 1.0)
  * kf_max_iterations    = 4     (base default; 6/8 over-fit degenerate scans → worse)
  * ZUPT                 kept on
  * degeneracy routing   DISABLED on the 360° Mid-360 (it misfires there)

Robustness guards (added after inspecting the indoor3 logs, where the estimate
diverged and the run truncated at ~408 poses):
  (1) HOLD-ON-SKIP.  When registration is unavailable on a degenerate scan
      (method 'skip_degen'), the base dead-reckoned on the IMU — Horizon indoor3
      jumped +45 m in a few scans.  Instead we HOLD the last translation, keep the
      gyro-propagated rotation, and zero the velocity ("when you can't see, don't
      move").  Rotation from the gyro is trustworthy; translation from double-
      integrated accel is not.
  (2) MOTION CLAMP.  Any accepted update that moves the pose more than
      `opt_max_step` metres in one scan (implausible at indoor speeds) is clamped
      back to that step and the velocity zeroed — a hard backstop against
      explosions from any path.  The map is frozen on a clamped/held scan.

Everything else is inherited from nrlio.  All values remain overridable via the
opt_* / base params.

    ros2 run regnonrep lio_nrlio_optimized.py --ros-args -p opt_max_step:=0.6
"""
from collections import deque

import numpy as np

from lio_base import run_node
from lio_nrlio import SuperLioNRLIO


class SuperLioNRLIOOpt(SuperLioNRLIO):
    NODE_NAME = "super_lio_nrlio_opt"
    VARIANT_DESC = ("nrlio + campaign-optimized params + robustness guards "
                    "(hold-on-skip, motion clamp; degeneracy off on 360° Mid-360)")

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        # --- campaign-best values (14 h Tier convergence campaign, _campaign2) ---
        # mean ATE-SE3 2.75 -> 1.00 m over the 6 Tier sequences, 0 divergences in
        # 18 confirmation runs.  Finer voxels + tighter acceptance + 1-frame accum
        # + less-aggressive degeneracy flagging lifted the precision ceiling.
        self.nr_accum_voxel = float(gp("opt_accum_voxel", 0.08).value)   # was 0.16
        self.chi2_threshold = float(gp("opt_chi2", 100.0).value)         # was 200
        self.nr_knee_ms = float(gp("opt_knee_ms", 100.0).value)          # 1-frame (was 200)
        self.nr_accum_max = max(1, int(round(self.nr_knee_ms / max(self.nr_scan_ms, 1.0))))
        self._acc = deque(maxlen=max(0, self.nr_accum_max - 1))
        # finer registration voxel floor + less-aggressive degeneracy detection
        # (flag fewer scans → update on more scans → converge instead of coast)
        self.gen_d_min = float(gp("opt_voxel_min", 0.03).value)          # was 0.05
        self.degen_planarity_ratio = float(gp("opt_degen_planarity", 0.01).value)  # was 0.03
        self.degen_min_extent = float(gp("opt_degen_extent", 0.5).value)           # was 1.0
        # degeneracy routing misfires on the 360° Mid-360 -> disable it there only
        self.opt_degen_off_360 = bool(gp("opt_degen_off_360", True).value)
        is_360 = ("eve" in self.lidar_topic) or ("mid" in self.lidar_topic.lower())
        if self.opt_degen_off_360 and is_360:
            self.degen_enable = False

        # --- robustness guards ---
        self.opt_hold_on_skip = bool(gp("opt_hold_on_skip", True).value)
        self.opt_max_step = float(gp("opt_max_step", 1.0).value)   # [m] max |Δp|/scan (was 0.6)
        self._n_hold = 0
        self._n_clamp = 0

        self.get_logger().info(
            f"  nrlio_optimized: accum_voxel={self.nr_accum_voxel} voxel_min={self.gen_d_min} "
            f"chi2={self.chi2_threshold:.0f} knee={self.nr_knee_ms:.0f}ms({self.nr_accum_max}f) "
            f"degen(planar={self.degen_planarity_ratio},extent={self.degen_min_extent},"
            f"{'OFF-360' if not self.degen_enable else 'on'}) "
            f"| guards: hold_on_skip={self.opt_hold_on_skip} max_step={self.opt_max_step}m")

    def _register(self):
        # last accepted pose (previous scan's output); the anchor the guards hold to
        p_prev = self._last_pose_p.copy()
        super()._register()
        m = self.last_method

        if m == "stationary(zupt)":
            return                          # ZUPT already holds pose + zeros velocity

        # (1) registration unavailable on a degenerate scan -> HOLD translation,
        #     keep gyro-propagated rotation, zero velocity.  Never IMU-dead-reckon.
        if self.opt_hold_on_skip and m.startswith("skip_degen"):
            self.kf.p = p_prev.copy()
            self.kf.v = np.zeros(3)
            self._skip_map = True
            self.last_method = "skip_degen(hold)"
            self._n_hold += 1
            return

        # (2) per-scan motion clamp: reject implausible jumps from any path.
        step = float(np.linalg.norm(self.kf.p - p_prev))
        if step > self.opt_max_step:
            d = (self.kf.p - p_prev) / (step + 1e-9)
            self.kf.p = p_prev + d * self.opt_max_step
            self.kf.v = np.zeros(3)
            self._skip_map = True
            self.last_method = f"{m}|clamp{step:.1f}m"
            self._n_clamp += 1

    def shutdown(self):
        super().shutdown()
        self.get_logger().info(
            f"[nrlio_opt] guards fired: skip_degen-holds={self._n_hold}, "
            f"motion-clamps={self._n_clamp} (max_step={self.opt_max_step}m)")


def main():
    run_node(SuperLioNRLIOOpt)


if __name__ == "__main__":
    main()
