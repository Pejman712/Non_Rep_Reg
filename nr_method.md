# The `nrlio_optimized` algorithm

A complete, ground-up explanation of `nrlio_optimized` — the algorithm, every
stage, and why each piece is there. It is a stack of four classes, each adding a
layer:

```
SuperLioBase        (lio_base.py)          → iESKF LiDAR-inertial backbone
  └ SuperLioGen     (lio_gen_lio.py)       → adaptive voxelization + hybrid registration + degeneracy routing
      └ SuperLioNRLIO (lio_nrlio.py)       → gyro-adaptive accumulation + ZUPT
          └ SuperLioNRLIOOpt (lio_nrlio_optimized.py) → tuned params + robustness guards
```

---

## 1. State representation — an iterated error-state Kalman filter (iESKF)

The filter carries an **18-dimensional error state** around a nominal state:

| block | symbol | meaning |
|---|---|---|
| 0:3 | R | orientation (SO(3), stored as a rotation matrix) |
| 3:6 | p | position |
| 6:9 | v | velocity |
| 9:12 | b_g | gyro bias |
| 12:15 | b_a | accelerometer bias |
| 15:18 | g | gravity vector (kept at 9.8015 magnitude) |

with an 18×18 covariance `P`. "Error-state" means the filter estimates small
corrections `δx` to this nominal state; "iterated" means each measurement update
is solved by several Gauss-Newton iterations (`kf_max_iterations`, default 4)
re-linearizing at the current estimate.

---

## 2. Inputs & IMU conditioning

- **LiDAR**: raw Livox stream (`/avia/livox/points`, `/livox/points`, or Mid-360
  `/eve/lidar3d`). Each point carries an intra-scan timestamp (`offset_time`).
- **IMU**: For the Tier sensors, the Avia/Horizon built-in IMU reports
  acceleration in **g**, so a rescaler (`imu_rescale.py`) multiplies accel ×9.81
  → m/s² and republishes to `/bench/imu`. The Mid-360's Xsens IMU is already m/s²
  (no rescale). This is applied via `self.imu_scale`.
- IMU samples are buffered; each LiDAR scan is paired with the batch of IMU
  samples spanning it (`self._meas_imu`) plus the scan payload (`self._meas_lidar`).

---

## 3. The per-scan pipeline (`_state_process`)

Every scan runs seven steps, timed with `perf_counter` for the proc-ms diagnostics:

```
1. _propagation_undistort   ← IMU predict across the scan + per-point deskew
2. _prefilter               ← optional light cleanup (off by default)
3. _downsample
4. _register                ← the variant-specific brain (detailed below)
5. force_z_zero             ← optional 2-D constraint (z=0, vz=0)
6. _update_map              ← insert scan into the voxel map (unless frozen)
7. _output                  ← publish odom + write TUM + buffer for accumulation
```

### Step 1 — IMU propagation + continuous-time deskew

For each IMU sample in the scan window, `predict()` does midpoint integration:

- debiased gyro `ω = ½(gyrₖ+gyrₖ₋₁) − b_g`, accel `a = ½(accₖ+accₖ₋₁)·scale − b_a`
- `R ← R·exp(ω·dt)`, `p ← p + v·dt + ½(R·a+g)·dt²`, `v ← v + (R·a+g)·dt`
- covariance `P ← F P Fᵀ + Fw Q Fwᵀ` with the standard ESKF Jacobians
  (right-Jacobian gyro-bias block, skew(accel) coupling, etc.)

It stores the pose at each IMU sub-step. Then **deskew**: every LiDAR point is
undistorted at *its own* `offset_time` by interpolating (SLERP for rotation via
axis-angle, quadratic for position) between the bracketing IMU sub-poses, and
expressed in the end-of-scan frame. This is the continuous-time correction that
makes the accumulated scan geometrically consistent even under motion.

### Step 4 — Registration: the decision tree

This is where `nrlio_optimized` layers everything. Reading it top-down:

**(4a) ZUPT stationarity gate** *(from nrlio)* — runs first.
`_is_stationary()` looks at the IMU batch only: mean gyro magnitude
`‖gyr−b_g‖ < zupt_gyro_thresh` (0.015 rad/s) **and** the accelerometer-magnitude
coefficient of variation `std(|a|)/mean(|a|) < zupt_acc_cv` (0.012, unit-free so
it works for g and m/s²). With a 3-scan debounce, if truly still it **holds** the
pose at the anchor captured at stillness onset, **zeros velocity** (kills IMU
drift), **freezes the map**, tags `stationary(zupt)`, and returns. This is why a
stationary robot doesn't drift.

**(4b) Accumulation** *(from nrlio, `_gen_preprocess`)* — gyro-adaptive coverage.
Livox rosette coverage grows with integration time (we measured the knee at
~200 ms). So it accumulates the last few deskewed scans into the current frame,
but the **depth is gyro-gated**: slow motion → up to `nr_accum_max` frames (here
2, from `nr_knee_ms=200`); fast turn (`|ω| ≥ nr_gyro_hi=1.0`) → single scan
(avoid smearing). Buffered scans are **motion-compensated** into the current
frame using each scan's stored pose before merging, then voxel-downsampled at
`nr_accum_voxel` (0.16).

**(4c) Adaptive voxelization** *(from gen_lio, `_gen_scale_and_voxel`)* — GenZ-style.
A smoothed **scale indicator** m̄ = mean over a window of the median point range
tells whether the scene is confined or open. A scale-informed target point-count
`N_desired` is set, and a **PD controller** adjusts the downsampling voxel size
`d` so the voxelized cloud tracks that target — finer in tight rooms, coarser in
open space. `d` drives both registration clouds.

**(4d) Degeneracy detection** *(from gen_lio, `_scan_degeneracy`)*.
PCA on the body-frame cloud: eigenvalues λ₀≤λ₁≤λ₂ of the covariance. The scan is
**degenerate** if too few points, or dominant-axis spread `√λ₂ < degen_min_extent`
(concentrated), or `λ₀/λ₂ < degen_planarity_ratio` (collapsed onto a plane/line —
e.g. hugging a wall, so motion parallel to it is unobservable). On Mid-360 this
check is **disabled** in `nrlio_optimized` (the 360° FoV isn't degeneracy-prone
and the guard misfired there).

**(4e) Routing:**

- **Degenerate** → freeze the map and route to **GICP** (its point-to-point-style
  constraints still anchor observable directions). If GICP is unavailable/rejected
  → `skip_degen` (would otherwise dead-reckon).
- **Well-conditioned** → **scale-switched hybrid** with hysteresis
  (`gen_switch_low/high`): confined scenes → **point-to-plane**; open scenes →
  **nonrep + GICP**.

**Point-to-plane update** (`_observe_point_to_plane`): for each downsampled body
point, find its 5 nearest map points (`ivox.knn5`), fit a local plane (n, d),
residual `err = n·(Rp+t)+d`. A **distance-scaled robust gate** keeps a
correspondence only if `range > 81·err²` (rejects outliers, tighter far away).
The Jacobian is full 6-DoF `J = [ (p×(nᵀR))ᵀ , n ]`, and the information terms
`HᵀWH`, `HᵀW·err` feed the **iterated** Kalman update.

**GICP path**: the nonrep front-end (`NonRepetitiveLiDARProcessor` —
feature/geometric/temporal-extrapolation blend) produces the **initial guess**,
GICP aligns the scan to the local submap, and the resulting absolute pose is
fused via `update_pose`, which is **χ²-gated**:
`χ² = r₀ᵀ(P₀:₆ + R_n)⁻¹r₀`, rejected if above `gicp_chi2_threshold` (200 here).
Rejected updates don't corrupt the state.

**(4f) Robustness guards** *(new in `nrlio_optimized`)* — applied after the base
registration returns:

- **Guard 1 — hold-on-skip.** If the method came back `skip_degen`, instead of
  keeping the IMU-dead-reckoned pose (which exploded +45 m on horizon3), it
  **holds translation at the previous pose, keeps the gyro-propagated rotation,
  zeros velocity, freezes the map** → `skip_degen(hold)`. Rationale: gyro rotation
  is trustworthy, double-integrated accel is not — "when you can't see, don't move."
- **Guard 2 — motion clamp.** Compute the per-scan step `‖p − p_prev‖`; if it
  exceeds `opt_max_step` (0.6 m — implausible at indoor speed), **clamp** the
  translation back to 0.6 m in the same direction, zero velocity, freeze the map.
  A hard backstop against explosions from *any* path.

### Steps 5–7

- **`force_z_zero`** (Tier is treated as planar) pins z and vz to 0 after the update.
- **`_update_map`** inserts the (world-frame) scan into the **voxel-hash map**
  (`ivox`, OctVoxMap-style: one centroid per cell, FIFO eviction at
  `voxel_map_max_voxels`, pruned beyond `voxel_map_prune_radius`) — unless the
  scan was frozen by degeneracy/ZUPT/guards.
- **`_output`** publishes `/lio/odom` + TF, streams the pose to the evaluated TUM,
  and **buffers this scan** (downsampled body cloud + final pose) into the
  accumulation deque for the next frame; it also records the pose as the
  ZUPT/guard anchor.

---

## 4. The campaign-optimized parameters (and why)

`nrlio_optimized.__init__` forces these over the base/config defaults, from the
14 h sweep:

| param | value | reason |
|---|---|---|
| `nr_accum_voxel` | **0.16** | finer (0.12) diverged on the Tier screen set; 0.16 gave 0 divergences |
| `gicp_chi2_threshold` | **200** | the tight default (50) *rejected good updates* and dead-reckoned into divergence |
| `nr_knee_ms` | **200** (2-frame) | 3+ frames hurt; 5 was catastrophic |
| ZUPT | **on** | removing it made everything worse |
| degeneracy routing | **off on Mid-360** | the narrow-FoV guard misfires on 360°; off → `loop` 15.7 → 0.02 m |
| `opt_max_step` | 0.6 m | motion-clamp threshold |
| `opt_hold_on_skip` | true | hold-instead-of-dead-reckon |

Mid-360 is auto-detected from the lidar topic (`/eve/lidar3d`).

---

## 5. What it is, in one paragraph

`nrlio_optimized` is a tightly-coupled iESKF LiDAR-inertial odometry for
narrow/low-FoV non-repetitive Livox sensors. It continuously-time-deskews each
scan, accumulates a gyro-gated number of frames to reach the sensor's coverage
knee, adaptively voxelizes by scene scale, and registers with a **scale-switched
hybrid** (point-to-plane in tight scenes, nonrep-seeded GICP in open ones),
routing geometrically degenerate scans to GICP with a frozen map. On top sit
three "don't move when you can't measure" safeguards — a ZUPT stationarity hold,
a hold-on-skip for unregisterable scans, and a per-scan motion clamp — plus the
campaign-tuned constants.

---

## 6. Honest limitations (from what we've seen)

- **Rotation reporting**: point-to-plane/GICP correct orientation, but the reported
  orientation still carries the ~110° Tier `lidar→base_link` frame offset — real
  *shape* (ATE-SE3) is good; the raw orientation metric looks bad.
- **indoor3 still fails**: the guards stop *explosions* (horizon3 36→13.5 m) but
  not the **confident smooth slide** in feature-poor starts (avia3) — that needs
  degenerate-direction locking (TSVD), which isn't in this variant yet.
- **Non-determinism**: results vary run-to-run because the bench feeds data in
  real-time through a multi-threaded node, amplified on near-divergence sequences.
