# regnonrep LIO — progress report

_Development of a LiDAR-inertial odometry variant for narrow / low-FoV
non-repetitive Livox sensors (Avia 70°×70°, Horizon 80°×27°, Mid-360 360°×59°),
benchmarked on the Tier (Avia/Horizon) and iilab (Mid-360) datasets._

Companion doc: [`nr_method.md`](nr_method.md) — full algorithm description.

---

## 1. Goal

Build a non-repetitive-LiDAR-aware LIO (`nrlio`) that produces an accurate,
robust trajectory on the narrow-FoV Tier sensors, and drive it to the best
achievable accuracy through measurement, ablation, and a parameter-optimization
campaign — all runnable from the web bench.

---

## 2. Sensor characterization (measured from the bags)

Grounded the design in measurements of the Livox rosette pattern:

- **Coverage vs integration time** (fraction of a 0.5° FoV grid filled):
  | sensor | 100 ms (1 scan) | 200 ms | knee |
  |---|---|---|---|
  | Avia (70°×70°) | 57 % | 75 % | ~200 ms |
  | Horizon (80°×27°) | 85 % | 94 % | ~100 ms |
  Horizon fills fast (narrow vertical band); Avia needs ~2 scans.
- **Motion per accumulation window** (indoor GT): 100 ms → 3.2 cm / 0.7° median
  (p95 rotation ~11°, i.e. fast turns smear a fixed window).
- **Beam self-revisit within one 100 ms scan**: Avia ~5.4 ms, Horizon ≤2 ms —
  the center of the FoV is re-sampled ~18× per frame (this is *intra*-scan, not
  the 10 Hz frame rate).

**Conclusion:** accumulation depth must be sensor-specific and motion-adaptive.

---

## 3. The `nrlio` variant family

Built as an inheritance stack on the existing `SuperLioBase` (iESKF) →
`SuperLioGen` (adaptive voxelization + hybrid registration + degeneracy routing):

### `nrlio` (`lio_nrlio.py`)
- **Gyro-adaptive, coverage-targeted accumulation** — accumulates up to a
  sensor-specific coverage knee, but collapses to a single scan on fast turns.
- **ZUPT stationarity gate** — IMU-only "no motion" detection (low gyro + low
  accel-CV, debounced); holds pose + zeros velocity + freezes map when still.
- Inherits continuous-time per-point deskew, nonrep-seeded scale-switched
  point-to-plane ↔ GICP registration, and degeneracy→GICP routing.
- **Loop closure**: a keyframe SE(2) pose-graph back-end was added, then
  **removed** — it degraded accuracy/performance on these short indoor sequences.

### `nrlio_loop` (removed)
- Fixed a bench-timing problem (the shutdown-time back-end was killed before it
  finished) by triggering loop closure on **input-idle** instead. Kept as a
  variant briefly, then removed along with loop closure.

### `nrlio_optimized` (`lio_nrlio_optimized.py`) — the current best
- `nrlio` + campaign-tuned parameters + two robustness guards (below).

---

## 4. Robustness guards (from log diagnosis of indoor3 divergences)

The annotated plots exposed that `indoor3` (both sensors) diverged and the run
truncated (~408 poses). Two distinct failure modes were found in the logs:

- **Horizon3 = explosion**: a `skip_degen` scan IMU-dead-reckoned and jumped
  **+45 m** in a few scans.
- **Avia3 = confident slide**: normal p2p registration *confidently* (conf 0.76)
  locked onto the wrong solution in a feature-poor start.

Two guards were added to `nrlio_optimized`:
1. **Hold-on-skip** — when registration is unavailable, HOLD translation, keep
   the gyro-propagated rotation, zero velocity ("when you can't see, don't
   move"). This eliminated the +45 m explosion (Horizon3 max excursion 36 → 13.5 m,
   ATE 10.4 → 4.5 m).
2. **Motion clamp** — reject any per-scan step > `opt_max_step` as implausible.

The guards fixed the *explosions*; the *confident-slide* mode needs
degenerate-direction locking (TSVD), noted as future work.

---

## 5. Parameter-optimization campaigns (web-bench-driven)

Added **per-run parameter overrides** to the bench (`--params-overlay` in
`run_benchmarks.sh` + the launch file), a resumable **campaign orchestrator**
(`nrlio_campaign.py`) that drives `run_benchmarks.sh` per experiment, and a
**WebSocket control panel** (`campaign_web.py`, port 8078) with Start/Stop + live
progress.

### Campaign 1 — broad sweep (all 10 sequences, 1.0×)
- Found the dominant levers but was noisy (node couldn't keep up at 1.0×), and a
  parser bug scored the Mid-360 (iilabs3d format) as failures.
- Established: coarse accumulation voxel + looser chi2 *stop* divergence; degeneracy
  routing *hurts* the 360° Mid-360 (→ disabled there); accumulation depth of 1–2
  frames beats 3+.

### Campaign 2 — Tier convergence campaign (6 Tier sequences)
Redesigned from the log diagnosis of the accuracy *plateau* (tracks but stalls at
~1–2.5 m): swept the convergence levers, confirmed the best with **3 repeats ×
all 6**. **Result:**

| config | mean ATE-SE(3) | diverged reps |
|---|---|---|
| **combined-best** | **0.996 m** | **0 / 18** |
| previous `nrlio_optimized` | 2.745 m | 0 |

**A ~64 % error reduction, every sequence < 2.2 m, zero divergences.**

Key findings (some confirmed the diagnosis, one refuted it):
- ✅ **Finer voxels win** now that the guards prevent divergence (accum 0.08 > 0.16;
  registration floor 0.03 > 0.05) — the precision ceiling was real.
- ✅ **Tighter chi2 = 100 > 200** — the loose gate admitted conf≈0.1 GICP updates.
- ✅ **Flag fewer scans degenerate** (planarity 0.01 > 0.03) — it was coasting too much.
- ✅ **Less accumulation** (1-frame, knee 100) beat 2–3 frames.
- ❌ **More iterations HURT** (kf_iters 4 > 6 > 8) — extra Gauss-Newton steps
  over-fit noisy/degenerate correspondences (my "raise iterations" hypothesis was wrong).

---

## 6. Final baked-in configuration (`nrlio_optimized`)

```yaml
opt_accum_voxel: 0.08        # finer accumulation voxel (was 0.16)
opt_voxel_min:   0.03        # finer registration voxel floor (was 0.05)
opt_chi2:        100         # tighter acceptance (was 200)
opt_knee_ms:     100         # 1-frame accumulation (was 200/2-frame)
opt_degen_planarity: 0.01    # flag fewer scans degenerate (was 0.03)
opt_degen_extent:    0.5     # (was 1.0)
kf_max_iterations:   4       # (6/8 were worse)
opt_max_step:    1.0         # motion clamp
opt_hold_on_skip: true       # hold-instead-of-dead-reckon
# Mid-360 only: degeneracy routing auto-disabled (detected from lidar topic)
```

Expected performance: **~1.0 m mean ATE-SE(3) across the 6 Tier sequences, 0
divergences.** All values remain overridable via the `opt_*` params.

---

## 7. Diagnostics & visualization tooling

- **Annotated route plots** (`plot_annotated.py`) — the estimate is coloured by
  registration mode (point-to-plane / GICP / degenerate) with icons for
  ZUPT / degeneracy / motion-clamp, plus companion strips for the adaptive voxel
  size and accumulation depth. Auto-selected by the bench for `nrlio*` variants
  via a per-pose annotation sidecar (`*.ann.csv`). Lets you *see* where each
  mechanism fires — e.g. Horizon is degeneracy-dominated (~30 % of scans).
- **Standard metrics** in the bench: origin-ATE, ATE-SE(3), rotation-ATE, RPE-1m,
  plus per-scan processing time and event diagnostics.

---

## 8. Key cross-cutting findings

1. **Frame offset**: a persistent ~110–130° rotation between the estimate and GT
   frames (uncalibrated lidar→base_link extrinsic) **inflates the origin-aligned
   RMSE**. The true accuracy is the **ATE-SE(3)** column; the trajectory *shape*
   is good even when the raw plot looks rotated.
2. **Run-to-run variance**: the bench feeds data in real-time through a
   multi-threaded node, so results are **non-deterministic** — amplified on
   near-divergence sequences (indoor3 flipped between 0.4 m and 10–19 m across
   runs of identical code). Mitigations: play at ≤0.5× (node keeps up), average
   over repeats, or build a deterministic offline replay.
3. **The bench runs methods sequentially**, replaying the bag once per (method,
   sequence) — so methods aren't compared on identical input. A record-once /
   replay-to-all offline runner would fix both variance and fairness.
4. **Horizon is the hard sensor** — its narrow 27° vertical FoV is geometrically
   degenerate ~30 % of the time; the guards + tuning are what keep it converging.

---

## 9. Files changed / added

| file | purpose |
|---|---|
| `scripts/lio_nrlio.py` | `nrlio` variant (accumulation + ZUPT + annotation sidecar) |
| `scripts/lio_nrlio_optimized.py` | `nrlio_optimized` — campaign-best params + guards |
| `scripts/plot_annotated.py` | annotated route plots |
| `scripts/nrlio_campaign.py` | resumable param-optimization campaign orchestrator |
| `scripts/campaign_web.py` | WebSocket campaign control panel (port 8078) |
| `scripts/run_benchmarks.sh` | `--params-overlay`, annotation copy + plotter choice, `PYTHONUNBUFFERED` |
| `launch/lio_variant.launch.py` | `params_overlay` arg; skip `odom_to_tum` for `nrlio*` |
| `config/lio_tier_avia.yaml`, `lio_tier_horizen.yaml`, `lio.yaml` | nrlio param blocks |
| `nr_method.md` | full algorithm description |
| `CMakeLists.txt` | install the new scripts |

---

## 10. Known limitations & next steps

- **Confident-slide divergence** (avia3-style) — add **TSVD degenerate-direction
  locking**: update the observable subspace, hold only the unobservable DoF,
  instead of holding/skipping the whole pose (recovers correction on the ~30 % of
  Horizon scans currently coasted).
- **Confidence-weighted fusion** — scale measurement information by registration
  fitness instead of a binary chi² accept/reject.
- **Frame extrinsic** — estimate the lidar→body rotation (offline or online) so
  the reported orientation and origin-aligned RMSE become meaningful.
- **Deterministic offline replay** — record the converted stream once and feed
  each method synchronously for reproducible, fair comparison.
- **Re-confirm `nrlio_optimized` at 0.5×** — Campaign 2 ran at 1.0× (variance);
  the clean-input numbers should be even better and tighter.
- **Mid-360 confirmation** — validate the baked-in config on the iilab sequences
  (degeneracy auto-off there).

---

_Report generated 2026-07-08._
