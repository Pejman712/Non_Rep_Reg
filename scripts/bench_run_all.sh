#!/usr/bin/env bash
# bench_run_all.sh — run every benchmark sequence and plot all three curves.
#
# Each bench_run.sh invocation automatically produces:
#   result_regnonrep_imu.tum      — IMU + LiDAR fused
#   result_regnonrep_imuonly.tum  — IMU dead-reckoning only
#   result_regnonrep_noimu.tum    — LiDAR only
#
# Estimated wall-clock time (sequential): ~3 h 25 min
#   Tier (0.1x): 6 seq × 2 passes ≈ 142 min
#   iilab (1x):  4 seq × 2 passes ≈  63 min

set -euo pipefail
SCRIPT_DIR="$(realpath "$(dirname "$0")")"
BENCH="${SCRIPT_DIR}/bench_run.sh"

log() { echo "[bench_all] $(date '+%H:%M:%S')  $*"; }

run() {
    local label="$1"; shift
    log "START  $label"
    local t0=$SECONDS
    bash "$BENCH" "$@"
    local elapsed=$(( SECONDS - t0 ))
    log "DONE   $label  ($(( elapsed/60 ))m $(( elapsed%60 ))s)"
}

START_SEC=$SECONDS

# ── Tier / Livox Avia ────────────────────────────────────────────────────────
for seq in indoor1_avia indoor2_avia indoor3_avia; do
    run "$seq" "$seq" --dataset tier --full-bag
done

# ── Tier / Livox Horizon ─────────────────────────────────────────────────────
for seq in indoor1_horizen indoor2_horizen indoor3_horizen; do
    run "$seq" "$seq" --dataset tier_horizen --full-bag
done

# ── iilab ────────────────────────────────────────────────────────────────────
for seq in nav_a_diff nav_a_omni loop slippage; do
    run "$seq" "$seq" --full-bag
done

ELAPSED=$(( SECONDS - START_SEC ))
log "ALL DONE  total=$(( ELAPSED/60 ))m $(( ELAPSED%60 ))s"
