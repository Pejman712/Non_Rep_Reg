#!/usr/bin/env bash
# run_all.sh — run every dataset/sequence with IMU and without IMU
#
# Usage:
#   ./run_all.sh [--skip-iilab] [--skip-tier] [--skip-tier-horizen] [--dry-run]
#
# All bags are played in full (--full-bag).
# Per-run logs are written to <LOG_DIR>/<dataset>_<sequence>_<imu|noimu>.log
# A summary is printed when all runs complete.

set -uo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
BENCH="${SCRIPT_DIR}/bench_run.sh"
LOG_DIR="${SCRIPT_DIR}/../benchmark_results/$(date +%Y%m%d_%H%M%S)"

RUN_IILAB=true
RUN_TIER=true
RUN_TIER_HORIZEN=true
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --skip-iilab)         RUN_IILAB=false ;;
        --skip-tier)          RUN_TIER=false ;;
        --skip-tier-horizen)  RUN_TIER_HORIZEN=false ;;
        --dry-run)          DRY_RUN=true ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ── Benchmark matrix ─────────────────────────────────────────────────────────
# Format: "dataset:sequence"
RUNS=()
$RUN_IILAB      && RUNS+=(
    "iilab:nav_a_diff"
    "iilab:nav_a_omni"
    "iilab:loop"
    "iilab:slippage"
)
$RUN_TIER       && RUNS+=(
    "tier:indoor1_avia"
    "tier:indoor2_avia"
    "tier:indoor3_avia"
)
$RUN_TIER_HORIZEN && RUNS+=(
    "tier_horizen:indoor1_horizen"
    "tier_horizen:indoor2_horizen"
    "tier_horizen:indoor3_horizen"
)

TOTAL=${#RUNS[@]}
CURRENT=0
PASS=()
FAIL=()

mkdir -p "$LOG_DIR"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  run_all.sh — ${TOTAL} sequences (each runs fused + no-imu passes)"
echo "  logs → ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════"

# ── Main loop ─────────────────────────────────────────────────────────────────
for entry in "${RUNS[@]}"; do
    ds="${entry%%:*}"
    seq="${entry##*:}"

    tag="${ds}/${seq}"
    log="${LOG_DIR}/${ds}_${seq}.log"
    CURRENT=$(( CURRENT + 1 ))

    echo ""
    echo "────────────────────────────────────────────────────────────"
    printf "  [%d/%d]  %s\n" "$CURRENT" "$TOTAL" "$tag"
    echo "────────────────────────────────────────────────────────────"

    if $DRY_RUN; then
        echo "  DRY-RUN: bash ${BENCH} ${seq} --dataset ${ds} --full-bag"
        PASS+=("$tag")
        continue
    fi

    # Run bench_run.sh; tee to log + stdout; capture its exit code via PIPESTATUS.
    # A single call already runs both the fused (IMU+LiDAR) and no-imu passes.
    bash "$BENCH" "$seq" --dataset "$ds" --full-bag 2>&1 | tee "$log"
    rc="${PIPESTATUS[0]}"

    if [[ $rc -eq 0 ]]; then
        PASS+=("$tag")
        echo "  [OK] ${tag}"
    else
        FAIL+=("$tag")
        echo "  [FAILED] ${tag}  (rc=${rc})  — log: ${log}" >&2
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  SUMMARY  (${#PASS[@]} passed / ${#FAIL[@]} failed / ${TOTAL} total)"
echo "════════════════════════════════════════════════════════════"
if [[ ${#PASS[@]} -gt 0 ]]; then
    echo "  PASSED:"
    for t in "${PASS[@]}"; do echo "    OK  $t"; done
fi
if [[ ${#FAIL[@]} -gt 0 ]]; then
    echo "  FAILED:"
    for t in "${FAIL[@]}"; do echo "    !!  $t"; done
fi
echo "  Logs: ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════"

[[ ${#FAIL[@]} -eq 0 ]]   # exit 0 if everything passed, 1 otherwise
