#!/usr/bin/env bash
# bench_run.sh — automated benchmark pipeline for regnonrep
#
# Usage:
#   ./bench_run.sh <sequence> [sensor] [flags]
#
# A single invocation (without --no-imu) always produces all three trajectories:
#   result_regnonrep_imu.tum      — IMU + LiDAR fused
#   result_regnonrep_imuonly.tum  — IMU dead-reckoning only (no LiDAR correction)
#   result_regnonrep_noimu.tum    — LiDAR only (no IMU)
#
# The final plot contains all three curves.
# Pass --no-imu to produce only the LiDAR-only trajectory (single pass).
#
# Examples:
#   ./bench_run.sh nav_a_diff
#   ./bench_run.sh loop livox_mid-360
#   ./bench_run.sh nav_a_diff --no-frame-correct
#   ./bench_run.sh loop --duration 60
#   ./bench_run.sh loop --full-bag
#   ./bench_run.sh loop --no-imu
#   ./bench_run.sh indoor1_avia --dataset tier
#   ./bench_run.sh indoor2_avia --dataset tier --full-bag
#   ./bench_run.sh indoor3_avia --dataset tier --full-bag
#   ./bench_run.sh indoor1_horizen --dataset tier_horizen --full-bag
#   ./bench_run.sh indoor2_horizen --dataset tier_horizen --full-bag
#   ./bench_run.sh indoor3_horizen --dataset tier_horizen --full-bag
#
# iilab dataset sequences: nav_a_diff  nav_a_omni  loop  slippage
# Tier/Avia sequences (--dataset tier): indoor1_avia  indoor2_avia  indoor3_avia
# Tier/Horizon sequences (--dataset tier_horizen): indoor1_horizen  indoor2_horizen  indoor3_horizen

set -euo pipefail

WS="$(realpath "$(dirname "$0")/../../..")"
TUM_LIVE="${WS}/src/regnonrep/tum/lio_odom.tum"
TUM_LIVE_IMUONLY="${WS}/src/regnonrep/tum/lio_imuonly.tum"

NO_FRAME_CORRECT=false
DURATION=10   # seconds; 0 = full bag
USE_IMU=true
DATASET=iilab
USER_GT=""    # optional path to GT file (.csv or .tum)
BAG_RATE=""   # playback rate (empty = use dataset default)

# ── Known ground-truth CSV paths (auto-loaded when --gt is omitted) ───────────
declare -A KNOWN_GT=(
    [indoor1_avia]="/u/97/habibip1/unix/Downloads/indoor01_optitrack.csv"
    [indoor2_avia]="/u/97/habibip1/unix/Downloads/indoor02_optitrack.csv"
    [indoor3_avia]="/u/97/habibip1/unix/Downloads/indoor03_optitrack.csv"
    [indoor1_horizen]="/u/97/habibip1/unix/Downloads/indoor01_optitrack.csv"
    [indoor2_horizen]="/u/97/habibip1/unix/Downloads/indoor02_optitrack.csv"
    [indoor3_horizen]="/u/97/habibip1/unix/Downloads/indoor03_optitrack.csv"
)

# parse flags
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-frame-correct) NO_FRAME_CORRECT=true ; shift ;;
        --full-bag)         DURATION=0            ; shift ;;
        --duration)         DURATION="$2"         ; shift 2 ;;
        --duration=*)       DURATION="${1#*=}"    ; shift ;;
        --no-imu)           USE_IMU=false         ; shift ;;
        --dataset)          DATASET="$2"          ; shift 2 ;;
        --dataset=*)        DATASET="${1#*=}"     ; shift ;;
        --gt)               USER_GT="$2"          ; shift 2 ;;
        --gt=*)             USER_GT="${1#*=}"     ; shift ;;
        --bag-rate)         BAG_RATE="$2"         ; shift 2 ;;
        --bag-rate=*)       BAG_RATE="${1#*=}"    ; shift ;;
        *) POSITIONAL+=("$1") ; shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

SEQUENCE="${1:?Usage: $0 <sequence> [sensor] [--dataset iilab|tier|tier_horizen] [--no-imu]}"

# ── dataset-specific settings ─────────────────────────────────────────────────
if [[ "$DATASET" == "tier" ]]; then
    DATASET_ROOT="/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/Tier"
    SENSOR="${2:-Livox_avia}"
    INSTALLED_CFG="${WS}/install/regnonrep/share/regnonrep/config/lio_tier_avia.yaml"
    LAUNCH_PKG_FILE="lio_tier.launch.py"
    REQUIRE_GT=false
    FRAME_CORRECT=false
    [[ -z "$BAG_RATE" ]] && BAG_RATE="0.3"
    POST_BAG_WAIT=30
elif [[ "$DATASET" == "tier_horizen" ]]; then
    DATASET_ROOT="/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/Tier"
    SENSOR="${2:-Livox_horizen}"
    INSTALLED_CFG="${WS}/install/regnonrep/share/regnonrep/config/lio_tier_horizen.yaml"
    LAUNCH_PKG_FILE="lio_tier_horizen.launch.py"
    REQUIRE_GT=false
    FRAME_CORRECT=false
    [[ -z "$BAG_RATE" ]] && BAG_RATE="0.3"
    POST_BAG_WAIT=30
else
    DATASET_ROOT="/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/iilab_benchmark"
    SENSOR="${2:-livox_mid-360}"
    INSTALLED_CFG="${WS}/install/regnonrep/share/regnonrep/config/lio.yaml"
    LAUNCH_PKG_FILE="lio.launch.py"
    REQUIRE_GT=true
    FRAME_CORRECT=true
    [[ -z "$BAG_RATE" ]] && BAG_RATE="0.8"
    POST_BAG_WAIT=3
fi

# ── Auto-load known ground truth if --gt was not supplied ────────────────────
if [[ -z "$USER_GT" && -v KNOWN_GT["$SEQUENCE"] ]]; then
    USER_GT="${KNOWN_GT[$SEQUENCE]}"
    echo "      [gt] auto-selected: ${USER_GT}"
fi

SEQ_DIR="${DATASET_ROOT}/${SENSOR}/${SEQUENCE}"
BAG_DIR="${SEQ_DIR}/${SEQUENCE}"
GT_TUM="${SEQ_DIR}/${SEQUENCE}.tum"
RESULT_TUM_FUSED="${SEQ_DIR}/result_regnonrep_imu.tum"
RESULT_TUM_NOIMU="${SEQ_DIR}/result_regnonrep_noimu.tum"
RESULT_TUM_IMUONLY="${SEQ_DIR}/result_regnonrep_imuonly.tum"
PLOT_PNG="${SEQ_DIR}/trajectory_plot.png"
PLOT_SCRIPT="$(dirname "$0")/plot_tum.py"
CSV_TO_TUM="$(dirname "$0")/csv_to_tum.py"

# Primary result for evaluation: fused when IMU is on, LiDAR-only otherwise
PRIMARY_RESULT_TUM=$( $USE_IMU && echo "$RESULT_TUM_FUSED" || echo "$RESULT_TUM_NOIMU" )

# ── GT conversion (CSV → TUM) if --gt supplied ────────────────────────────────
if [[ -n "$USER_GT" ]]; then
    if [[ ! -f "$USER_GT" ]]; then
        echo "ERROR: GT file not found: ${USER_GT}" >&2
        exit 1
    fi
    if [[ "$USER_GT" == *.csv ]]; then
        echo "Converting GT CSV → TUM …"
        mkdir -p "$SEQ_DIR"
        python3 "$CSV_TO_TUM" "$USER_GT" "$GT_TUM"
    else
        mkdir -p "$SEQ_DIR"
        cp -f "$USER_GT" "$GT_TUM"
    fi
    REQUIRE_GT=true
fi

# ── validate ──────────────────────────────────────────────────────────────────
if [[ ! -d "$BAG_DIR" ]]; then
    echo "ERROR: bag directory not found: ${BAG_DIR}" >&2
    echo "  Sensor dirs available: $(ls "$DATASET_ROOT" 2>/dev/null | tr '\n' '  ')" >&2
    exit 1
fi
if [[ ! -f "$GT_TUM" ]]; then
    if $REQUIRE_GT; then
        echo "ERROR: ground truth not found: ${GT_TUM}" >&2
        exit 1
    else
        echo "WARNING: ground truth not found: ${GT_TUM} — evaluation will be skipped" >&2
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  dataset  : ${DATASET}"
echo "  sequence : ${SEQUENCE}"
echo "  sensor   : ${SENSOR}"
echo "  bag      : ${BAG_DIR}"
echo "  gt       : ${GT_TUM}"
if $USE_IMU; then
    echo "  passes   : fused (IMU+LiDAR)  +  LiDAR-only  [auto]"
else
    echo "  passes   : LiDAR-only  (--no-imu)"
fi
if [[ "$DURATION" -gt 0 ]]; then
    echo "  duration : first ${DURATION}s"
else
    echo "  duration : full bag"
fi
echo "════════════════════════════════════════════════════════════"
echo ""

# ── source workspace ──────────────────────────────────────────────────────────
set +u
source "${WS}/install/setup.bash"
set -u

# ── always restore config on exit (safety net) ───────────────────────────────
restore_cfg() {
    sed -i 's/use_imu: false/use_imu: true/' "$INSTALLED_CFG" 2>/dev/null || true
}
trap restore_cfg EXIT

# ─────────────────────────────────────────────────────────────────────────────
# run_pass PASS_USE_IMU RESULT_TUM
#   Launches the algorithm, plays the bag, collects the TUM, and aligns
#   timestamps.  Handles config patching and restoration internally.
# ─────────────────────────────────────────────────────────────────────────────
run_pass() {
    local pass_use_imu="$1"
    local pass_result_tum="$2"

    if $pass_use_imu; then
        echo ""
        echo "──────────────────────────────────────────────────────────────"
        echo "  PASS 1/2 : fused  (IMU + LiDAR)"
        echo "──────────────────────────────────────────────────────────────"
    else
        echo ""
        echo "──────────────────────────────────────────────────────────────"
        echo "  PASS 2/2 : LiDAR-only  (no IMU)"
        echo "──────────────────────────────────────────────────────────────"
    fi

    # patch config for IMU setting
    if ! $pass_use_imu; then
        sed -i 's/use_imu: true/use_imu: false/' "$INSTALLED_CFG"
        echo "      [config] use_imu → false"
    else
        sed -i 's/use_imu: false/use_imu: true/' "$INSTALLED_CFG"
        echo "      [config] use_imu → true"
    fi

    # kill any leftover nodes
    pkill -9 -f "ros_lio.py"             2>/dev/null || true
    pkill -9 -f "ros_non_rep.py"         2>/dev/null || true
    pkill -9 -f "odom_to_tum.py"         2>/dev/null || true
    pkill -9 -f "ros2 bag play"          2>/dev/null || true
    pkill -9 -f "ros2 launch regnonrep"  2>/dev/null || true
    sleep 1

    # clear live TUM files
    rm -f "$TUM_LIVE" "$TUM_LIVE_IMUONLY"

    # launch algorithm
    echo "      Launching regnonrep …"
    ros2 launch regnonrep "$LAUNCH_PKG_FILE" &
    local algo_pid=$!
    echo "      PID: ${algo_pid}"
    sleep 3

    # play bag
    echo "      Playing bag at rate ${BAG_RATE}x, skipping first 5s …"
    if [[ "$DURATION" -gt 0 ]]; then
        timeout "$DURATION" ros2 bag play "$BAG_DIR" --clock --rate "$BAG_RATE" --start-offset 5 || {
            rc=$?
            [[ $rc -eq 124 ]] && echo "      Reached ${DURATION}s limit." || exit $rc
        }
    else
        ros2 bag play "$BAG_DIR" --clock --rate "$BAG_RATE" --start-offset 5
    fi

    echo "      Bag done. Waiting ${POST_BAG_WAIT}s for flush …"
    sleep "$POST_BAG_WAIT"
    kill "$algo_pid" 2>/dev/null || true
    wait "$algo_pid" 2>/dev/null || true
    echo "      Algorithm stopped."

    # check and save fused/noimu result
    if [[ ! -s "$TUM_LIVE" ]]; then
        echo "ERROR: no TUM written at ${TUM_LIVE}" >&2
        exit 1
    fi
    cp "$TUM_LIVE" "$pass_result_tum"
    echo "      Saved → ${pass_result_tum}"

    # save IMU-only dead-reckoning (only produced by the fused pass)
    if $pass_use_imu && [[ -s "$TUM_LIVE_IMUONLY" ]]; then
        cp "$TUM_LIVE_IMUONLY" "$RESULT_TUM_IMUONLY"
        echo "      Saved → ${RESULT_TUM_IMUONLY}"
    fi

    # timestamp alignment for Tier datasets (hardware time → GT Unix time)
    if [[ ( "$DATASET" == "tier" || "$DATASET" == "tier_horizen" ) && -f "$GT_TUM" ]]; then
        echo "      Aligning timestamps to GT time base …"
        python3 - "$GT_TUM" "$pass_result_tum" <<'PYEOF'
import sys
gt_t0   = float(open(sys.argv[1]).readline().split()[0])
lines   = open(sys.argv[2]).readlines()
odom_t0 = float(lines[0].split()[0])
offset  = gt_t0 - odom_t0
aligned = []
for line in lines:
    parts = line.split()
    parts[0] = "{:.9f}".format(float(parts[0]) + offset)
    aligned.append(" ".join(parts))
open(sys.argv[2], "w").write("\n".join(aligned) + "\n")
print("  offset={:.3f}s  odom_t0={:.3f} -> gt_t0={:.3f}".format(
    offset, odom_t0, gt_t0))
PYEOF
        # align IMU-only with same method
        if $pass_use_imu && [[ -s "$RESULT_TUM_IMUONLY" ]]; then
            python3 - "$GT_TUM" "$RESULT_TUM_IMUONLY" <<'PYEOF'
import sys
gt_t0   = float(open(sys.argv[1]).readline().split()[0])
lines   = open(sys.argv[2]).readlines()
odom_t0 = float(lines[0].split()[0])
offset  = gt_t0 - odom_t0
aligned = []
for line in lines:
    parts = line.split()
    parts[0] = "{:.9f}".format(float(parts[0]) + offset)
    aligned.append(" ".join(parts))
open(sys.argv[2], "w").write("\n".join(aligned) + "\n")
print("  imu-only offset={:.3f}s".format(offset))
PYEOF
        fi
    fi

    # frame correction (iilab-specific)
    if ! $NO_FRAME_CORRECT && $FRAME_CORRECT; then
        echo "      Correcting frame: lidar → base_link (${SENSOR}) …"
        rm -f "${pass_result_tum%.tum}.orig.tum"
        iilabs3d correct-frame "$pass_result_tum" lidar \
            --sensor "${SENSOR//-/_}" 2>/dev/null || \
            echo "      (skipped — already in base_link or sensor not recognised)"
        # Apply the same correction to the IMU-only trajectory so all three
        # curves are expressed in the same frame for fair comparison.
        if $pass_use_imu && [[ -s "$RESULT_TUM_IMUONLY" ]]; then
            echo "      Correcting frame: IMU-only lidar → base_link (${SENSOR}) …"
            rm -f "${RESULT_TUM_IMUONLY%.tum}.orig.tum"
            iilabs3d correct-frame "$RESULT_TUM_IMUONLY" lidar \
                --sensor "${SENSOR//-/_}" 2>/dev/null || \
                echo "      (skipped — already in base_link or sensor not recognised)"
        fi
    fi
}

# ── run passes ───────────────────────────────────────────────────────────────
if $USE_IMU; then
    run_pass true  "$RESULT_TUM_FUSED"
    run_pass false "$RESULT_TUM_NOIMU"
else
    run_pass false "$RESULT_TUM_NOIMU"
fi

# ── evaluate (primary result) ─────────────────────────────────────────────────
echo ""
echo "[eval] Evaluating primary result: $(basename "$PRIMARY_RESULT_TUM") …"
echo ""
if [[ -f "$GT_TUM" ]]; then
    if [[ "$DATASET" == "tier" || "$DATASET" == "tier_horizen" ]]; then
        evo_ape tum "$GT_TUM" "$PRIMARY_RESULT_TUM" \
            --align \
            --t_max_diff 0.05 \
            --verbose 2>&1 || \
            echo "      (evo_ape warning — check timestamp alignment)"
    else
        iilabs3d eval "$GT_TUM" "$PRIMARY_RESULT_TUM" || \
            echo "      (eval warning — trajectory may be too short for 10m RTE intervals)"
    fi
else
    echo "      (skipped — no ground truth available)"
fi

# ── trajectory plot ──────────────────────────────────────────────────────────
echo ""
echo "[plot] Plotting trajectories …"
DURATION_ARG=""
[[ "$DURATION" -gt 0 ]] && DURATION_ARG="--duration ${DURATION}"

EST_FILES=()
for f in "$RESULT_TUM_FUSED" "$RESULT_TUM_NOIMU" "$RESULT_TUM_IMUONLY"; do
    [[ -f "$f" ]] && EST_FILES+=("$f")
done

GT_ARG=""
[[ -f "$GT_TUM" ]] && GT_ARG="--gt $GT_TUM"

python3 "$PLOT_SCRIPT" $GT_ARG "${EST_FILES[@]}" "$PLOT_PNG" $DURATION_ARG || \
    echo "      (plot failed — check matplotlib / numpy)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done: ${SEQUENCE} / ${SENSOR}"
for f in "$RESULT_TUM_FUSED" "$RESULT_TUM_NOIMU" "$RESULT_TUM_IMUONLY"; do
    [[ -f "$f" ]] && echo "  tum  : $f"
done
[[ -f "$PLOT_PNG" ]] && echo "  plot : ${PLOT_PNG}"
echo "════════════════════════════════════════════════════════════"
