#!/usr/bin/env bash
# bench_run.sh — automated iilab benchmark pipeline for regnonrep
#
# Usage:
#   ./bench_run.sh <sequence> [sensor]
#
# Examples:
#   ./bench_run.sh nav_a_diff
#   ./bench_run.sh loop livox_mid-360
#   ./bench_run.sh nav_a_diff --no-frame-correct
#   ./bench_run.sh loop --duration 60
#   ./bench_run.sh loop --full-bag
#   ./bench_run.sh loop --no-imu
#
# Available sequences: nav_a_diff  nav_a_omni  loop  slippage
# Available sensors:   livox_mid-360  (add others as downloaded)
#
# No topic remapping needed — bag topics already match the algorithm config:
#   /eve/lidar3d   (PointCloud2)
#   /eve/imu/data  (Imu)

set -euo pipefail

DATASET_ROOT="/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/iilab_benchmark"
WS="$(realpath "$(dirname "$0")/../../..")"
TUM_LIVE="${WS}/src/regnonrep/tum/lio_odom.tum"

NO_FRAME_CORRECT=false
DURATION=10   # seconds; 0 = full bag
USE_IMU=true

# parse flags
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-frame-correct) NO_FRAME_CORRECT=true ; shift ;;
        --full-bag)         DURATION=0            ; shift ;;
        --duration)         DURATION="$2"         ; shift 2 ;;
        --duration=*)       DURATION="${1#*=}"    ; shift ;;
        --no-imu)           USE_IMU=false         ; shift ;;
        *) POSITIONAL+=("$1") ; shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

SEQUENCE="${1:?Usage: $0 <sequence> [sensor] [--no-frame-correct]}"
SENSOR="${2:-livox_mid-360}"

SEQ_DIR="${DATASET_ROOT}/${SENSOR}/${SEQUENCE}"
BAG_DIR="${SEQ_DIR}/${SEQUENCE}"
GT_TUM="${SEQ_DIR}/${SEQUENCE}.tum"
IMU_SUFFIX=$( $USE_IMU && echo "imu" || echo "noimu" )
RESULT_TUM="${SEQ_DIR}/result_regnonrep_${IMU_SUFFIX}.tum"
PLOT_PNG="${SEQ_DIR}/trajectory_plot.png"
PLOT_SCRIPT="$(dirname "$0")/plot_tum.py"
INSTALLED_CFG="${WS}/install/regnonrep/share/regnonrep/config/lio.yaml"

# ── validate ──────────────────────────────────────────────────────────────────
if [[ ! -d "$BAG_DIR" ]]; then
    echo "ERROR: bag directory not found: ${BAG_DIR}" >&2
    echo "  Sensor dirs available: $(ls "$DATASET_ROOT" 2>/dev/null | tr '\n' '  ')" >&2
    exit 1
fi
if [[ ! -f "$GT_TUM" ]]; then
    echo "ERROR: ground truth not found: ${GT_TUM}" >&2
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  sequence : ${SEQUENCE}"
echo "  sensor   : ${SENSOR}"
echo "  bag      : ${BAG_DIR}"
echo "  gt       : ${GT_TUM}"
echo "  result   : ${RESULT_TUM}"
if [[ "$DURATION" -gt 0 ]]; then
    echo "  duration : first ${DURATION}s"
else
    echo "  duration : full bag"
fi
echo "  use_imu  : ${USE_IMU}"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── kill any leftover nodes from previous runs ────────────────────────────────
pkill -9 -f "ros_lio.py"    2>/dev/null || true
pkill -9 -f "ros_non_rep.py" 2>/dev/null || true
pkill -9 -f "odom_to_tum.py" 2>/dev/null || true
pkill -9 -f "ros2 bag play"  2>/dev/null || true
pkill -9 -f "ros2 launch regnonrep" 2>/dev/null || true
sleep 1

# ── source workspace ──────────────────────────────────────────────────────────
set +u
source "${WS}/install/setup.bash"
set -u

# ── patch installed config for IMU setting ───────────────────────────────────
if ! $USE_IMU; then
    sed -i 's/use_imu: true/use_imu: false/' "$INSTALLED_CFG"
    echo "      [config] use_imu patched → false"
fi
restore_cfg() {
    sed -i 's/use_imu: false/use_imu: true/' "$INSTALLED_CFG"
}
$USE_IMU || trap restore_cfg EXIT

# ── clear previous TUM ───────────────────────────────────────────────────────
rm -f "$TUM_LIVE"

# ── launch algorithm (background) ────────────────────────────────────────────
echo "[1/3] Launching regnonrep algorithm …"
ros2 launch regnonrep lio.launch.py &
ALGO_PID=$!
echo "      PID: ${ALGO_PID}"
sleep 3   # wait for nodes to initialise

# ── play bag ─────────────────────────────────────────────────────────────────
echo "[2/3] Playing bag …"
if [[ "$DURATION" -gt 0 ]]; then
    timeout "$DURATION" ros2 bag play "$BAG_DIR" --clock || {
        rc=$?
        [[ $rc -eq 124 ]] && echo "      Reached ${DURATION}s limit." || exit $rc
    }
else
    ros2 bag play "$BAG_DIR" --clock
fi
# bag play exits when the bag ends (or duration elapses)

echo "      Bag finished. Waiting 3s for algorithm to flush …"
sleep 3
kill "$ALGO_PID" 2>/dev/null || true
wait "$ALGO_PID" 2>/dev/null || true
echo "      Algorithm stopped."

# ── check output ─────────────────────────────────────────────────────────────
if [[ ! -s "$TUM_LIVE" ]]; then
    echo "" >&2
    echo "ERROR: no TUM written at ${TUM_LIVE}" >&2
    echo "       Check that odom_to_tum.enabled=true in config/lio.yaml" >&2
    exit 1
fi
cp "$TUM_LIVE" "$RESULT_TUM"
echo "      Odometry saved → ${RESULT_TUM}"

# ── optional frame correction ────────────────────────────────────────────────
if ! $NO_FRAME_CORRECT; then
    echo ""
    echo "      Correcting frame: lidar → base_link (sensor: ${SENSOR}) …"
    # Remove stale backup so iilabs3d never prompts for overwrite
    rm -f "${RESULT_TUM%.tum}.orig.tum"
    iilabs3d correct-frame "$RESULT_TUM" lidar --sensor "${SENSOR//-/_}" 2>/dev/null || {
        echo "      (skipped — already in base_link, or sensor name not recognised)"
    }
fi

# ── evaluate ─────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Evaluating …"
echo ""
iilabs3d eval "$GT_TUM" "$RESULT_TUM" || \
    echo "      (eval warning — trajectory may be too short for 10m RTE intervals)"

# ── trajectory plot ──────────────────────────────────────────────────────────
echo ""
echo "[4/4] Plotting trajectory …"
DURATION_ARG=""
[[ "$DURATION" -gt 0 ]] && DURATION_ARG="--duration ${DURATION}"

# Collect all result files that exist for this sequence
EST_FILES=()
for f in "${SEQ_DIR}/result_regnonrep_imu.tum" "${SEQ_DIR}/result_regnonrep_noimu.tum"; do
    [[ -f "$f" ]] && EST_FILES+=("$f")
done
# Fall back to the current result if named differently
[[ ${#EST_FILES[@]} -eq 0 && -f "$RESULT_TUM" ]] && EST_FILES+=("$RESULT_TUM")

python3 "$PLOT_SCRIPT" "$GT_TUM" "${EST_FILES[@]}" "$PLOT_PNG" $DURATION_ARG || \
    echo "      (plot failed — check matplotlib / numpy)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done: ${SEQUENCE} / ${SENSOR}"
[[ -f "$PLOT_PNG" ]] && echo "  plot : ${PLOT_PNG}"
echo "════════════════════════════════════════════════════════════"
