#!/usr/bin/env bash
# run_one.sh — launch a regnonrep node on the indoor1_avia bag (full pass) and
# RELIABLY stop it afterwards.
#
# Why this works (and the old version didn't):
#   * `ros2 run <pkg> <exe>` SPAWNS a child node process — killing the wrapper
#     PID orphans the node, which then keeps consuming the next bag play and
#     corrupts later results.  So we launch the INSTALLED SCRIPT DIRECTLY with
#     python, making $! the real node PID.
#   * `pkill` and any `-9` (SIGKILL) are blocked by the agent harness and abort
#     the whole command — so we never use them; we kill the captured PID with
#     SIGINT then SIGTERM, and verify the process is gone.
#
# Usage:
#   run_one.sh <installed_node_script> <node_node_name> <params_yaml> \
#              [extra "-p k:=v" args] [odom_topic] [out_tum]
set -o pipefail
WS=/u/97/habibip1/unix/ros2_ws
BAG=/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/Tier/Livox_avia/indoor1_avia/indoor1_avia
RATE=0.3
LIBDIR="${WS}/install/regnonrep/lib/regnonrep"
RUNDIR="${WS}/src/regnonrep/tum/run3"

NODE_SCRIPT="$1"        # e.g. ros_lio.py  (resolved under install lib dir)
NODE_NAME="$2"          # e.g. lio_node
PARAMS="$3"             # params yaml
EXTRA="${4:-}"          # extra ros args, e.g. "-p debug_csv:=/path.csv"
ODOM_TOPIC="${5:-}"     # optional: record this topic to TUM
OUT_TUM="${6:-}"

set +u; source "${WS}/install/setup.bash" >/dev/null 2>&1; set -u

# ---- stop helper: SIGINT, then SIGTERM, then report (never SIGKILL / pkill) --
stop_pid() {
    local pid="$1"
    [[ -z "$pid" ]] && return 0
    kill -INT  "$pid" 2>/dev/null; sleep 3
    kill -TERM "$pid" 2>/dev/null; sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        echo "  WARNING: pid $pid still alive after INT+TERM" >&2
    else
        echo "  pid $pid stopped cleanly"
    fi
}

# ---- launch node DIRECTLY so $! is the node (not a ros2 wrapper child) -------
# shellcheck disable=SC2086
python3.10 "${LIBDIR}/${NODE_SCRIPT}" --ros-args --params-file "$PARAMS" \
    -r __node:="$NODE_NAME" $EXTRA > "${RUNDIR}/${NODE_NAME}.log" 2>&1 &
NODE=$!
echo "[run_one] node ${NODE_SCRIPT} pid=${NODE}"

REC=""
if [[ -n "$ODOM_TOPIC" && -n "$OUT_TUM" ]]; then
    rm -f "$OUT_TUM"
    python3.10 "${LIBDIR}/odom_to_tum.py" --ros-args \
        -p odom_to_tum.enabled:=true -p odom_to_tum.odom_topic:="$ODOM_TOPIC" \
        -p odom_to_tum.output_path:="$OUT_TUM" -p odom_to_tum.use_msg_time:=true \
        -p odom_to_tum.flush_every_n:=1 -p odom_to_tum.append:=false \
        > "${RUNDIR}/recorder.log" 2>&1 &
    REC=$!
    echo "[run_one] recorder pid=${REC} -> ${OUT_TUM}"
fi

sleep 3
echo "[run_one] playing bag full @${RATE} ($(date +%T)) ..."
ros2 bag play "$BAG" --clock --rate "$RATE" > "${RUNDIR}/play.log" 2>&1
echo "[run_one] bag done ($(date +%T)); flushing ..."; sleep 8

echo "[run_one] stopping node ..."; stop_pid "$NODE"
[[ -n "$REC" ]] && { echo "[run_one] stopping recorder ..."; stop_pid "$REC"; }
echo "[run_one] done."
