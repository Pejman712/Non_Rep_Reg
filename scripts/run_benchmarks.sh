#!/usr/bin/env bash
# run_benchmarks.sh — consolidated benchmark runner for the lio_base.py variants.
#
# Runs each selected METHOD (a lio_base.py variant) over the full dataset matrix,
# one pass per sequence (the variants are always IMU-fused — no use_imu toggle),
# then evaluates (evo_ape / iilabs3d) and plots the route per (method, sequence).
#
# Method results are written into each sequence folder as:
#     result_<method>.tum            (e.g. result_nonrep_gicp.tum)
#     trajectory_<method>.png
# and every route plot is also collected at the end into:
#     benchmark_results/<timestamp>/plots/<method>__<dataset>_<sequence>.png
#
# Dataset matrix (lidar topic / imu topic come from the YAMLs):
#   Tier / Livox Avia    indoor1/2/3_avia       (config/lio_tier_avia.yaml)
#   Tier / Livox Horizon indoor1/2/3_horizen    (config/lio_tier_horizen.yaml)
#   iilab / livox_mid-360 nav_a_diff nav_a_omni loop slippage  (config/lio.yaml)
# NOTE: variants bind only lidar_topic/imu_topic from these YAMLs; extrinsics and
#       IMU tuning fall back to lio_base defaults (Avia-tuned).
#
# Usage:
#   ./run_benchmarks.sh [options]
# Options:
#   --methods a,b,c        comma list of method names to run (default: all 13)
#   --list-methods         print the method names and exit
#   --skip-tier-avia       skip the Tier/Avia sequences
#   --skip-tier-horizen    skip the Tier/Horizon sequences
#   --skip-iilab           skip the iilab sequences
#   --duration N           play only first N seconds of each bag (0 = full) [default 0]
#   --dry-run              print what would run without launching anything
#
# After editing config/*.yaml rebuild:  colcon build --packages-select regnonrep
set -uo pipefail

# Unbuffered node stdout so post-bag shutdown output (e.g. nrlio's loop-closure
# back-end: keyframes / loop closures / [timing] / accumulation depth) is flushed
# to the per-run log before the node is killed, instead of dying in the buffer.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
WS="$(realpath "${SCRIPT_DIR}/../../..")"
TUM_LIVE="${WS}/src/regnonrep/tum/lio_odom.tum"
MAP_LIVE="${WS}/src/regnonrep/tum/lio_map.pcd"   # map_saver checkpoint (opt-in --save-maps)
PLOT_SCRIPT="${SCRIPT_DIR}/plot_tum.py"
LOG_DIR="${SCRIPT_DIR}/../benchmark_results/$(date +%Y%m%d_%H%M%S)"

ROOT_TIER="/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/Tier"
ROOT_IILAB="/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/iilab_benchmark"
ROOT_CERN="/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/CERN"

# ── methods: "name|executable" (all subclass lio_base.SuperLioBase) ───────────
# Method names renamed to describe the method: <registration>[_<fusion>][_<degeneracy-strategy>].
# (All gicp/ndt variants also use the non-repetitive init-guess predictor; the
#  script files keep their original lio_*.py names.)  Old→new for reference:
#   p2p→p2plane_base  nonrep_gicp→gicp  nonrep_ndt→ndt
#   nonrep_gicp_p2p→gicp_p2plane  fused_gated→gicp_p2plane_gated
#   fused_gated_ndt→ndt_p2plane_gated  fused_gated_p2p_gicp_ndt→gicp_ndt_p2plane_gated
#   nonrep_fused_degen→gicp_p2plane_fused_degen  nonrep_gicp_p2p_degen→gicp_p2plane_degen_gyro
#   degen_reuse→gicp_p2plane_degen_reuse  tsvd→gicp_degen_tsvd  pgo→gicp_p2plane_pgo
#   gyro_gicp_p2p→gyro_gicp_p2plane  gyro_slowrot→gyro_gicp_slowrot
ALL_METHODS=(
    "p2plane_base|lio_p2p.py"
    "gicp|lio_nonrep_gicp.py"
    "ndt|lio_nonrep_ndt.py"
    "ndt_p2plane_gated|lio_fused_gated_ndt.py"
    "gicp_ndt_p2plane_gated|lio_fused_gated_p2p_gicp_ndt.py"
    "gicp_p2plane|lio_nonrep_gicp_p2p.py"
    "gicp_p2plane_degen_gyro|lio_nonrep_gicp_p2p_degen.py"
    "gicp_p2plane_fused_degen|lio_nonrep_fused_degen.py"
    "gicp_p2plane_gated|lio_fused_gated.py"
    "gyro_gicp|lio_gyro_gicp.py"
    "gyro_gicp_degen|lio_gyro_gicp_degen.py"
    "gyro_gicp_p2plane|lio_gyro_gicp_p2p.py"
    "gyro_gicp_slowrot|lio_gyro_slowrot.py"
    "gicp_p2plane_degen_reuse|lio_degen_reuse.py"
    "gicp_degen_tsvd|lio_tsvd.py"
    "gicp_p2plane_pgo|lio_pgo.py"
    "gen_lio|lio_gen_lio.py"
    "gen_liotier1|lio_gen_lio_tier1.py"
    "gen_liotier2|lio_gen_lio_tier2.py"
    "gen_liotier3|lio_gen_lio_tier3.py"
    "gen_liotier4|lio_gen_lio_tier4.py"
    "gen_lio_intensity|lio_gen_lio_intensity.py"
    "nrlio|lio_nrlio.py"
    "nrlio_optimized|lio_nrlio_optimized.py"
    "nrlio_optA|lio_nrlio_optA.py"
    "nrlio_optB|lio_nrlio_optB.py"
    "nrlio_op_den|lio_nrlio_op_den.py"
    "nrlio_plus|lio_nrlio_plus.py"
)

# ── external LIO packages (run via `ros2 run`, fed the converter's
#    /velodyne_points, odom remapped to /bench/odom and recorded). ────────────
# faster_lio removed — heap-corruption (malloc invalid size) crash in its ROS2 core.
ALL_EXTERNAL=(dlio fast_lio ig_lio point_lio super_lio)
SRC="${WS}/src"

is_external() { local x; for x in "${ALL_EXTERNAL[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }

# ext_spec <method> <dataset> -> sets EXT_PKG EXT_EXE EXT_CFG EXT_ODOM
#   EXT_RAW_IMU EXT_TFIELD EXT_PARAMS[] EXT_REMAPS[]
ext_spec() {
    local m="$1" ds="$2"
    case "$ds" in
        tier_avia)    EXT_RAW_IMU=/avia/livox/imu; EXT_TFIELD=offset_time ;;
        tier_horizen) EXT_RAW_IMU=/livox/imu;      EXT_TFIELD=offset_time ;;
        iilab)        EXT_RAW_IMU=/eve/imu/data;   EXT_TFIELD=timestamp ;;
    esac
    EXT_ODOM=/bench/odom
    EXT_PARAMS=(); EXT_REMAPS=(); EXT_IMU_REMAP=0
    case "$m" in
        dlio)
            EXT_PKG=direct_lidar_inertial_odometry; EXT_EXE=dlio_odom_node
            [[ "$ds" == iilab ]] && EXT_CFG="$SRC/dlio/cfg/dlio_iilab.yaml" \
                                 || EXT_CFG="$SRC/dlio/cfg/dlio_tier.yaml"
            EXT_PARAMS=("$SRC/dlio/cfg/params.yaml")
            # dlio reads IMU via the "imu" remap; the imu:= entry is appended in
            # run_one_external once the (possibly rescaled) effective topic is known.
            EXT_REMAPS=("pointcloud:=/velodyne_points" "odom:=${EXT_ODOM}"); EXT_IMU_REMAP=1 ;;
        fast_lio)
            EXT_PKG=fast_lio; EXT_EXE=fastlio_mapping; EXT_REMAPS=("/Odometry:=${EXT_ODOM}")
            case "$ds" in
                tier_avia)    EXT_CFG="$SRC/fast_lio2/config/avia.yaml" ;;
                tier_horizen) EXT_CFG="$SRC/fast_lio2/config/horizon.yaml" ;;
                iilab)        EXT_CFG="$SRC/fast_lio2/config/iilab_mid360.yaml" ;;
            esac ;;
        faster_lio)
            EXT_PKG=faster_lio; EXT_EXE=run_mapping_online; EXT_REMAPS=("/Odometry:=${EXT_ODOM}")
            case "$ds" in
                tier_avia)    EXT_CFG="$SRC/faster_lio/faster-lio/config/tier_avia.yaml" ;;
                tier_horizen) EXT_CFG="$SRC/faster_lio/faster-lio/config/tier_horizon.yaml" ;;
                iilab)        EXT_CFG="$SRC/faster_lio/faster-lio/config/iilab_mid360.yaml" ;;
            esac ;;
        ig_lio)
            EXT_PKG=ig_lio; EXT_EXE=ig_lio_node; EXT_REMAPS=("/lio/odometry:=${EXT_ODOM}")
            case "$ds" in
                tier_avia)    EXT_CFG="$SRC/ig_lio/config/tier_avia.yaml" ;;
                tier_horizen) EXT_CFG="$SRC/ig_lio/config/tier_horizon.yaml" ;;
                iilab)        EXT_CFG="$SRC/ig_lio/config/iilab_mid360.yaml" ;;
            esac ;;
        point_lio)
            EXT_PKG=point_lio; EXT_EXE=pointlio_mapping; EXT_REMAPS=("/aft_mapped_to_init:=${EXT_ODOM}")
            case "$ds" in
                tier_avia)    EXT_CFG="$SRC/point_lio_ros2/config/avia.yaml" ;;
                tier_horizen) EXT_CFG="$SRC/point_lio_ros2/config/horizon.yaml" ;;
                iilab)        EXT_CFG="$SRC/point_lio_ros2/config/mid360.yaml" ;;
            esac ;;
        super_lio)
            EXT_PKG=super_lio; EXT_EXE=super_lio_node; EXT_REMAPS=("/lio/odom:=${EXT_ODOM}")
            case "$ds" in
                tier_avia)    EXT_CFG="$SRC/super_lio/config/livox_avia.yaml" ;;
                tier_horizen) EXT_CFG="$SRC/super_lio/config/livox_horizon.yaml" ;;
                iilab)        EXT_CFG="$SRC/super_lio/config/iilab_mid360.yaml" ;;
            esac ;;
    esac
}

# ── options ───────────────────────────────────────────────────────────────────
RUN_TIER_AVIA=true
RUN_TIER_HORIZEN=true
RUN_IILAB=true
RUN_CERN=true
DURATION=0
DRY_RUN=false
METHOD_FILTER=""
SEQ_FILTER=""
BAG_RATE_OVERRIDE=""
WAIT_OVERRIDE=""
START_OFFSET=5
PREFILTER=""          # ""=use config default; on/off to override (regnonrep variants)
PREFILTER_VOXEL=""
PREFILTER_RANGE=""
PREFILTER_ROR=""
PREFILTER_ROR_TAU=""
PREFILTER_SOR=""
SAVE_MAPS="off"       # on = dump the accumulated LIO map to .pcd (regnonrep variants) for MapEval
PARAMS_OVERLAY=""     # optional YAML overlay (param-sweep campaigns); overrides cfg

# ── optional CPU pinning / priority (empty ⇒ unchanged behaviour) ──────────────
# The livox→velodyne converter is single-threaded Python and is the throughput
# bottleneck; giving it a dedicated core (and the LIO node the rest) keeps it from
# being starved.  Set via environment, e.g.:
#   BENCH_CONV_CPU=0      pin the converter to core 0
#   BENCH_NODE_CPU=1-11   pin the LIO node to cores 1..11
#   BENCH_NICE=-5         renice the LIO node (negative needs privilege/sudo)
CONV_PREFIX=(); NODE_PREFIX=()
[[ -n "${BENCH_CONV_CPU:-}" ]] && CONV_PREFIX=(taskset -c "$BENCH_CONV_CPU")
[[ -n "${BENCH_NODE_CPU:-}" ]] && NODE_PREFIX=(taskset -c "$BENCH_NODE_CPU")
[[ -n "${BENCH_NICE:-}"     ]] && NODE_PREFIX+=(nice -n "$BENCH_NICE")

for arg in "$@"; do
    case $arg in
        --methods=*)         METHOD_FILTER="${arg#*=}" ;;
        --sequences=*)       SEQ_FILTER="${arg#*=}" ;;
        --list-methods)      for m in "${ALL_METHODS[@]}"; do echo "${m%%|*}"; done
                             for m in "${ALL_EXTERNAL[@]}"; do echo "$m"; done; exit 0 ;;
        --skip-tier-avia)    RUN_TIER_AVIA=false ;;
        --skip-tier-horizen) RUN_TIER_HORIZEN=false ;;
        --skip-iilab)        RUN_IILAB=false ;;
        --skip-cern)         RUN_CERN=false ;;
        --duration=*)        DURATION="${arg#*=}" ;;
        --bag-rate=*)        BAG_RATE_OVERRIDE="${arg#*=}" ;;
        --post-wait=*)       WAIT_OVERRIDE="${arg#*=}" ;;
        --start-offset=*)    START_OFFSET="${arg#*=}" ;;
        --prefilter=*)       PREFILTER="${arg#*=}" ;;
        --prefilter-voxel=*) PREFILTER_VOXEL="${arg#*=}" ;;
        --prefilter-range=*) PREFILTER_RANGE="${arg#*=}" ;;
        --prefilter-ror=*)   PREFILTER_ROR="${arg#*=}" ;;
        --prefilter-ror-tau=*) PREFILTER_ROR_TAU="${arg#*=}" ;;
        --prefilter-sor=*)   PREFILTER_SOR="${arg#*=}" ;;
        --save-maps=*)       SAVE_MAPS="${arg#*=}" ;;
        --params-overlay=*)  PARAMS_OVERLAY="${arg#*=}" ;;
        --dry-run)           DRY_RUN=true ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# select methods.  Each entry is "name|exe" for lio_base variants, or
# "name|__EXT__" for an external package (dispatched to run_one_external).
METHODS=()
if [[ -n "$METHOD_FILTER" ]]; then
    IFS=',' read -ra want <<< "$METHOD_FILTER"
    for w in "${want[@]}"; do
        found=false
        for m in "${ALL_METHODS[@]}"; do
            [[ "${m%%|*}" == "$w" ]] && { METHODS+=("$m"); found=true; break; }
        done
        if ! $found && is_external "$w"; then METHODS+=("$w|__EXT__"); found=true; fi
        $found || { echo "Unknown method: $w (see --list-methods)" >&2; exit 1; }
    done
else
    METHODS=("${ALL_METHODS[@]}")
    for x in "${ALL_EXTERNAL[@]}"; do METHODS+=("$x|__EXT__"); done
fi

# ── sequence matrix: "dataset|sensor|sequence" ────────────────────────────────
RUNS=()
$RUN_TIER_AVIA    && RUNS+=("tier_avia|Livox_avia|indoor1_avia" \
                            "tier_avia|Livox_avia|indoor2_avia" \
                            "tier_avia|Livox_avia|indoor3_avia" \
                            "tier_avia|Livox_avia|indoor6_avia")
$RUN_TIER_HORIZEN && RUNS+=("tier_horizen|Livox_horizen|indoor1_horizen" \
                            "tier_horizen|Livox_horizen|indoor2_horizen" \
                            "tier_horizen|Livox_horizen|indoor3_horizen" \
                            "tier_horizen|Livox_horizen|indoor6_horizen")
$RUN_IILAB        && RUNS+=("iilab|livox_mid-360|nav_a_diff" \
                            "iilab|livox_mid-360|nav_a_omni" \
                            "iilab|livox_mid-360|loop" \
                            "iilab|livox_mid-360|slippage")
$RUN_CERN         && RUNS+=("cern|unitree_unilidar_L1|BA6" \
                            "cern|unitree_unilidar_L1|BA51" \
                            "cern|unitree_unilidar_L1|BA52" \
                            "cern|unitree_unilidar_L1|BA801" \
                            "cern|unitree_unilidar_L1|BA802" \
                            "cern|unitree_unilidar_L1|BA803" \
                            "cern|unitree_unilidar_L1|927full" \
                            "cern|unitree_unilidar_L1|charm" \
                            "cern|unitree_unilidar_L1|Dumparea")

# filter by sequence name if requested (comma list)
if [[ -n "$SEQ_FILTER" ]]; then
    IFS=',' read -ra want_seq <<< "$SEQ_FILTER"
    FILTERED=()
    for entry in "${RUNS[@]}"; do
        eseq="${entry##*|}"
        for w in "${want_seq[@]}"; do
            [[ "$eseq" == "$w" ]] && { FILTERED+=("$entry"); break; }
        done
    done
    RUNS=("${FILTERED[@]}")
fi

# ── per-dataset settings ──────────────────────────────────────────────────────
configure_dataset() {
    local ds="$1"
    local cfgdir="${WS}/install/regnonrep/share/regnonrep/config"
    case "$ds" in
        tier_avia)
            DATASET_ROOT="$ROOT_TIER" ; INSTALLED_CFG="${cfgdir}/lio_tier_avia.yaml"
            BAG_RATE=0.3 ; POST_BAG_WAIT=30 ; FRAME_CORRECT=false ; ALIGN_TS=true ; EVAL_KIND=evo
            IMU_SCALE=9.81 ;;                     # Avia built-in IMU reports accel in g
        tier_horizen)
            DATASET_ROOT="$ROOT_TIER" ; INSTALLED_CFG="${cfgdir}/lio_tier_horizen.yaml"
            BAG_RATE=0.3 ; POST_BAG_WAIT=30 ; FRAME_CORRECT=false ; ALIGN_TS=true ; EVAL_KIND=evo
            IMU_SCALE=9.81 ;;                     # Horizon built-in IMU reports accel in g
        iilab)
            DATASET_ROOT="$ROOT_IILAB" ; INSTALLED_CFG="${cfgdir}/lio.yaml"
            BAG_RATE=0.8 ; POST_BAG_WAIT=3 ; FRAME_CORRECT=true ; ALIGN_TS=false ; EVAL_KIND=iilabs3d
            IMU_SCALE=1.0 ;;                      # Xsens MTi-630 already m/s²
        cern)
            DATASET_ROOT="$ROOT_CERN" ; INSTALLED_CFG="${cfgdir}/lio_cern.yaml"
            BAG_RATE=1.0 ; POST_BAG_WAIT=5 ; FRAME_CORRECT=false ; ALIGN_TS=true ; EVAL_KIND=evo
            IMU_SCALE=1.0 ;;                      # Unitree Unilidar L1 IMU already m/s²
        *) echo "Unknown dataset: $ds" >&2; return 1 ;;
    esac
    [[ -n "$BAG_RATE_OVERRIDE" ]] && BAG_RATE="$BAG_RATE_OVERRIDE"
    [[ -n "$WAIT_OVERRIDE" ]] && POST_BAG_WAIT="$WAIT_OVERRIDE"
    return 0
}

align_timestamps() {  # $1 = gt.tum  $2 = result.tum  $3 = start-offset [s]
    python3 - "$1" "$2" "${3:-0}" <<'PYEOF'
import sys
gt_t0   = float(open(sys.argv[1]).readline().split()[0])
lines   = open(sys.argv[2]).readlines()
odom_t0 = float(lines[0].split()[0])
start_offset = float(sys.argv[3])
# The bag is played with `--start-offset start_offset`, so the estimate's first
# pose corresponds to GT at (gt_t0 + start_offset), NOT gt_t0.  Omitting this term
# mispairs every pose by `start_offset` seconds in evo's --t_max_diff matching.
offset  = gt_t0 - odom_t0 + start_offset
out = []
for line in lines:
    p = line.split()
    p[0] = "{:.9f}".format(float(p[0]) + offset)
    out.append(" ".join(p))
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("      ts-align offset={:.3f}s (incl start-offset {:.1f}s)".format(offset, start_offset))
PYEOF
}

# extra standard trajectory metrics for the table (evo path only):
#   ate_se3  = SE(3)-aligned ATE RMSE  [m]  (full Umeyama — the published-comparison
#              standard; note: Umeyama alignment, unlike the raw origin-aligned RMSE)
#   are_deg  = rotation ATE RMSE       [deg] (origin-aligned)
#   rpe_1m   = translation RPE RMSE per 1 m  [m]  (relative drift, alignment-light)
extra_metrics() {  # $1 = gt.tum  $2 = result.tum
    local se3 are rpe
    se3=$(evo_ape tum "$1" "$2" --align --t_max_diff 0.05 2>/dev/null \
          | awk '/^ *rmse/{print $2; exit}')
    are=$(evo_ape tum "$1" "$2" --align_origin --pose_relation angle_deg --t_max_diff 0.05 2>/dev/null \
          | awk '/^ *rmse/{print $2; exit}')
    rpe=$(evo_rpe tum "$1" "$2" --align_origin --delta 1 --delta_unit m --pose_relation trans_part --t_max_diff 0.05 2>/dev/null \
          | awk '/^ *rmse/{print $2; exit}')
    echo "  [metrics] ate_se3=${se3:-nan} are_deg=${are:-nan} rpe_1m=${rpe:-nan}"
}

# ── run + evaluate + plot one (method, sequence) ──────────────────────────────
run_one() {  # $1 method_name  $2 executable  $3 dataset  $4 sensor  $5 sequence
    local method="$1" exe="$2" ds="$3" sensor="$4" seq="$5"
    configure_dataset "$ds" || return 1

    local seq_dir="${DATASET_ROOT}/${sensor}/${seq}"
    local bag_dir="${seq_dir}/${seq}"
    local gt_tum="${seq_dir}/${seq}.tum"
    local result_tum="${seq_dir}/result_${method}.tum"
    local plot_png="${seq_dir}/trajectory_${method}.png"

    echo "════════════════════════════════════════════════════════════"
    echo "  method=${method}  exe=${exe}"
    echo "  ${ds} / ${sensor} / ${seq}"
    echo "  bag: ${bag_dir}"
    echo "  gt : ${gt_tum}$( [[ -f "$gt_tum" ]] || echo '  (missing — eval skipped)')"
    echo "════════════════════════════════════════════════════════════"

    if [[ ! -d "$bag_dir" ]]; then echo "  ERROR: bag dir not found: ${bag_dir}" >&2; return 1; fi
    if $DRY_RUN; then echo "  DRY-RUN: would launch ${exe} + play + eval + plot"; return 0; fi

    pkill -9 -f "lib/regnonrep/" 2>/dev/null || true
    pkill -9 -f "ros2 bag play" 2>/dev/null || true
    pkill -9 -f "ros2 launch regnonrep" 2>/dev/null || true
    sleep 1
    rm -f "$TUM_LIVE" "${TUM_LIVE%.tum}.ann.csv" "${TUM_LIVE%.tum}.proc.csv" "$MAP_LIVE"

    echo "  launching ${exe} …"
    local largs=("exe:=${exe}" "cfg:=${INSTALLED_CFG}")
    [[ -n "$PREFILTER" ]]         && largs+=("prefilter:=${PREFILTER}")
    [[ -n "$PREFILTER_VOXEL" ]]   && largs+=("prefilter_voxel:=${PREFILTER_VOXEL}")
    [[ -n "$PREFILTER_RANGE" ]]   && largs+=("prefilter_range:=${PREFILTER_RANGE}")
    [[ -n "$PREFILTER_ROR" ]]     && largs+=("prefilter_ror:=${PREFILTER_ROR}")
    [[ -n "$PREFILTER_ROR_TAU" ]] && largs+=("prefilter_ror_tau:=${PREFILTER_ROR_TAU}")
    [[ -n "$PREFILTER_SOR" ]]     && largs+=("prefilter_sor:=${PREFILTER_SOR}")
    [[ "$SAVE_MAPS" == "on" ]]    && largs+=("save_maps:=on" "map_out:=${MAP_LIVE}")
    [[ -n "$PARAMS_OVERLAY" ]]  && largs+=("params_overlay:=${PARAMS_OVERLAY}")
    ros2 launch regnonrep lio_variant.launch.py "${largs[@]}" &
    local algo_pid=$!
    sleep 3

    echo "  playing bag at ${BAG_RATE}x (skip first ${START_OFFSET}s) …"
    if [[ "$DURATION" -gt 0 ]]; then
        timeout "$DURATION" ros2 bag play "$bag_dir" --clock --rate "$BAG_RATE" --start-offset "$START_OFFSET" || {
            rc=$?; [[ $rc -eq 124 ]] && echo "  reached ${DURATION}s limit." || { kill "$algo_pid" 2>/dev/null; return $rc; }
        }
    else
        ros2 bag play "$bag_dir" --clock --rate "$BAG_RATE" --start-offset "$START_OFFSET"
    fi

    echo "  bag done, waiting ${POST_BAG_WAIT}s for flush …"
    sleep "$POST_BAG_WAIT"
    kill "$algo_pid" 2>/dev/null || true
    wait "$algo_pid" 2>/dev/null || true

    if [[ ! -s "$TUM_LIVE" ]]; then echo "  ERROR: no TUM written (${method}/${seq})" >&2; return 1; fi
    cp "$TUM_LIVE" "$result_tum"
    # per-pose mechanism annotations (nrlio family only) → alongside the TUM
    local result_ann="${result_tum%.tum}.ann.csv"
    rm -f "$result_ann"
    [[ -f "${TUM_LIVE%.tum}.ann.csv" ]] && cp "${TUM_LIVE%.tum}.ann.csv" "$result_ann"
    # per-scan processing time (every regnonrep variant) → alongside the TUM
    local result_proc="${result_tum%.tum}.proc.csv"
    rm -f "$result_proc"
    [[ -f "${TUM_LIVE%.tum}.proc.csv" ]] && cp "${TUM_LIVE%.tum}.proc.csv" "$result_proc"
    # accumulated LIO map (opt-in --save-maps) → alongside the TUM, for MapEval/MME
    if [[ "$SAVE_MAPS" == "on" ]]; then
        local result_map="${result_tum%.tum}.pcd"
        rm -f "$result_map"
        [[ -s "$MAP_LIVE" ]] && cp "$MAP_LIVE" "$result_map" \
            && echo "  saved map → $(basename "$result_map")"
    fi
    echo "  saved → $(basename "$result_tum")"

    if $ALIGN_TS && [[ -f "$gt_tum" ]]; then align_timestamps "$gt_tum" "$result_tum" "$START_OFFSET"; fi
    if $FRAME_CORRECT; then
        echo "  frame-correct lidar → base_link (${sensor}) …"
        rm -f "${result_tum%.tum}.orig.tum"
        iilabs3d correct-frame "$result_tum" lidar --sensor "${sensor//-/_}" 2>/dev/null \
            || echo "  (frame-correct skipped)"
    fi

    echo "  [eval] $(basename "$result_tum") vs GT …"
    if [[ -f "$gt_tum" ]]; then
        if [[ "$EVAL_KIND" == evo ]]; then
            evo_ape tum "$gt_tum" "$result_tum" --align_origin --t_max_diff 0.05 --verbose 2>&1 \
                || echo "      (evo_ape warning)"
            extra_metrics "$gt_tum" "$result_tum"
        else
            iilabs3d eval "$gt_tum" "$result_tum" || echo "      (iilabs3d eval warning)"
        fi
    else
        echo "      (skipped — no ground truth)"
    fi

    echo "  [plot] route …"
    local gt_arg=() ; [[ -f "$gt_tum" ]] && gt_arg=(--gt "$gt_tum")
    if [[ -f "${result_ann:-}" ]]; then
        # nrlio family: annotated route (mode colours + ZUPT/degeneracy/clamp icons + strips)
        python3 "${SCRIPT_DIR}/plot_annotated.py" --no-show "${gt_arg[@]}" \
            --tum "$result_tum" --ann "$result_ann" --out "$plot_png" \
            || python3 "$PLOT_SCRIPT" --no-show "${gt_arg[@]}" "$result_tum" "$plot_png" \
            || echo "      (plot failed)"
    else
        python3 "$PLOT_SCRIPT" --no-show "${gt_arg[@]}" "$result_tum" "$plot_png" || echo "      (plot failed)"
    fi
    return 0
}

# ── run + evaluate + plot one (external method, sequence) ─────────────────────
run_one_external() {  # $1 method  $2 dataset  $3 sensor  $4 sequence
    local method="$1" ds="$2" sensor="$3" seq="$4"
    configure_dataset "$ds" || return 1
    ext_spec "$method" "$ds"

    local seq_dir="${DATASET_ROOT}/${sensor}/${seq}"
    local bag_dir="${seq_dir}/${seq}"
    local gt_tum="${seq_dir}/${seq}.tum"
    local result_tum="${seq_dir}/result_${method}.tum"
    local plot_png="${seq_dir}/trajectory_${method}.png"

    echo "════════════════════════════════════════════════════════════"
    echo "  method=${method} (external)  pkg=${EXT_PKG}  exe=${EXT_EXE}"
    echo "  cfg=${EXT_CFG}"
    echo "  ${ds} / ${sensor} / ${seq}  | odom=${EXT_ODOM} imu=${EXT_RAW_IMU}"
    echo "════════════════════════════════════════════════════════════"
    if [[ ! -d "$bag_dir" ]]; then echo "  ERROR: bag dir not found: ${bag_dir}" >&2; return 1; fi
    if [[ ! -f "$EXT_CFG" ]]; then echo "  ERROR: config not found: ${EXT_CFG}" >&2; return 1; fi
    if $DRY_RUN; then echo "  DRY-RUN: converter + ${EXT_PKG}/${EXT_EXE} + record ${EXT_ODOM}"; return 0; fi

    pkill -9 -f "lib/regnonrep/" 2>/dev/null || true
    pkill -9 -f "ros2 bag play" 2>/dev/null || true
    pkill -9 -f "lib/${EXT_PKG}/${EXT_EXE}" 2>/dev/null || true
    sleep 1
    rm -f "$TUM_LIVE"

    local raw_lidar
    case "$ds" in
        tier_avia) raw_lidar=/avia/livox/points ;;
        tier_horizen) raw_lidar=/livox/points ;;
        iilab) raw_lidar=/eve/lidar3d ;;
    esac
    echo "  starting livox→velodyne converter, recorder, ${EXT_EXE} …"
    "${CONV_PREFIX[@]}" ros2 run regnonrep livox_to_velodyne.py --ros-args \
        -p input_topic:="$raw_lidar" -p output_topic:=/velodyne_points \
        -p time_field:="$EXT_TFIELD" &
    local conv_pid=$!
    ros2 run regnonrep odom_to_tum.py --ros-args \
        -p odom_to_tum.enabled:=true -p odom_to_tum.odom_topic:="$EXT_ODOM" \
        -p odom_to_tum.output_path:="$TUM_LIVE" -p odom_to_tum.use_msg_time:=true \
        -p odom_to_tum.append:=false &
    local rec_pid=$!

    # IMU rescale (g→m/s²) for datasets whose built-in IMU reports accel in g (Tier).
    # Config-based nodes read the effective topic from their YAML (imu_topic:/bench/imu);
    # dlio gets it via the imu:= remap appended below.
    local eff_imu="$EXT_RAW_IMU" imu_pid=""
    if [[ "$IMU_SCALE" != "1.0" && "$IMU_SCALE" != "1" ]]; then
        eff_imu=/bench/imu
        echo "  IMU rescale ${EXT_RAW_IMU} → ${eff_imu} (accel ×${IMU_SCALE}, g→m/s²) …"
        "${CONV_PREFIX[@]}" ros2 run regnonrep imu_rescale.py --ros-args \
            -p input_topic:="$EXT_RAW_IMU" -p output_topic:="$eff_imu" \
            -p accel_scale:="$IMU_SCALE" &
        imu_pid=$!
    fi
    [[ "${EXT_IMU_REMAP:-0}" == 1 ]] && EXT_REMAPS+=("imu:=${eff_imu}")

    local pf=(--params-file "$EXT_CFG")
    local p; for p in "${EXT_PARAMS[@]}"; do pf+=(--params-file "$p"); done
    local rm=(); local r; for r in "${EXT_REMAPS[@]}"; do rm+=(-r "$r"); done
    # NOTE: do NOT set use_sim_time — these nodes process by message stamp; with
    # sim time, when a node lags playback its now() races ahead and it drops
    # "too old" scans, diverging the estimate. Bag is played with --clock anyway.
    "${NODE_PREFIX[@]}" ros2 run "$EXT_PKG" "$EXT_EXE" --ros-args "${pf[@]}" "${rm[@]}" \
        > "${LOG_DIR}/${method}__${ds}_${seq}.node.log" 2>&1 &
    local node_pid=$!
    sleep 6

    echo "  playing bag at ${BAG_RATE}x (skip first ${START_OFFSET}s) …"
    if [[ "$DURATION" -gt 0 ]]; then
        timeout "$DURATION" ros2 bag play "$bag_dir" --clock --rate "$BAG_RATE" --start-offset "$START_OFFSET" || true
    else
        ros2 bag play "$bag_dir" --clock --rate "$BAG_RATE" --start-offset "$START_OFFSET"
    fi
    echo "  bag done, waiting ${POST_BAG_WAIT}s for flush …"
    sleep "$POST_BAG_WAIT"

    # ── actual per-scan compute time for the external node ────────────────────
    # CPU time (utime+stime) / poses = mean processing ms per scan, independent of
    # the bag playback rate (this is real work done, not wall/queue time).  Read
    # from /proc before killing the node.  Robust to spaces in comm via '#*) '.
    local _rpid; _rpid=$(pgrep -f "lib/${EXT_PKG}/${EXT_EXE}" 2>/dev/null | head -1)
    if [[ -n "${_rpid:-}" && -r "/proc/${_rpid}/stat" ]]; then
        local _stat _rest; _stat=$(cat "/proc/${_rpid}/stat" 2>/dev/null); _rest=${_stat#*) }
        local _f; read -ra _f <<< "$_rest"
        local _npose; _npose=$(wc -l < "$TUM_LIVE" 2>/dev/null || echo 0)
        if [[ "${_npose:-0}" -gt 0 ]]; then
            awk -v u="${_f[11]:-0}" -v s="${_f[12]:-0}" -v tk="$(getconf CLK_TCK)" -v n="$_npose" \
                'BEGIN{c=(u+s)/tk; printf "  [timing] node cpu: total=%.2fs poses=%d mean=%.2f ms/scan (actual compute)\n", c, n, 1000*c/n}'
        fi
    fi

    kill "$node_pid" "$conv_pid" "$rec_pid" ${imu_pid:+"$imu_pid"} 2>/dev/null || true
    sleep 1
    pkill -9 -f "lib/${EXT_PKG}/${EXT_EXE}" 2>/dev/null || true
    pkill -9 -f "lib/regnonrep/" 2>/dev/null || true

    if [[ ! -s "$TUM_LIVE" ]]; then
        echo "  ERROR: no TUM written (${method}/${seq}) — see ${method}__${ds}_${seq}.node.log" >&2
        return 1
    fi
    cp "$TUM_LIVE" "$result_tum"
    echo "  saved → $(basename "$result_tum")"
    if $ALIGN_TS && [[ -f "$gt_tum" ]]; then align_timestamps "$gt_tum" "$result_tum" "$START_OFFSET"; fi
    if $FRAME_CORRECT; then
        rm -f "${result_tum%.tum}.orig.tum"
        iilabs3d correct-frame "$result_tum" lidar --sensor "${sensor//-/_}" 2>/dev/null \
            || echo "  (frame-correct skipped)"
    fi

    echo "  [eval] $(basename "$result_tum") vs GT …"
    if [[ -f "$gt_tum" ]]; then
        if [[ "$EVAL_KIND" == evo ]]; then
            evo_ape tum "$gt_tum" "$result_tum" --align_origin --t_max_diff 0.05 --verbose 2>&1 || echo "      (evo_ape warning)"
            extra_metrics "$gt_tum" "$result_tum"
        else
            iilabs3d eval "$gt_tum" "$result_tum" || echo "      (iilabs3d eval warning)"
        fi
    else
        echo "      (skipped — no ground truth)"
    fi
    echo "  [plot] route …"
    local gt_arg=() ; [[ -f "$gt_tum" ]] && gt_arg=(--gt "$gt_tum")
    if [[ -f "${result_ann:-}" ]]; then
        # nrlio family: annotated route (mode colours + ZUPT/degeneracy/clamp icons + strips)
        python3 "${SCRIPT_DIR}/plot_annotated.py" --no-show "${gt_arg[@]}" \
            --tum "$result_tum" --ann "$result_ann" --out "$plot_png" \
            || python3 "$PLOT_SCRIPT" --no-show "${gt_arg[@]}" "$result_tum" "$plot_png" \
            || echo "      (plot failed)"
    else
        python3 "$PLOT_SCRIPT" --no-show "${gt_arg[@]}" "$result_tum" "$plot_png" || echo "      (plot failed)"
    fi
    return 0
}

# ── main ──────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
set +u; source "${WS}/install/setup.bash"; set -u

NSEQ=${#RUNS[@]}; NMETH=${#METHODS[@]}; TOTAL=$(( NSEQ * NMETH ))
CURRENT=0; PASS=(); FAIL=()
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  run_benchmarks.sh — ${NMETH} method(s) × ${NSEQ} sequence(s) = ${TOTAL} run(s)"
echo "  methods : $(for m in "${METHODS[@]}"; do printf '%s ' "${m%%|*}"; done)"
echo "  duration: $([[ "$DURATION" -gt 0 ]] && echo "first ${DURATION}s" || echo 'full bag')"
echo "  logs    : ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════"

for m in "${METHODS[@]}"; do
    method="${m%%|*}"; exe="${m##*|}"
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r ds sensor seq <<< "$entry"
        CURRENT=$(( CURRENT + 1 ))
        tag="${method}/${ds}/${seq}"
        log="${LOG_DIR}/${method}__${ds}_${seq}.log"
        printf "\n[%d/%d] %s\n" "$CURRENT" "$TOTAL" "$tag"
        if [[ "$exe" == "__EXT__" ]]; then
            run_one_external "$method" "$ds" "$sensor" "$seq" 2>&1 | tee "$log"
        else
            run_one "$method" "$exe" "$ds" "$sensor" "$seq" 2>&1 | tee "$log"
        fi
        rc="${PIPESTATUS[0]}"
        if [[ "$rc" -eq 0 ]]; then PASS+=("$tag"); else FAIL+=("$tag"); fi
    done
done

# ── collect every route plot + trajectory into the results dir ────────────────
PLOTS_DIR="${LOG_DIR}/plots"; TRAJ_DIR="${LOG_DIR}/trajectories"
mkdir -p "$PLOTS_DIR" "$TRAJ_DIR"; NPLOTS=0; NTRAJ=0
for m in "${METHODS[@]}"; do
    method="${m%%|*}"
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r ds sensor seq <<< "$entry"
        configure_dataset "$ds" || continue
        base="${DATASET_ROOT}/${sensor}/${seq}"
        src_png="${base}/trajectory_${method}.png"
        src_tum="${base}/result_${method}.tum"
        if [[ -f "$src_png" ]]; then
            cp -f "$src_png" "${PLOTS_DIR}/${method}__${ds}_${seq}.png"; NPLOTS=$(( NPLOTS + 1 ))
        fi
        if [[ -f "$src_tum" ]]; then
            cp -f "$src_tum" "${TRAJ_DIR}/${method}__${ds}_${seq}.tum"; NTRAJ=$(( NTRAJ + 1 ))
        fi
        [[ -f "${base}/result_${method}.ann.csv" ]] && \
            cp -f "${base}/result_${method}.ann.csv" "${TRAJ_DIR}/${method}__${ds}_${seq}.ann.csv"
        [[ -f "${base}/result_${method}.proc.csv" ]] && \
            cp -f "${base}/result_${method}.proc.csv" "${TRAJ_DIR}/${method}__${ds}_${seq}.proc.csv"
        # saved map (opt-in --save-maps): collect .pcd + auto MME + 5 white-bg views
        if [[ "$SAVE_MAPS" == "on" && -f "${base}/result_${method}.pcd" ]]; then
            cp -f "${base}/result_${method}.pcd" "${TRAJ_DIR}/${method}__${ds}_${seq}.pcd"
            python3 "${SCRIPT_DIR}/map_report.py" "${base}/result_${method}.pcd" \
                --out-dir "$PLOTS_DIR" --tag "${method}__${ds}_${seq}" \
                --method "$method" --dataset "$ds" --seq "$seq" \
                --mme-csv "${LOG_DIR}/mme.csv" \
                && echo "  map report (MME + 5 views) → ${method}__${ds}_${seq}"
        fi
    done
done
echo ""
echo "  saved ${NTRAJ} trajectory file(s) → ${TRAJ_DIR}"
echo "  saved ${NPLOTS} route plot(s)     → ${PLOTS_DIR}"

# ── per-sequence processing-time comparison plot (all methods) ────────────────
declare -A SEEN_SEQ
for entry in "${RUNS[@]}"; do
    IFS='|' read -r ds sensor seq <<< "$entry"
    key="${ds}|${seq}"
    [[ -n "${SEEN_SEQ[$key]:-}" ]] && continue
    SEEN_SEQ[$key]=1
    configure_dataset "$ds" || continue
    seqdir="${DATASET_ROOT}/${sensor}/${seq}"
    if ls "${seqdir}"/result_*.proc.csv >/dev/null 2>&1; then
        python3 "${SCRIPT_DIR}/plot_proctime.py" --dir "$seqdir" \
            --out "${PLOTS_DIR}/zz_proctime__${ds}_${seq}.png" \
            --title "Processing time — ${ds}/${seq}" 2>/dev/null \
            && echo "  saved proc-time plot → zz_proctime__${ds}_${seq}.png"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  SUMMARY  (${#PASS[@]} passed / ${#FAIL[@]} failed / ${TOTAL} total)"
if [[ ${#FAIL[@]} -gt 0 ]]; then
    echo "  FAILED:"; for t in "${FAIL[@]}"; do echo "    !!  $t"; done
fi
echo "  logs: ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════"
[[ ${#FAIL[@]} -eq 0 ]]
