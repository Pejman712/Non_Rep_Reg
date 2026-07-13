#!/usr/bin/env python3
"""
nrlio_campaign.py — Tier-focused CONVERGENCE optimization campaign for the
`nrlio_optimized` variant, driven through the web-bench engine (run_benchmarks.sh).

This campaign targets the accuracy PLATEAU (not divergence) identified from the
logs: the estimator tracks but stalls at ~0.4-2.4 m instead of converging tighter.
Design decisions from that diagnosis:

  * Tier only (Avia 1/2/3 + Horizon 1/2/3) — "best performance among Tier".
  * Bag rate 0.5x — the node ALWAYS keeps up (proc ~50-120 ms < 200 ms budget),
    so the input is clean and the run-to-run timing variance is removed.
  * Sweep the convergence levers the logs pointed at:
      - opt_accum_voxel   (precision ceiling — coarse 0.16 caps accuracy)
      - gen_voxel_min     (registration voxel floor — fine geometry)
      - kf_max_iterations (per-scan iESKF convergence — only 4 today)
      - opt_chi2          (acceptance quality — loose 200 admits conf=0.1 GICP)
      - degen_planarity_ratio / degen_min_extent (how often it "coasts")
      - opt_knee_ms       (accumulation depth) / opt_max_step (clamp)
  * Phase 3 confirms the best combined configs on all 6 with REPEATS for a
    stable mean ± std (indoor3 is fragile).

NOTE `nrlio_optimized` FORCES its tuned values from opt_* params, so those axes
override the opt_* names; the rest override the base names.

Resumable (skips finished (exp,rep) rows), honors a wall-clock budget (14 h).
    python3 nrlio_campaign.py            # run / resume
    python3 nrlio_campaign.py --report   # re-print analysis
"""
import csv
import json
import os
import re
import subprocess
import sys
import time

REPO = "/u/97/habibip1/unix/ros2_ws/src/regnonrep"
RUN_SH = f"{REPO}/scripts/run_benchmarks.sh"
TIER = "/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/Tier"
OUTDIR = f"{REPO}/benchmark_results/_campaign2"
CSV_PATH = f"{OUTDIR}/master.csv"
PROG = f"{OUTDIR}/progress.log"
REPORT = f"{OUTDIR}/report.md"
OVLDIR = f"{OUTDIR}/overlays"
RUNLOGS = f"{OUTDIR}/runlogs"

METHOD = "nrlio_optimized"
BUDGET_S = float(os.environ.get("CAMPAIGN_BUDGET_S", 14 * 3600))
BAG_RATE = os.environ.get("CAMPAIGN_BAG_RATE", "0.5")
POST_WAIT = os.environ.get("CAMPAIGN_POST_WAIT", "10")
DIVERGE_ATE = 10.0

AVIA = ["indoor1_avia", "indoor2_avia", "indoor3_avia"]
HORIZEN = ["indoor1_horizen", "indoor2_horizen", "indoor3_horizen"]
FULL = AVIA + HORIZEN
SCREEN = ["indoor1_avia", "indoor3_avia", "indoor1_horizen", "indoor3_horizen"]

CSV_COLS = ["exp_id", "rep", "phase", "axis", "overlay", "dataset", "seq",
            "ate_se3", "ate_origin", "are_deg", "rpe_1m", "proc_ms", "poses",
            "diverged", "status", "ts"]


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


def write_overlay(exp_id, overlay):
    os.makedirs(OVLDIR, exist_ok=True)
    path = f"{OVLDIR}/{exp_id}.yaml"
    with open(path, "w") as f:
        f.write("lio_node:\n  ros__parameters:\n")
        for k, v in overlay.items():
            f.write(f"    {k}: {fmt(v)}\n")
    return path


# ---- experiment queue --------------------------------------------------------
def build_static_queue():
    q = []
    q.append(dict(exp_id="p0_baseline", phase=0, axis="baseline", overlay={},
                  seqs=FULL, reps=1))
    # convergence sweeps (screen subset, single clean 0.5x run each)
    sweeps = {
        "accvox":   [("opt_accum_voxel", v) for v in (0.08, 0.10, 0.12, 0.14, 0.16)],
        "voxmin":   [("gen_voxel_min", v) for v in (0.03, 0.05, 0.08)],
        "iters":    [("kf_max_iterations", v) for v in (4, 6, 8)],
        "chi2":     [("opt_chi2", v) for v in (75.0, 100.0, 150.0, 200.0)],
        "dplanar":  [("degen_planarity_ratio", v) for v in (0.01, 0.03, 0.06)],
        "dextent":  [("degen_min_extent", v) for v in (0.5, 1.0, 1.5)],
        "knee":     [("opt_knee_ms", v) for v in (100.0, 200.0, 300.0)],
        "maxstep":  [("opt_max_step", v) for v in (0.4, 0.6, 1.0)],
    }
    for axis, vals in sweeps.items():
        for name, v in vals:
            eid = f"p2_{axis}_{str(v).replace('.', 'p')}"
            q.append(dict(exp_id=eid, phase=2, axis=axis, overlay={name: v},
                          seqs=SCREEN, reps=1))
    return q


# ---- run one experiment through run_benchmarks.sh ----------------------------
SEQ_RE = re.compile(r"^\s*(tier_avia|tier_horizen)\s*/\s*\S+\s*/\s*(\S+)\s*$")
RMSE_RE = re.compile(r"^\s*rmse\s+([0-9.eE+-]+)")
METRICS_RE = re.compile(r"\[metrics\]\s*ate_se3=([0-9.eE+-]+)\s+are_deg=([0-9.eE+-]+)\s+rpe_1m=([0-9.eE+-]+)")
TIMING_RE = re.compile(r"\[timing\] per-scan processing:\s*mean=([0-9.]+)")


def sensor_dir(seq):
    return "Livox_avia" if seq in AVIA else "Livox_horizen"


def pose_count(seq):
    p = f"{TIER}/{sensor_dir(seq)}/{seq}/result_{METHOD}.tum"
    try:
        return sum(1 for _ in open(p))
    except OSError:
        return 0


def parse_run(text, seqs):
    res = {}
    cur = None
    for line in text.splitlines():
        m = SEQ_RE.match(line)
        if m:
            cur = m.group(2)
            res.setdefault(cur, dict(ate_se3="", ate_origin="", are_deg="",
                                     rpe_1m="", proc_ms="", diverged=0, status="run"))
            continue
        if cur is None:
            continue
        r = res[cur]
        mm = RMSE_RE.match(line)
        if mm and r["ate_origin"] == "":
            r["ate_origin"] = mm.group(1)
        mt = TIMING_RE.search(line)
        if mt:
            r["proc_ms"] = mt.group(1)
        me = METRICS_RE.search(line)
        if me:
            r["ate_se3"], r["are_deg"], r["rpe_1m"] = me.groups()
            r["status"] = "ok"
    for seq in seqs:
        if seq not in res:
            res[seq] = dict(ate_se3="", ate_origin="", are_deg="", rpe_1m="",
                            proc_ms="", diverged=1, status="missing")
            continue
        r = res[seq]
        try:
            if r["ate_se3"] and float(r["ate_se3"]) > DIVERGE_ATE:
                r["diverged"] = 1
        except ValueError:
            pass
        if r["status"] == "run" and not r["ate_se3"]:
            r["status"] = "no_metrics"; r["diverged"] = 1
    return res


def run_experiment(exp, rep):
    os.makedirs(RUNLOGS, exist_ok=True)
    seqs = exp["seqs"]
    cmd = ["bash", RUN_SH, f"--methods={METHOD}",
           f"--bag-rate={BAG_RATE}", f"--post-wait={POST_WAIT}", "--start-offset=5",
           "--skip-iilab", f"--sequences={','.join(seqs)}"]
    if not any(s in AVIA for s in seqs):
        cmd.append("--skip-tier-avia")
    if not any(s in HORIZEN for s in seqs):
        cmd.append("--skip-tier-horizen")
    if exp["overlay"]:
        cmd.insert(2, f"--params-overlay={write_overlay(exp['exp_id'], exp['overlay'])}")
    subprocess.run("pkill -9 -f 'lib/regnonrep' 2>/dev/null; pkill -9 -f 'ros2 bag play' 2>/dev/null; "
                   "pkill -9 -f imu_rescale 2>/dev/null; sleep 1", shell=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    timeout = 600 * len(seqs)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        out = p.stdout + "\n" + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + "\nTIMEOUT\n" + (e.stderr or ""); rc = 124
    with open(f"{RUNLOGS}/{exp['exp_id']}_r{rep}.log", "w") as f:
        f.write(out)
    subprocess.run("pkill -9 -f 'lib/regnonrep' 2>/dev/null; pkill -9 -f 'ros2 bag play' 2>/dev/null; "
                   "pkill -9 -f imu_rescale 2>/dev/null", shell=True)
    parsed = parse_run(out, seqs)
    dt = time.time() - t0
    log(f"  {exp['exp_id']} rep{rep} rc={rc} {dt/60:.1f}min  " +
        " ".join(f"{s.split('_')[0][:5]}={parsed[s]['ate_se3'] or 'X'}" for s in seqs))
    rows = []
    for seq in seqs:
        r = parsed[seq]
        ds = "tier_avia" if seq in AVIA else "tier_horizen"
        rows.append([exp["exp_id"], rep, exp["phase"], exp["axis"], json.dumps(exp["overlay"]),
                     ds, seq, r["ate_se3"], r["ate_origin"], r["are_deg"], r["rpe_1m"],
                     r["proc_ms"], pose_count(seq), r["diverged"], r["status"],
                     time.strftime("%Y-%m-%d %H:%M:%S")])
    return rows


# ---- CSV / logging -----------------------------------------------------------
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def load_done():
    done = {}
    if not os.path.exists(CSV_PATH):
        return done
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            done.setdefault((row["exp_id"], row["rep"]), set()).add(row["seq"])
    return done


def append_rows(rows):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_COLS)
        w.writerows(rows)


# ---- analysis ----------------------------------------------------------------
def score(vals):
    xs = []
    for v in vals:
        try:
            x = float(v)
        except (ValueError, TypeError):
            x = 100.0
        xs.append(x if x <= DIVERGE_ATE else 100.0)
    return sum(xs) / len(xs) if xs else 1e9


def analyze_and_report():
    if not os.path.exists(CSV_PATH):
        return {}
    rows = list(csv.DictReader(open(CSV_PATH)))
    lines = ["# nrlio_optimized — Tier convergence campaign\n",
             f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')}, {len(rows)} rows, "
             f"rate {BAG_RATE}x_\n"]

    # phase 2 per-axis
    p2 = [r for r in rows if r["phase"] == "2"]
    axes = {}
    for r in p2:
        axes.setdefault(r["axis"], {}).setdefault(r["exp_id"], []).append(r)
    best = {}
    lines.append("\n## Phase 2 — convergence sweeps (screen subset, mean ATE-SE3)\n")
    for axis, exps in sorted(axes.items()):
        lines.append(f"\n### {axis}\n| overlay | mean ATE-SE3 | diverged |\n|---|---|---|")
        ranked = []
        for eid, rs in exps.items():
            sc = score([r["ate_se3"] for r in rs])
            nd = sum(int(r["diverged"]) for r in rs)
            ranked.append((sc, rs[0]["overlay"], nd))
        for sc, ov, nd in sorted(ranked):
            lines.append(f"| `{ov}` | {sc:.3f} | {nd} |")
        best[axis] = sorted(ranked)[0]

    combined = {}
    for axis, (sc, ov, nd) in best.items():
        if nd == 0:                      # only adopt an axis winner that didn't diverge
            combined.update(json.loads(ov))
    lines.append("\n## Suggested combined-best overlay\n```yaml\nlio_node:\n  ros__parameters:")
    for k, v in combined.items():
        lines.append(f"    {k}: {fmt(v)}")
    lines.append("```")

    # phase 3: mean ± std per config on FULL
    p3 = [r for r in rows if r["phase"] == "3"] + [r for r in rows if r["exp_id"] == "p0_baseline"]
    if p3:
        lines.append("\n## Phase 3 — combined-best vs baseline (all 6, mean±std over reps)\n")
        lines.append("| config | mean ATE-SE3 | per-seq (mean) | diverged reps |")
        lines.append("|---|---|---|---|")
        groups = {}
        for r in p3:
            groups.setdefault(r["exp_id"], []).append(r)
        out = []
        for eid, rs in groups.items():
            import statistics
            seqvals = {}
            for r in rs:
                seqvals.setdefault(r["seq"], []).append(r["ate_se3"])
            per = {s: score(v) for s, v in seqvals.items()}
            m = sum(per.values()) / len(per)
            nd = sum(int(r["diverged"]) for r in rs)
            detail = " ".join(f"{s.split('_')[0][:4]}:{per[s]:.2f}" for s in per)
            out.append((m, eid, detail, nd))
        for m, eid, detail, nd in sorted(out):
            lines.append(f"| {eid} | {m:.3f} | {detail} | {nd} |")

    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"report -> {REPORT}")
    return combined


def build_phase3(combined):
    exps = [dict(exp_id="p3_baseline", phase=3, axis="confirm", overlay={}, seqs=FULL, reps=3)]
    if combined:
        exps.append(dict(exp_id="p3_combined", phase=3, axis="confirm",
                         overlay=combined, seqs=FULL, reps=3))
        # precision-only subset: voxel/iters/chi2 winners, leave knee/maxstep default
        prec = {k: v for k, v in combined.items()
                if k in ("opt_accum_voxel", "gen_voxel_min", "kf_max_iterations", "opt_chi2")}
        if prec and prec != combined:
            exps.append(dict(exp_id="p3_precision", phase=3, axis="confirm",
                             overlay=prec, seqs=FULL, reps=3))
    return exps


# ---- main --------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if "--report" in sys.argv:
        analyze_and_report(); return
    t_start = time.time()
    log(f"=== nrlio_optimized Tier convergence campaign (budget {BUDGET_S/3600:.1f}h, "
        f"rate {BAG_RATE}x) ===")
    queue = build_static_queue()
    done = load_done()

    def budget_left():
        return BUDGET_S - (time.time() - t_start)

    def run_all(exp):
        for rep in range(exp["reps"]):
            if budget_left() < 600:
                log("budget nearly exhausted — stopping"); return False
            have = done.get((exp["exp_id"], str(rep)), set())
            if have.issuperset(set(exp["seqs"])):
                continue
            log(f"RUN {exp['exp_id']} rep{rep} phase{exp['phase']} [{len(exp['seqs'])} seqs] "
                f"budget_left={budget_left()/3600:.1f}h")
            append_rows(run_experiment(exp, rep))
        return True

    for exp in queue:
        if not run_all(exp):
            break

    if budget_left() > 2400:
        combined = analyze_and_report()
        done = load_done()
        for exp in build_phase3(combined):
            if budget_left() < 600:
                break
            run_all(exp)

    analyze_and_report()
    log(f"=== campaign done: {(time.time()-t_start)/3600:.1f}h elapsed ===")


if __name__ == "__main__":
    main()
