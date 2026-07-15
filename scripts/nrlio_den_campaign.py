#!/usr/bin/env python3
"""
nrlio_den_campaign.py — 14 h Tier campaign to tune the DENSITY-based p2p↔GICP
registration switch of the `nrlio_op_den` variant, driven through the web-bench
engine (run_benchmarks.sh).

Problem being optimised: after the smoothing fix, nrlio_op_den matches
nrlio_optimized on Avia but still WRONGLY fires GICP on Horizon (narrow 27°
vertical FoV → different density dynamics), degrading it (indoor2/3_horizen
0.57→3.26, 1.72→3.05).  GICP hurts on these confined scenes, so "good switching"
here means: stay p2p in closed space, only fire GICP on a genuine sustained
density drop.  This campaign sweeps the switch params to find that.

Design:
  * method = nrlio_op_den, Tier only (Avia+Horizon indoor1/2/3/6).
  * bag rate 0.5x → node keeps up → clean, low-variance input.
  * sweeps: den_window, den_ratio band, den_voxel, den_base_alpha, den_subsample.
  * records GICP% per run (from the annotation sidecar) so switching behaviour is
    visible alongside ATE.
  * a "p2p-floor" reference (band so wide GICP never fires) = the target to beat.
  * Phase 3 confirms the best on all 8 with repeats (mean±std).

Resumable, 14 h budget.  Usage:  python3 nrlio_den_campaign.py [--report]
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
OUTDIR = f"{REPO}/benchmark_results/_campaign_den"
CSV_PATH = f"{OUTDIR}/master.csv"
PROG = f"{OUTDIR}/progress.log"
REPORT = f"{OUTDIR}/report.md"
OVLDIR = f"{OUTDIR}/overlays"
RUNLOGS = f"{OUTDIR}/runlogs"

METHOD = "nrlio_op_den"
BUDGET_S = float(os.environ.get("CAMPAIGN_BUDGET_S", 14 * 3600))
BAG_RATE = os.environ.get("CAMPAIGN_BAG_RATE", "0.5")
POST_WAIT = os.environ.get("CAMPAIGN_POST_WAIT", "10")
DIVERGE_ATE = 10.0

AVIA = ["indoor1_avia", "indoor2_avia", "indoor3_avia", "indoor6_avia"]
HORIZEN = ["indoor1_horizen", "indoor2_horizen", "indoor3_horizen", "indoor6_horizen"]
FULL = AVIA + HORIZEN
# screen: the two failing Horizon seqs + two well-behaved Avia seqs
SCREEN = ["indoor2_horizen", "indoor3_horizen", "indoor2_avia", "indoor3_avia"]

CSV_COLS = ["exp_id", "rep", "phase", "axis", "overlay", "dataset", "seq",
            "ate_se3", "ate_origin", "are_deg", "rpe_1m", "proc_ms", "gicp_pct",
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
    q.append(dict(exp_id="p0_default", phase=0, axis="baseline", overlay={},
                  seqs=FULL, reps=1))
    # p2p-floor reference: band so wide GICP never fires -> pure point-to-plane
    q.append(dict(exp_id="p0_p2p_floor", phase=0, axis="baseline",
                  overlay={"den_ratio_low": 0.01, "den_ratio_high": 100.0},
                  seqs=FULL, reps=1))
    # sweeps (screen subset, single clean 0.5x run each)
    sweeps = {
        "window":  [("den_window", v) for v in (30, 60, 90, 120)],
        "voxel":   [("den_voxel", v) for v in (0.2, 0.3, 0.5)],
        "alpha":   [("den_base_alpha", v) for v in (0.01, 0.02, 0.05)],
        "subsamp": [("den_subsample", v) for v in (2000, 4000, 8000)],
    }
    for axis, vals in sweeps.items():
        for name, v in vals:
            eid = f"p2_{axis}_{str(v).replace('.', 'p')}"
            q.append(dict(exp_id=eid, phase=2, axis=axis, overlay={name: v},
                          seqs=SCREEN, reps=1))
    # band sweep (two params together)
    for lo, hi in ((0.90, 1.10), (0.85, 1.15), (0.80, 1.20), (0.75, 1.25), (0.65, 1.35)):
        q.append(dict(exp_id=f"p2_band_{str(lo).replace('.','p')}_{str(hi).replace('.','p')}",
                      phase=2, axis="band",
                      overlay={"den_ratio_low": lo, "den_ratio_high": hi}, seqs=SCREEN, reps=1))
    return q


# ---- run one experiment ------------------------------------------------------
SEQ_RE = re.compile(r"^\s*(tier_avia|tier_horizen)\s*/\s*\S+\s*/\s*(\S+)\s*$")
RMSE_RE = re.compile(r"^\s*rmse\s+([0-9.eE+-]+)")
METRICS_RE = re.compile(r"\[metrics\]\s*ate_se3=([0-9.eE+-]+)\s+are_deg=([0-9.eE+-]+)\s+rpe_1m=([0-9.eE+-]+)")
TIMING_RE = re.compile(r"\[timing\] per-scan processing:\s*mean=([0-9.]+)")


def sensor_dir(seq):
    return "Livox_avia" if seq in AVIA else "Livox_horizen"


def gicp_pct(seq):
    """Fraction of scans that used a GICP path (from the annotation sidecar)."""
    p = f"{TIER}/{sensor_dir(seq)}/{seq}/result_{METHOD}.ann.csv"
    try:
        rows = list(csv.DictReader(open(p)))
    except OSError:
        return ""
    if not rows:
        return ""
    g = sum(1 for r in rows if "gicp" in r.get("method", "") and "gicp*" not in r.get("method", ""))
    return f"{100.0 * g / len(rows):.1f}"


def parse_run(text, seqs):
    res, cur = {}, None
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
           "--skip-iilab", "--skip-cern", f"--sequences={','.join(seqs)}"]
    if not any(s in AVIA for s in seqs):
        cmd.append("--skip-tier-avia")
    if not any(s in HORIZEN for s in seqs):
        cmd.append("--skip-tier-horizen")
    if exp["overlay"]:
        cmd.insert(2, f"--params-overlay={write_overlay(exp['exp_id'], exp['overlay'])}")
    subprocess.run("pkill -9 -f 'lib/regnonrep' 2>/dev/null; pkill -9 -f 'ros2 bag play' 2>/dev/null; "
                   "pkill -9 -f imu_rescale 2>/dev/null; sleep 1", shell=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600 * len(seqs), env=env)
        out = p.stdout + "\n" + p.stderr; rc = p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + "\nTIMEOUT\n" + (e.stderr or ""); rc = 124
    with open(f"{RUNLOGS}/{exp['exp_id']}_r{rep}.log", "w") as f:
        f.write(out)
    subprocess.run("pkill -9 -f 'lib/regnonrep' 2>/dev/null; pkill -9 -f 'ros2 bag play' 2>/dev/null; "
                   "pkill -9 -f imu_rescale 2>/dev/null", shell=True)
    parsed = parse_run(out, seqs)
    dt = time.time() - t0
    log(f"  {exp['exp_id']} rep{rep} rc={rc} {dt/60:.1f}min  " +
        " ".join(f"{s.split('_')[0][:5]}={parsed[s]['ate_se3'] or 'X'}[{gicp_pct(s) or '?'}%g]" for s in seqs))
    rows = []
    for seq in seqs:
        r = parsed[seq]
        ds = "tier_avia" if seq in AVIA else "tier_horizen"
        rows.append([exp["exp_id"], rep, exp["phase"], exp["axis"], json.dumps(exp["overlay"]),
                     ds, seq, r["ate_se3"], r["ate_origin"], r["are_deg"], r["rpe_1m"],
                     r["proc_ms"], gicp_pct(seq), r["diverged"], r["status"],
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
    for row in csv.DictReader(open(CSV_PATH)):
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


def _gicp_mean(rs):
    xs = [float(r["gicp_pct"]) for r in rs if r.get("gicp_pct")]
    return sum(xs) / len(xs) if xs else float("nan")


def analyze_and_report():
    if not os.path.exists(CSV_PATH):
        return {}
    rows = list(csv.DictReader(open(CSV_PATH)))
    L = ["# nrlio_op_den — density-switch tuning campaign\n",
         f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')}, {len(rows)} rows, rate {BAG_RATE}x_\n",
         "\nGoal: switching that stays p2p in closed space (GICP hurts here). "
         "Lower ATE-SE3 is better; GICP% shows how much it switched.\n"]
    p2 = [r for r in rows if r["phase"] == "2"]
    axes = {}
    for r in p2:
        axes.setdefault(r["axis"], {}).setdefault(r["exp_id"], []).append(r)
    best = {}
    L.append("\n## Phase 2 — per-axis (screen subset)\n")
    for axis, exps in sorted(axes.items()):
        L.append(f"\n### {axis}\n| overlay | mean ATE-SE3 | GICP% | diverged |\n|---|---|---|---|")
        ranked = []
        for eid, rs in exps.items():
            sc = score([r["ate_se3"] for r in rs]); nd = sum(int(r["diverged"]) for r in rs)
            ranked.append((sc, rs[0]["overlay"], _gicp_mean(rs), nd))
        for sc, ov, gp, nd in sorted(ranked):
            L.append(f"| `{ov}` | {sc:.3f} | {gp:.0f} | {nd} |")
        best[axis] = sorted(ranked)[0]
    combined = {}
    for axis, (sc, ov, gp, nd) in best.items():
        if nd == 0:
            combined.update(json.loads(ov))
    L.append("\n## Suggested combined-best switch config\n```yaml\nlio_node:\n  ros__parameters:")
    for k, v in combined.items():
        L.append(f"    {k}: {fmt(v)}")
    L.append("```")
    # phase 0 / 3 full comparison
    ref = [r for r in rows if r["phase"] in ("0", "3")]
    if ref:
        L.append("\n## Full-set comparison (mean over reps)\n| config | mean ATE-SE3 | GICP% | diverged reps |\n|---|---|---|---|")
        groups = {}
        for r in ref:
            groups.setdefault(r["exp_id"], []).append(r)
        out = []
        for eid, rs in groups.items():
            sv = {}
            for r in rs:
                sv.setdefault(r["seq"], []).append(r["ate_se3"])
            per = {s: score(v) for s, v in sv.items()}
            out.append((sum(per.values()) / len(per), eid, _gicp_mean(rs),
                        sum(int(r["diverged"]) for r in rs)))
        for m, eid, gp, nd in sorted(out):
            L.append(f"| {eid} | {m:.3f} | {gp:.0f} | {nd} |")
    with open(REPORT, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"report -> {REPORT}")
    return combined


def build_phase3(combined):
    exps = [dict(exp_id="p3_p2p_floor", phase=3, axis="confirm",
                 overlay={"den_ratio_low": 0.01, "den_ratio_high": 100.0}, seqs=FULL, reps=3)]
    if combined:
        exps.insert(0, dict(exp_id="p3_combined", phase=3, axis="confirm",
                            overlay=combined, seqs=FULL, reps=3))
    return exps


# ---- main --------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if "--report" in sys.argv:
        analyze_and_report(); return
    t_start = time.time()
    log(f"=== nrlio_op_den density-switch campaign (budget {BUDGET_S/3600:.1f}h, rate {BAG_RATE}x) ===")
    queue = build_static_queue()
    done = load_done()

    def budget_left():
        return BUDGET_S - (time.time() - t_start)

    def run_all(exp):
        for rep in range(exp["reps"]):
            if budget_left() < 600:
                log("budget nearly exhausted — stopping"); return False
            if done.get((exp["exp_id"], str(rep)), set()).issuperset(set(exp["seqs"])):
                continue
            log(f"RUN {exp['exp_id']} rep{rep} phase{exp['phase']} [{len(exp['seqs'])} seqs] "
                f"left={budget_left()/3600:.1f}h")
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
    log(f"=== done: {(time.time()-t_start)/3600:.1f}h elapsed ===")


if __name__ == "__main__":
    main()
