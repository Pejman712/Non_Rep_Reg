#!/usr/bin/env python3.10
"""
tune_lio_v2.py — Optuna tuning of the IESKF18 Kalman filter AND the GICP
                 registration parameters in ros_lio_v2.py, evaluated on the
                 FULL indoor1_avia bag against its TUM ground truth.

Per trial
---------
  1. Sample ~23 params: filter noise/init/gating/motion/ZUPT + GICP
     corr-distance / voxel / iterations / submap-radius / min-conf / map-voxel /
     IMU-vs-nonrep initial-guess weight.
  2. Render a YAML (base = tum/run3/lio_v2_avia.yaml) with the overrides.
  3. Launch ros_lio_v2.py DIRECTLY (installed script, via `exec` so the tracked
     PID is the node), play the full bag, collect the node's per-scan debug CSV.
  4. Stop the node with os.killpg (SIGINT→SIGKILL syscalls — never shell
     pkill/`-9`, which the harness blocks).
  5. ATE-RMSE [m] from the debug CSV (auto clock-offset time-association +
     Umeyama SE(3) alignment) is the objective Optuna minimises.

Study is SQLite-backed (resumable, interruptible).

Prereq:  python3.10 -m pip install --user optuna
Run:     python3.10 scripts/tune_lio_v2.py                 # 50 trials, full bag
         python3.10 scripts/tune_lio_v2.py --baseline-only # eval base cfg once
"""

import argparse
import copy
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:
    sys.exit("PyYAML missing: python3.10 -m pip install --user pyyaml")

# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
PKG  = os.path.dirname(HERE)
WS   = os.path.realpath(os.path.join(PKG, "..", ".."))
SETUP_BASH = os.path.join(WS, "install", "setup.bash")
V2_SCRIPT  = os.path.join(WS, "install", "regnonrep", "lib", "regnonrep", "ros_lio_v2.py")

DEFAULT_BASE_CFG = os.path.join(PKG, "tum", "run3", "lio_v2_avia.yaml")
DEFAULT_BAG = ("/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/"
               "Tier/Livox_avia/indoor1_avia/indoor1_avia")
DEFAULT_GT  = ("/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/"
               "Tier/Livox_avia/indoor1_avia/indoor1_avia.tum")
NODE_KEY = "lio_node_v2"
WORK_DIR = os.path.join(PKG, "tum", "optuna_full")
FAIL_PENALTY = 1.0e3


# --------------------------------------------------------------------------- #
# Trajectory evaluation
# --------------------------------------------------------------------------- #
def load_tum(path: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 4 and not line.startswith("#"):
                rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
    if not rows:
        return np.empty(0), np.empty((0, 3))
    a = np.asarray(rows)
    return a[:, 0], a[:, 1:4]


def load_debug_xyz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read stamp,x,y,z (cols 1..4) from the node's debug CSV."""
    t, xyz = [], []
    with open(path) as fh:
        fh.readline()                       # header
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 5:
                continue
            try:
                t.append(float(p[1]))
                xyz.append([float(p[2]), float(p[3]), float(p[4])])
            except ValueError:
                continue
    return np.asarray(t), np.asarray(xyz).reshape(-1, 3)


def associate(t_est, t_gt, max_diff):
    if t_est.size == 0 or t_gt.size == 0:
        return np.empty(0, int), np.empty(0, int), 0.0
    order = np.argsort(t_gt)
    t_gt_s = t_gt[order]

    def nearest(te):
        j = np.clip(np.searchsorted(t_gt_s, te), 1, len(t_gt_s) - 1)
        return np.where(np.abs(te - t_gt_s[j - 1]) <= np.abs(te - t_gt_s[j]), j - 1, j)

    j0 = nearest(t_est)
    offset = float(np.median(t_est - t_gt_s[j0]))
    te = t_est - offset
    j = nearest(te)
    keep = np.abs(te - t_gt_s[j]) <= max_diff
    return np.where(keep)[0], order[j[keep]], offset


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / src.shape[0]
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def ate_rmse_csv(csv_path, gt_path, max_diff=0.05):
    t_e, p_e = load_debug_xyz(csv_path)
    t_g, p_g = load_tum(gt_path)
    if t_e.size < 10 or t_g.size < 10:
        return float("inf"), 0, 0.0
    ie, ig, off = associate(t_e, t_g, max_diff)
    if ie.size < 10:
        return float("inf"), int(ie.size), off
    R, t = umeyama(p_e[ie], p_g[ig])
    aligned = (R @ p_e[ie].T).T + t
    err = np.linalg.norm(aligned - p_g[ig], axis=1)
    return float(np.sqrt(np.mean(err ** 2))), int(ie.size), off


# --------------------------------------------------------------------------- #
# Config + process control
# --------------------------------------------------------------------------- #
def render_cfg(base, overrides, out_path):
    cfg = copy.deepcopy(base)
    params = cfg[NODE_KEY]["ros__parameters"]
    params.update(overrides)
    with open(out_path, "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)


def _popen(cmd, log):
    """Launch `cmd` in its own session; exec so the PID is the real target."""
    return subprocess.Popen(["bash", "-c", f"source '{SETUP_BASH}' >/dev/null 2>&1 && exec {cmd}"],
                            stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def _stop(proc):
    if proc is None or proc.poll() is not None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=4)
            return
        except subprocess.TimeoutExpired:
            continue


def run_lio(cfg_path, out_csv, bag_dir, rate, settle, post_wait, log_path) -> bool:
    for f in (out_csv,):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    node = None
    with open(log_path, "w") as log:
        try:
            node = _popen(
                f"python3.10 '{V2_SCRIPT}' --ros-args --params-file '{cfg_path}' "
                f"-r __node:={NODE_KEY} -p debug_csv:='{out_csv}'", log)
            time.sleep(settle)
            # full bag; generous wall timeout = bag_len/rate + margin
            play_to = 130.0 / rate + 180.0
            subprocess.run(["bash", "-c",
                            f"source '{SETUP_BASH}' >/dev/null 2>&1 && "
                            f"exec ros2 bag play '{bag_dir}' --clock --rate {rate}"],
                           stdout=log, stderr=subprocess.STDOUT, timeout=play_to)
        except subprocess.TimeoutExpired:
            log.write("\n[bag play timeout]\n")
        except Exception as e:  # noqa: BLE001
            log.write(f"\n[run_lio error: {e}]\n")
        finally:
            time.sleep(post_wait)
            _stop(node)
    return os.path.exists(out_csv) and os.path.getsize(out_csv) > 0


# --------------------------------------------------------------------------- #
# Search space: Kalman filter + registration
# --------------------------------------------------------------------------- #
def sample_params(trial) -> Dict[str, object]:
    imu_w = trial.suggest_float("imu_base_weight", 0.05, 0.95)
    return {
        # ---- IMU process noise -------------------------------------------
        "ieskf_sigma_gyro":  trial.suggest_float("ieskf_sigma_gyro",  1e-4, 1e-2, log=True),
        "ieskf_sigma_accel": trial.suggest_float("ieskf_sigma_accel", 1e-3, 1e-1, log=True),
        "ieskf_sigma_bg":    trial.suggest_float("ieskf_sigma_bg",    1e-5, 1e-3, log=True),
        "ieskf_sigma_ba":    trial.suggest_float("ieskf_sigma_ba",    1e-4, 1e-2, log=True),
        # ---- GICP measurement noise --------------------------------------
        "ieskf_meas_noise_pos": trial.suggest_float("ieskf_meas_noise_pos", 1e-3, 5e-1, log=True),
        "ieskf_meas_noise_rot": trial.suggest_float("ieskf_meas_noise_rot", 1e-3, 2e-1, log=True),
        # ---- Initial covariances -----------------------------------------
        "ieskf_init_p_cov":  trial.suggest_float("ieskf_init_p_cov",  1e-3, 1e1, log=True),
        "ieskf_init_bg_cov": trial.suggest_float("ieskf_init_bg_cov", 1e-5, 1e-1, log=True),
        "ieskf_init_ba_cov": trial.suggest_float("ieskf_init_ba_cov", 1e-5, 1e-1, log=True),
        "ieskf_init_g_cov":  trial.suggest_float("ieskf_init_g_cov",  1e-4, 1e0,  log=True),
        # Initial attitude/velocity covariance — now exposed by ros_lio_v2 so the
        # over-confident frozen prior (was 1e-6) can be tuned away.
        "ieskf_init_rot_cov": trial.suggest_float("ieskf_init_rot_cov", 1e-4, 1e-1, log=True),
        "ieskf_init_v_cov":   trial.suggest_float("ieskf_init_v_cov",   1e-4, 1e-1, log=True),
        # ---- Gating / iterations -----------------------------------------
        # Capped to a robust range: the old [10,200] let the optimiser disable
        # outlier rejection (→ sudden jumps).  Point-to-plane now supplies the
        # map constraint, so the GICP gate can stay tight.
        "gicp_chi2_threshold": trial.suggest_float("gicp_chi2_threshold", 10.0, 30.0),
        "ieskf_max_iters":     trial.suggest_int("ieskf_max_iters", 1, 6),
        # ---- Motion-adaptive noise scaling -------------------------------
        "motion_rot_noise_scale":       trial.suggest_float("motion_rot_noise_scale", 1.0, 10.0),
        "motion_trans_pos_noise_scale": trial.suggest_float("motion_trans_pos_noise_scale", 0.1, 5.0),
        # ---- ZUPT / soft-floor -------------------------------------------
        "zupt_sigma_v": trial.suggest_float("zupt_sigma_v", 1e-3, 1e-1, log=True),
        "soft_z_sigma": trial.suggest_float("soft_z_sigma", 1e-2, 5e-1, log=True),
        # ---- Registration (GICP scan-to-submap) --------------------------
        "gicp_max_corr_distance": trial.suggest_float("gicp_max_corr_distance", 0.5, 5.0),
        "gicp_voxel_size":        trial.suggest_float("gicp_voxel_size", 0.05, 0.5, log=True),
        "gicp_max_iterations":    trial.suggest_int("gicp_max_iterations", 20, 100),
        "gicp_submap_radius":     trial.suggest_float("gicp_submap_radius", 5.0, 50.0),
        "gicp_min_conf":          trial.suggest_float("gicp_min_conf", 0.05, 0.5),
        "map_voxel":              trial.suggest_float("map_voxel", 0.05, 0.3, log=True),
        # ---- Super-LIO point-to-plane refinement (sequential, after GICP) -
        "p2p_sigma":         trial.suggest_float("p2p_sigma", 1e-2, 2e-1, log=True),
        "p2p_max_corr_dist": trial.suggest_float("p2p_max_corr_dist", 0.2, 1.0),
        "p2p_scan_voxel":    trial.suggest_float("p2p_scan_voxel", 0.2, 0.5),
        "p2p_submap_voxel":  trial.suggest_float("p2p_submap_voxel", 0.2, 0.5),
        "p2p_min_corr":      trial.suggest_int("p2p_min_corr", 20, 80),
        "p2p_huber_delta":   trial.suggest_float("p2p_huber_delta", 0.05, 0.3),
        "p2p_max_iters":     trial.suggest_int("p2p_max_iters", 2, 4),
        # ---- Initial-guess fusion weights (complementary) ----------------
        "imu_base_weight":    imu_w,
        "nonrep_base_weight": 1.0 - imu_w,
    }


# --------------------------------------------------------------------------- #
def make_objective(args, base_cfg):
    def objective(trial) -> float:
        overrides = sample_params(trial)
        tag = f"trial{trial.number:04d}"
        cfg = os.path.join(WORK_DIR, f"{tag}.yaml")
        csv = os.path.join(WORK_DIR, f"{tag}.csv")
        log = os.path.join(WORK_DIR, f"{tag}.log")
        render_cfg(base_cfg, overrides, cfg)

        t0 = time.time()
        ok = run_lio(cfg, csv, args.bag, args.rate, args.settle, args.post_wait, log)
        if not ok:
            print(f"[{tag}] no trajectory — penalised", flush=True)
            return FAIL_PENALTY
        rmse, n, off = ate_rmse_csv(csv, args.gt, args.max_diff)
        if not np.isfinite(rmse):
            print(f"[{tag}] eval failed (matched={n}) — penalised", flush=True)
            return FAIL_PENALTY
        trial.set_user_attr("n_matched", n)
        trial.set_user_attr("wall_sec", round(time.time() - t0, 1))
        print(f"[{tag}] ATE-RMSE={rmse:.4f} m  matched={n}  ({time.time()-t0:.0f}s)", flush=True)
        return rmse
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=DEFAULT_BAG)
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--base-cfg", default=DEFAULT_BASE_CFG)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--post-wait", type=float, default=8.0)
    ap.add_argument("--max-diff", type=float, default=0.05)
    ap.add_argument("--study", default="lio_v2_full_indoor1_p2p")
    ap.add_argument("--storage", default="")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    with open(args.base_cfg) as fh:
        base_cfg = yaml.safe_load(fh)

    if args.baseline_only:
        cfg = os.path.join(WORK_DIR, "baseline.yaml")
        csv = os.path.join(WORK_DIR, "baseline.csv")
        log = os.path.join(WORK_DIR, "baseline.log")
        render_cfg(base_cfg, {}, cfg)
        t0 = time.time()
        ok = run_lio(cfg, csv, args.bag, args.rate, args.settle, args.post_wait, log)
        if not ok:
            print(f"baseline: NO trajectory (see {log})"); sys.exit(1)
        rmse, n, off = ate_rmse_csv(csv, args.gt, args.max_diff)
        print(f"baseline ATE-RMSE={rmse:.4f} m  matched={n}  offset={off:.3f}s  "
              f"({time.time()-t0:.0f}s)")
        sys.exit(0)

    try:
        import optuna
    except ImportError:
        sys.exit("optuna missing: python3.10 -m pip install --user optuna")

    storage = args.storage or f"sqlite:///{os.path.join(WORK_DIR, args.study + '.db')}"
    study = optuna.create_study(study_name=args.study, storage=storage,
                                direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42))
    print(f"=== Optuna full-bag tuning: {args.study} (KF + registration) ===")
    print(f"  bag {args.bag}\n  gt {args.gt}\n  rate {args.rate}  storage {storage}")
    print(f"  trials {args.n_trials} (done: {len(study.trials)})")

    study.optimize(make_objective(args, base_cfg), n_trials=args.n_trials)

    print("\n=== BEST ===")
    print(f"  value (ATE-RMSE): {study.best_value:.4f} m")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    best_cfg = os.path.join(PKG, "config", "lio_v2_tier_avia_tuned_p2p.yaml")
    best = dict(study.best_params)
    best["nonrep_base_weight"] = 1.0 - best.get("imu_base_weight", 0.3)
    render_cfg(base_cfg, best, best_cfg)
    print(f"\nBest config → {best_cfg}")


if __name__ == "__main__":
    main()
