#!/usr/bin/env python3.10
"""One-off: run ros_lio_v2 on the full indoor1_avia bag with debug CSV output.
Reuses tune_lio_v2's proven launch/shutdown (SIGINT via killpg)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tune_lio_v2 import run_lio, ate_rmse_csv, DEFAULT_BAG, DEFAULT_GT, PKG

cfg = os.path.join(PKG, "config", "lio_v2_tier_avia_tuned_full.yaml")
out_dir = os.path.join(PKG, "tum", "fixes_check")
os.makedirs(out_dir, exist_ok=True)
csv = os.path.join(out_dir, "indoor1_p2p_primary.csv")
log = os.path.join(out_dir, "indoor1_p2p_primary.log")

print(f"config : {cfg}")
print(f"bag    : {DEFAULT_BAG}")
print(f"csv    : {csv}")
t0 = time.time()
ok = run_lio(cfg, csv, DEFAULT_BAG, rate=0.3, settle=3.0, post_wait=8.0, log_path=log)
if not ok:
    print(f"NO trajectory produced (see {log})"); sys.exit(1)
rmse, n, off = ate_rmse_csv(csv, DEFAULT_GT, max_diff=0.05)
print(f"DONE in {time.time()-t0:.0f}s  ATE-RMSE={rmse:.4f} m  matched={n}  offset={off:.3f}s")
print(f"csv -> {csv}")
print(f"log -> {log}")
