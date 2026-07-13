#!/usr/bin/env python3
"""
plot_proctime.py — per-sequence processing-time comparison across all methods.

Reads every result_<method>.proc.csv (scan,proc_ms) in a sequence directory
(written by every regnonrep variant) and produces a two-panel comparison:
  * top    — per-scan processing time over the run, one line per method
  * bottom — mean / p95 / max processing time per method (bar chart)

External SOTA methods do not emit per-scan timing, so they only appear if a
proc.csv exists; their aggregate proc time is in the bench diagnostics table.

    plot_proctime.py --dir <seq_dir> --out <png> [--title "..."]
"""
import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b",
          "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#637939"]


def read_proc(path):
    scans, ms = [], []
    for r in csv.DictReader(open(path)):
        try:
            scans.append(int(r["scan"])); ms.append(float(r["proc_ms"]))
        except (ValueError, KeyError):
            pass
    return np.array(scans), np.array(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "result_*.proc.csv")))
    methods = []
    for f in files:
        name = os.path.basename(f)[len("result_"):-len(".proc.csv")]
        s, ms = read_proc(f)
        if ms.size:
            methods.append((name, s, ms))
    if not methods:
        print(f"plot_proctime: no proc.csv in {a.dir}"); return

    methods.sort(key=lambda x: np.median(x[2]))     # fastest first
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(11, 8),
                                  gridspec_kw={"height_ratios": [2, 1], "hspace": 0.32})

    for i, (name, s, ms) in enumerate(methods):
        c = COLORS[i % len(COLORS)]
        ax.plot(s, ms, lw=0.8, color=c, alpha=0.85,
                label=f"{name}  (med {np.median(ms):.0f} / p95 {np.percentile(ms,95):.0f} ms)")
    ax.axhline(100, color="k", ls="--", lw=0.7, alpha=0.5)   # 10 Hz real-time budget
    ax.text(0.995, 100, " 100 ms (10 Hz)", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="k", alpha=0.6)
    ax.set_ylabel("processing time [ms]"); ax.set_xlabel("scan index")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title(a.title or "Per-scan processing time — all methods", fontsize=12)
    top = max(np.percentile(ms, 99) for _, _, ms in methods)
    ax.set_ylim(0, top * 1.1)

    # bar chart: mean / p95 / max per method
    names = [m[0] for m in methods]
    x = np.arange(len(names)); w = 0.27
    mean = [m[2].mean() for m in methods]
    p95 = [np.percentile(m[2], 95) for m in methods]
    mx = [m[2].max() for m in methods]
    axb.bar(x - w, mean, w, label="mean", color="#2ca02c")
    axb.bar(x, p95, w, label="p95", color="#ff7f0e")
    axb.bar(x + w, mx, w, label="max", color="#d62728")
    axb.axhline(100, color="k", ls="--", lw=0.7, alpha=0.5)
    axb.set_xticks(x); axb.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axb.set_ylabel("proc time [ms]"); axb.grid(alpha=0.3, axis="y"); axb.legend(fontsize=8)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"Saved proc-time plot -> {a.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
