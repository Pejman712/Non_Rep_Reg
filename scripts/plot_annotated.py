#!/usr/bin/env python3
"""
plot_annotated.py — annotated route plot for nrlio-family runs.

Colours the estimated trajectory by registration mode and overlays icons where
ZUPT / motion-clamp / degeneracy fired, plus two companion strips showing the
adaptive voxel size and the gyro-adaptive accumulation depth over the run.  Reads
the per-pose annotation sidecar (result.ann.csv) written by the nrlio node,
matched 1:1 (by row index) with the estimate TUM.

    plot_annotated.py --gt gt.tum --tum result.tum --ann result.ann.csv --out out.png
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

MODE_COLOR = {"p2p": "#2ca02c", "gicp": "#1f77b4",
              "degen": "#d62728", "skip": "#d62728",
              "clamp": "#d62728", "zupt": "#ff7f0e", "other": "#7f7f7f"}


def read_tum(path):
    a = []
    for line in open(path):
        p = line.split()
        if len(p) >= 8:
            try:
                a.append([float(x) for x in p[:8]])
            except ValueError:
                pass
    return np.array(a)


def read_ann(path):
    rows = list(csv.DictReader(open(path)))
    method = [r.get("method", "") for r in rows]
    accum = np.array([float(r.get("accum") or 1) for r in rows])
    vox = np.array([float(r.get("voxel_d") or 0) for r in rows])
    proc = np.array([float(r.get("proc_ms") or 0) for r in rows])   # 0 if old sidecar
    return method, accum, vox, proc


def categorize(m):
    if m.startswith("stationary"):
        return "zupt"
    if "clamp" in m:
        return "clamp"
    if m.startswith("skip_degen"):
        return "skip"
    if "gicp*" in m or m.startswith("gicp→") or m.startswith("gicp->"):
        return "degen"
    if "gicp" in m:
        return "gicp"
    if "p2p" in m:
        return "p2p"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=None)
    ap.add_argument("--tum", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    est = read_tum(a.tum)
    method, accum, vox, proc = read_ann(a.ann)
    n = min(len(est), len(method))
    if n < 2:
        print("annotated plot: too few poses"); return
    est, method = est[:n], method[:n]
    accum, vox, proc = accum[:n], vox[:n], proc[:n]
    cats = [categorize(m) for m in method]
    exy = est[:, 1:3] - est[0, 1:3]     # anchor at own first pose (starts at 0,0)

    fig = plt.figure(figsize=(9, 13))
    gs = fig.add_gridspec(5, 1, height_ratios=[6, 1, 1, 1, 0.6], hspace=0.32)
    ax = fig.add_subplot(gs[0])
    s1 = fig.add_subplot(gs[1]); s2 = fig.add_subplot(gs[2])
    s3 = fig.add_subplot(gs[3]); s4 = fig.add_subplot(gs[4])   # proc-time + mode-timeline

    ref = exy
    if a.gt and Path(a.gt).exists():
        gt = read_tum(a.gt)
        gt_xy = gt[:, 1:3] - gt[0, 1:3]   # GT also anchored at its own first pose (0,0)
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "--", color="steelblue", lw=2.2,
                label=f"ground truth ({len(gt_xy)})", zorder=5)
        ref = gt_xy

    # --- estimate coloured by registration mode (segment i uses cat of point i) ---
    segs = np.stack([exy[:-1], exy[1:]], axis=1)
    seg_colors = [MODE_COLOR.get(c, "#7f7f7f") for c in cats[:-1]]
    ax.add_collection(LineCollection(segs, colors=seg_colors, linewidths=2.0, zorder=6))
    ax.plot(*exy[0], "o", color="black", ms=7, zorder=8)

    # --- event icons ---
    cats_a = np.array(cats)
    z = cats_a == "zupt"
    cl = cats_a == "clamp"
    dg = (cats_a == "degen") | (cats_a == "skip")
    if z.any():
        ax.scatter(exy[z, 0], exy[z, 1], marker="D", s=34, facecolor="#ff7f0e",
                   edgecolor="k", lw=0.4, zorder=9)
    if dg.any():
        ax.scatter(exy[dg, 0], exy[dg, 1], marker="^", s=28, facecolor="#d62728",
                   edgecolor="k", lw=0.3, zorder=9)
    if cl.any():
        ax.scatter(exy[cl, 0], exy[cl, 1], marker="X", s=75, color="magenta",
                   edgecolor="k", lw=0.5, zorder=11)

    legend = [
        Line2D([0], [0], color=MODE_COLOR["p2p"], lw=2, label="point-to-plane"),
        Line2D([0], [0], color=MODE_COLOR["gicp"], lw=2, label="GICP (open)"),
        Line2D([0], [0], color=MODE_COLOR["degen"], lw=2, label="degenerate/fallback"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#ff7f0e",
               markeredgecolor="k", ms=7, label=f"ZUPT hold ({int(z.sum())})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#d62728",
               markeredgecolor="k", ms=8, label=f"degeneracy ({int(dg.sum())})"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="magenta",
               markeredgecolor="k", ms=8, label=f"motion clamp ({int(cl.sum())})"),
    ]
    if a.gt:
        legend.insert(0, Line2D([0], [0], color="steelblue", ls="--", lw=2, label="ground truth"))
    ax.legend(handles=legend, loc="best", fontsize=8, framealpha=0.9)

    # robust square frame centred on the reference, extended for a tracking estimate
    r = float(np.abs(ref).max()) if len(ref) else 1.0
    er = float(np.abs(exy).max())
    if er <= 8.0 * max(r, 1e-6):
        r = max(r, er)
    r = max(r, 1.0) * 1.15
    ax.set_xlim(-r, r); ax.set_ylim(-r, r)
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.35)
    ax.set_title("nrlio route — line colour = registration mode; icons = ZUPT / degeneracy / clamp",
                 fontsize=11)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # --- companion strips: voxel size, accumulation, processing time, mode ---
    idx = np.arange(n)
    s1.plot(idx, vox, color="#8c564b", lw=1.2)
    s1.set_ylabel("voxel d\n[m]", fontsize=8); s1.grid(alpha=0.3); s1.set_xlim(0, n)
    s1.tick_params(labelbottom=False)
    s2.step(idx, accum, where="post", color="#17becf", lw=1.4)
    s2.set_ylabel("accum\n[frames]", fontsize=8); s2.grid(alpha=0.3); s2.set_xlim(0, n)
    s2.tick_params(labelbottom=False)
    if accum.size:
        s2.set_ylim(0.5, max(2.5, accum.max() + 0.5))

    # processing time per scan
    s3.plot(idx, proc, color="#9467bd", lw=1.0)
    if proc.size and proc.max() > 0:
        p95 = float(np.percentile(proc[proc > 0], 95))
        s3.axhline(p95, color="#9467bd", ls=":", lw=0.8, alpha=0.7)
        s3.text(0.995, 0.9, f"p95={p95:.0f}ms mean={proc[proc>0].mean():.0f}",
                transform=s3.transAxes, ha="right", va="top", fontsize=7, color="#9467bd")
    s3.set_ylabel("proc\n[ms]", fontsize=8); s3.grid(alpha=0.3); s3.set_xlim(0, n)
    s3.tick_params(labelbottom=False)

    # mode timeline — where each registration mode / mechanism ran (esp. GICP/nonrep)
    strip_color = {"p2p": MODE_COLOR["p2p"], "gicp": MODE_COLOR["gicp"],
                   "degen": MODE_COLOR["degen"], "skip": "#8c564b",
                   "clamp": "magenta", "zupt": MODE_COLOR["zupt"], "other": "#cccccc"}
    band = np.zeros((1, n, 3))
    for i, c in enumerate(cats):
        h = strip_color.get(c, "#cccccc").lstrip("#")
        band[0, i] = [int(h[j:j+2], 16) / 255 for j in (0, 2, 4)]
    s4.imshow(band, aspect="auto", extent=[0, n, 0, 1], interpolation="nearest")
    s4.set_yticks([]); s4.set_ylabel("mode", fontsize=8); s4.set_xlim(0, n)
    s4.set_xlabel("scan index")
    ng = int(((np.array(cats) == "gicp") | (np.array(cats) == "degen")).sum())
    s4.text(0.995, 0.5, f"GICP/nonrep: {ng} scans", transform=s4.transAxes,
            ha="right", va="center", fontsize=7,
            color="white", bbox=dict(boxstyle="round,pad=0.15", fc="#1f77b4", ec="none"))

    # shade degenerate scans on the value strips
    for i in np.where(dg)[0]:
        s1.axvspan(i - 0.5, i + 0.5, color="#d62728", alpha=0.10, lw=0)
        s2.axvspan(i - 0.5, i + 0.5, color="#d62728", alpha=0.10, lw=0)
        s3.axvspan(i - 0.5, i + 0.5, color="#d62728", alpha=0.10, lw=0)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print(f"Saved annotated plot -> {a.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
