#!/usr/bin/env python3.10
"""
Plot ground truth vs one or more estimated TUM trajectories.

Usage:
    python3 plot_tum.py <gt.tum> <est1.tum> [<est2.tum> ...] <output.png> [--duration N]
    python3 plot_tum.py <gt.tum> <est.tum> <output.png> [--duration N]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import os

# Use an interactive backend when a display is available, Agg otherwise.
# --no-show forces headless (Agg) so batch runs save PNGs without blocking.
_no_show = "--no-show" in sys.argv
_has_display = (not _no_show) and bool(
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if _has_display:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            _has_display = False
            matplotlib.use("Agg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# 6+ distinct colours, none == the GT colour (steelblue), so up to 6 methods +
# ground truth are all visually separable on the overlay.
COLORS = ["tomato", "mediumseagreen", "mediumpurple", "darkorange", "gold",
          "deeppink", "saddlebrown", "black"]


def read_tum(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                poses.append([float(x) for x in parts[:4]])  # t x y z
            except ValueError:
                continue
    if not poses:
        raise ValueError(f"No valid poses in {path}")
    return np.array(poses)


def clip_to_duration(poses, duration):
    if duration <= 0:
        return poses
    t0 = poses[0, 0]
    return poses[poses[:, 0] <= t0 + duration]


def center(xy):
    return xy - xy[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, default=None,
                        help="Ground-truth TUM file (optional; omit when none available)")
    parser.add_argument("files", nargs="+", type=Path,
                        help="est1.tum [est2.tum ...] output.png")
    parser.add_argument("--duration", type=float, default=0,
                        help="Clip to first N seconds (0 = all)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the PNG only; never open an interactive window")
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Need at least one est.tum and output.png")
        sys.exit(1)

    output    = args.files[-1]
    est_paths = args.files[:-1]

    duration_str = f" — first {args.duration:.0f}s" if args.duration > 0 else ""
    fig, ax = plt.subplots(figsize=(8, 8))

    if args.gt is not None:
        gt = read_tum(args.gt)
        if args.duration > 0:
            gt = clip_to_duration(gt, args.duration)
        gt_xy = center(gt[:, 1:3])
        ax.plot(gt_xy[:, 0], gt_xy[:, 1],
                linewidth=2.5, linestyle="--", color="steelblue",
                label=f"Ground truth  ({len(gt_xy)} poses)", zorder=10)
        ax.plot(*gt_xy[0], "o", color="steelblue", markersize=8, zorder=11)

    series = [gt_xy] if args.gt is not None else []
    for i, est_path in enumerate(est_paths):
        color = COLORS[i % len(COLORS)]
        label_name = est_path.stem.split("__")[0]              # drop __<dataset>_<seq> (overlays)
        label_name = (label_name.replace("result_regnonrep_", "")
                                 .replace("result_regnonrep", "LIO")
                                 .replace("result_", ""))       # external: result_<method>
        est = read_tum(est_path)
        if args.duration > 0:
            est = clip_to_duration(est, args.duration)
        est_xy = center(est[:, 1:3])
        series.append(est_xy)
        ax.plot(est_xy[:, 0], est_xy[:, 1],
                linewidth=2, linestyle="-", color=color,
                label=f"{label_name}  ({len(est_xy)} poses)")
        ax.plot(*est_xy[0], "o", color=color, markersize=8)

    if args.gt is not None:
        ax.set_title(f"Trajectory: Ground Truth vs regnonrep LIO{duration_str}", fontsize=13)
    else:
        ax.set_title(f"Trajectory: regnonrep LIO{duration_str}", fontsize=13)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    # Robust, square view centred on the reference (GT if present, else the
    # first estimate) so a single diverged run cannot shrink the good ones to a
    # dot.  Frame = reference extent, but never smaller than the estimates that
    # stay within ~8x of it (diverged runs simply exit the frame).
    ref = series[0]
    cx, cy = ref[:, 0].mean(), ref[:, 1].mean()
    ref_r = float(np.abs(ref - [cx, cy]).max()) if len(ref) else 1.0
    r = max(ref_r, 1.0)
    for s in series[1:]:
        s_r = float(np.abs(s - [cx, cy]).max())
        if s_r <= 8.0 * ref_r or ref_r < 1e-6:
            r = max(r, s_r)
    r *= 1.15
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    print(f"Saved trajectory plot → {output}")
    if _has_display:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
