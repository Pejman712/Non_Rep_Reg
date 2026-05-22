#!/usr/bin/env python3
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["tomato", "mediumseagreen", "mediumpurple", "darkorange", "steelblue"]


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
    parser.add_argument("files", nargs="+", type=Path,
                        help="gt.tum [est1.tum ...] output.png")
    parser.add_argument("--duration", type=float, default=0,
                        help="Clip to first N seconds (0 = all)")
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Need at least gt.tum and output.png")
        sys.exit(1)

    output   = args.files[-1]
    gt_path  = args.files[0]
    est_paths = args.files[1:-1]  # everything between gt and output

    gt = read_tum(gt_path)
    if args.duration > 0:
        gt = clip_to_duration(gt, args.duration)
    gt_xy = center(gt[:, 1:3])

    duration_str = f" — first {args.duration:.0f}s" if args.duration > 0 else ""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(gt_xy[:, 0], gt_xy[:, 1],
            linewidth=2.5, linestyle="--", color="steelblue",
            label=f"Ground truth  ({len(gt_xy)} poses)", zorder=10)
    ax.plot(*gt_xy[0], "o", color="steelblue", markersize=8, zorder=11)

    for i, est_path in enumerate(est_paths):
        color = COLORS[i % len(COLORS)]
        label_name = est_path.stem.replace("result_regnonrep_", "").replace("result_regnonrep", "LIO")
        est = read_tum(est_path)
        if args.duration > 0:
            est = clip_to_duration(est, args.duration)
        est_xy = center(est[:, 1:3])
        ax.plot(est_xy[:, 0], est_xy[:, 1],
                linewidth=2, linestyle="-", color=color,
                label=f"{label_name}  ({len(est_xy)} poses)")
        ax.plot(*est_xy[0], "o", color=color, markersize=8)

    ax.set_title(f"Trajectory: Ground Truth vs regnonrep LIO{duration_str}", fontsize=13)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()
    print(f"Saved trajectory plot → {output}")


if __name__ == "__main__":
    main()
