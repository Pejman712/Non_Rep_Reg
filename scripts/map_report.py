#!/usr/bin/env python3
"""
map_report.py — one-shot map report for a saved LIO map (.pcd):
  * computes MME (mean map entropy, no GT) and appends a row to a shared mme.csv,
  * renders 5 views of the map (top / front / side / 2 isometric) on a WHITE
    background, coloured by intensity if present else height.

Uses 2-D orthographic/rotated projections (no matplotlib 3-D backend, which is
unavailable in this env), so it renders reliably headless.

    map_report.py map.pcd --out-dir <plots> --tag nrlio_plus__tier_avia_indoor2_avia \
        --method nrlio_plus --dataset tier_avia --seq indoor2_avia --mme-csv <run>/mme.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mme as _mme

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rotmat(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return rx @ rz


VIEWS = [("top", (0, 1), None), ("front", (0, 2), None), ("side", (1, 2), None),
         ("iso1", None, (45, 25)), ("iso2", None, (-120, 20))]


def dense_mask(xyz, cell=0.5, rel=0.05, min_count=4):
    """Boolean mask of points in DENSE cells.  Voxelise the map; a cell is 'dense'
    if it holds at least T points, T = max(min_count, rel * 99th-pct cell count).
    Real surfaces pack many points per cell (100s); the diffuse noise spray is
    1-2 points per cell, so this cleanly separates structure from spray regardless
    of how many noise points there are or how far they spread.  Self-calibrating:
    a clean map (no spray) keeps ~everything."""
    q = np.floor(xyz / cell).astype(np.int64)
    off = 1 << 20
    keys = ((q[:, 0] + off) << 42) | ((q[:, 1] + off) << 21) | (q[:, 2] + off)
    uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    thr = max(min_count, rel * np.percentile(cnt, 99))
    return (cnt >= thr)[inv]


def render_views(xyz, color, out_dir, tag, cell=0.5, rel=0.05, min_count=4,
                 max_pts=500000, dpi=150):
    n = xyz.shape[0]
    if n > max_pts:
        idx = np.random.default_rng(0).choice(n, max_pts, replace=False)
        xyz, color = xyz[idx], color[idx]
    # focus on the MAIN (dense) structure — drop the diffuse low-density noise
    # spray.  MME uses the full cloud; only the pictures are cropped.
    if rel > 0 and xyz.shape[0] > 500:
        m = dense_mask(xyz, cell, rel, min_count)
        if m.sum() > 100:
            xyz, color = xyz[m], color[m]
    saved = []
    for name, cols, ang in VIEWS:
        if cols is not None:
            X, Y = xyz[:, cols[0]], xyz[:, cols[1]]
        else:
            P = xyz @ _rotmat(*ang).T
            X, Y = P[:, 0], P[:, 1]
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")
        # 'turbo' stays saturated (its light band is narrow) so points read on white
        ax.scatter(X, Y, s=0.5, c=color, cmap="turbo", linewidths=0, marker=".")
        # tight square frame around the (core) data
        cx, cy = (X.min() + X.max()) / 2, (Y.min() + Y.max()) / 2
        half = max(X.max() - X.min(), Y.max() - Y.min(), 1e-3) / 2 * 1.03
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal", "box"); ax.axis("off")
        ax.set_title(f"{tag} — {name}", fontsize=9, color="#333")
        out = os.path.join(out_dir, f"zzmap_{tag}_{name}.png")
        fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
    return saved


def append_mme(csv_path, row):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["method", "dataset", "seq", "points", "used", "mme", "radius"])
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcd")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--seq", default="")
    ap.add_argument("--mme-csv", default="")
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--cell", type=float, default=0.5,
                    help="voxel size [m] for the density crop")
    ap.add_argument("--rel", type=float, default=0.05,
                    help="a cell is kept if count >= rel*P99(counts); higher = "
                         "tighter focus on the densest structure (0=no crop)")
    ap.add_argument("--min-count", type=int, default=4,
                    help="absolute floor on points-per-cell to keep")
    a = ap.parse_args()

    xyz, inten = _mme.read_pcd_xyzi(a.pcd)
    if xyz.shape[0] == 0:
        print("map_report: empty cloud"); return
    os.makedirs(a.out_dir, exist_ok=True)

    mme_val, npts, used = _mme.compute_mme(xyz, radius=a.radius)
    if a.mme_csv:
        append_mme(a.mme_csv, [a.method, a.dataset, a.seq, npts, used,
                               f"{mme_val:.4f}" if mme_val == mme_val else "", a.radius])

    color = inten if inten is not None else xyz[:, 2]     # intensity else height
    render_views(xyz, color, a.out_dir, a.tag, cell=a.cell, rel=a.rel,
                 min_count=a.min_count)
    print(f"map_report: {a.tag}  points={npts:,}  MME={mme_val:.4f}  "
          f"(coloured by {'intensity' if inten is not None else 'height'}, 5 views)")


if __name__ == "__main__":
    main()
