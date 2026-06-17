#!/usr/bin/env python3
"""
debug_plot.py  —  Per-scan diagnostic plot for regnonrep LIO nodes.

Usage:
    python3 debug_plot.py <debug.csv> [--gt <gt.tum>] [--out <output.png>] [--title <text>]

The debug CSV is written by the LIO node when you pass:
    -p debug_csv:=/path/to/debug.csv

The GT TUM file lives next to the bag (e.g. indoor1_avia/indoor1_avia.tum).
It is automatically detected if the CSV and TUM share the same parent directory
structure; pass --gt explicitly otherwise.

Panels produced
───────────────
  A  XY trajectory  (estimated, GT if available; red=rejected, orange=low-conf)
  B  APE over time  (translational, vs GT; only when GT provided)
  C  GICP confidence over time  (red dashed = min threshold)
  D  chi² over time, log scale  (red dashed = rejection gate)
  E  Motion classification timeline
  F  Summary stats table

Exit code 0 on success, 1 if the CSV is missing or empty.
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
_has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
matplotlib.use("TkAgg" if _has_display else "Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_debug_csv(path: str) -> dict:
    """Return dict of column_name → np.ndarray."""
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split(","))
    if not rows:
        raise ValueError(f"Debug CSV is empty: {path}")
    arr = np.array(rows, dtype=object)
    out = {}
    for i, col in enumerate(header):
        col = col.strip()
        try:
            out[col] = arr[:, i].astype(float)
        except ValueError:
            out[col] = arr[:, i]   # keep as strings (e.g. 'motion')
    return out


def load_tum(path: str) -> np.ndarray:
    """Return (N, 8) array: [t, x, y, z, qx, qy, qz, qw]."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                rows.append([float(v) for v in parts[:8]])
    if not rows:
        raise ValueError(f"TUM file is empty: {path}")
    return np.array(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def interp_tum_at(tum: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    """
    For each query stamp, return the interpolated [x, y, z] from the TUM trajectory.
    Uses linear interpolation; queries outside the TUM time range are clamped.
    """
    t_gt = tum[:, 0]
    xyz_gt = tum[:, 1:4]
    out = np.zeros((len(stamps), 3))
    for i, t in enumerate(stamps):
        idx = np.searchsorted(t_gt, t)
        if idx == 0:
            out[i] = xyz_gt[0]
        elif idx >= len(t_gt):
            out[i] = xyz_gt[-1]
        else:
            t0, t1 = t_gt[idx - 1], t_gt[idx]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            out[i] = (1 - alpha) * xyz_gt[idx - 1] + alpha * xyz_gt[idx]
    return out


def umeyama_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Compute the best-fit rigid transform T (4×4) that maps src → dst
    using the Umeyama algorithm (scale=1).
    src, dst: (N, 3)
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = src_c.T @ dst_c / len(src)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4×4 T to (N, 3) array."""
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Path to debug CSV written by the LIO node")
    ap.add_argument("--gt",    default="", help="Ground-truth TUM file")
    ap.add_argument("--out",   default="", help="Output PNG path (default: <csv>.png)")
    ap.add_argument("--title", default="", help="Figure title prefix")
    ap.add_argument("--t-offset", dest="t_offset", type=float, default=None,
                    help="GT clock offset (gt_time = est_time + offset); default: first-pose anchor")
    args = ap.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    # ---- Auto-detect GT TUM (same dir, same stem as parent dir) ----
    gt_path = args.gt
    if not gt_path:
        csv_dir   = Path(csv_path).parent
        # e.g. /…/indoor1_avia/debug.csv → look for /…/indoor1_avia/indoor1_avia.tum
        seq_name  = csv_dir.name
        candidate = csv_dir / f"{seq_name}.tum"
        if candidate.exists():
            gt_path = str(candidate)
            print(f"[auto GT] {gt_path}")

    out_path = args.out or str(Path(csv_path).with_suffix(".png"))

    # ---- Load data ----
    d = load_debug_csv(csv_path)
    n = len(d["scan_num"])
    stamps   = d["stamp"]
    x, y, z  = d["x"], d["y"], d["z"]
    yaw      = d["yaw"]
    imu_x, imu_y = d["imu_x"], d["imu_y"]
    gicp_conf = d["gicp_conf"]
    chi2      = d["chi2"]
    accepted  = d["accepted"]   # -1=no GICP, 0=rejected, 1=accepted
    motion    = d["motion"]     # string array
    n_map_vox = d["n_map_vox"]
    n_pts     = d["n_scan_pts"]

    rejected_mask   = accepted == 0
    low_conf_mask   = (~np.isnan(gicp_conf)) & (gicp_conf < 0.25) & (accepted == 1)
    accepted_mask   = accepted == 1
    no_gicp_mask    = accepted == -1

    # ---- GT alignment ----
    has_gt   = bool(gt_path and os.path.exists(gt_path))
    ape      = None
    gt_xy    = None
    xyz_est  = np.column_stack([x, y, z])

    if has_gt:
        tum = load_tum(gt_path)
        t_off = (args.t_offset if args.t_offset is not None
                 else tum[0, 0] - stamps[0])   # GT hardware time → bag time
        stamps_aligned = stamps + t_off
        gt_interp = interp_tum_at(tum, stamps_aligned)

        # SE3 alignment (Umeyama) — match estimated to GT
        T_align  = umeyama_align(xyz_est, gt_interp)
        xyz_al   = apply_transform(T_align, xyz_est)
        ape      = np.linalg.norm(xyz_al - gt_interp, axis=1)
        gt_xy    = tum[:, 1:3] - tum[0, 1:3]   # GT centred at origin for XY plot
        xyz_est_plot = xyz_al                    # plot aligned estimate
    else:
        xyz_est_plot = xyz_est

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#0f0f0f")
    dark = "#0f0f0f"
    panel_bg = "#1a1a1a"
    text_col = "#e0e0e0"
    grid_col = "#2a2a2a"

    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35,
                  left=0.06, right=0.97, top=0.91, bottom=0.06)

    ax_traj = fig.add_subplot(gs[0:2, 0:2])   # A – XY trajectory (large)
    ax_ape  = fig.add_subplot(gs[0,   2])      # B – APE over time
    ax_conf = fig.add_subplot(gs[1,   2])      # C – GICP confidence
    ax_chi2 = fig.add_subplot(gs[2,   0])      # D – chi² (log)
    ax_mot  = fig.add_subplot(gs[2,   1])      # E – motion timeline
    ax_stat = fig.add_subplot(gs[2,   2])      # F – stats table

    for ax in [ax_traj, ax_ape, ax_conf, ax_chi2, ax_mot, ax_stat]:
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=text_col, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_col)

    t_rel = stamps - stamps[0]   # seconds since start

    # ── A: XY trajectory ─────────────────────────────────────────────────────
    ex = xyz_est_plot[:, 0] - xyz_est_plot[0, 0]
    ey = xyz_est_plot[:, 1] - xyz_est_plot[0, 1]

    if has_gt and ape is not None:
        cmap  = plt.cm.RdYlGn_r
        norm  = Normalize(vmin=0, vmax=max(float(np.percentile(ape, 95)), 0.1))
        sc    = ax_traj.scatter(ex, ey, c=ape, cmap=cmap, norm=norm, s=5,
                                zorder=3, label="Estimated (colour=APE)")
        cbar  = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax_traj,
                             fraction=0.035, pad=0.02)
        cbar.set_label("APE [m]", color=text_col, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=text_col, labelsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=text_col)
        ax_traj.plot(gt_xy[:, 0], gt_xy[:, 1], color="#4fc3f7", lw=1.5,
                     zorder=2, label="Ground truth", alpha=0.85)
    else:
        ax_traj.plot(ex, ey, color="#80cbc4", lw=1.2, zorder=2, label="Estimated")

    # Rejected scans
    if rejected_mask.any():
        ax_traj.scatter(ex[rejected_mask], ey[rejected_mask],
                        color="#ef5350", s=30, zorder=5, marker="x",
                        linewidths=1.5, label=f"Rejected ({rejected_mask.sum()})")
    if low_conf_mask.any():
        ax_traj.scatter(ex[low_conf_mask], ey[low_conf_mask],
                        color="#ffa726", s=20, zorder=4, marker="^",
                        label=f"Low conf ({low_conf_mask.sum()})")

    # Start marker
    ax_traj.plot(0, 0, "o", color="#a5d6a7", markersize=8, zorder=6)
    ax_traj.set_xlabel("x [m]", color=text_col, fontsize=8)
    ax_traj.set_ylabel("y [m]", color=text_col, fontsize=8)
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, color=grid_col, lw=0.5)
    leg = ax_traj.legend(fontsize=7, facecolor="#111", edgecolor=grid_col,
                          labelcolor=text_col, loc="upper left")
    ax_traj.set_title("XY Trajectory", color=text_col, fontsize=9, pad=4)

    # ── B: APE over time ─────────────────────────────────────────────────────
    if has_gt and ape is not None:
        ax_ape.plot(t_rel, ape, color="#80cbc4", lw=0.9, zorder=3)
        med_ape = float(np.median(ape))
        p90_ape = float(np.percentile(ape, 90))
        ax_ape.axhline(med_ape, color="#ffee58", lw=1, ls="--",
                       label=f"Median {med_ape:.3f}m")
        ax_ape.axhline(p90_ape, color="#ef9a9a", lw=1, ls=":",
                       label=f"90th pct {p90_ape:.3f}m")
        if rejected_mask.any():
            ax_ape.vlines(t_rel[rejected_mask], 0, ape[rejected_mask],
                          color="#ef5350", lw=0.6, alpha=0.6, zorder=2)
        ax_ape.legend(fontsize=6, facecolor="#111", edgecolor=grid_col,
                      labelcolor=text_col)
        ax_ape.set_title("APE vs Ground Truth", color=text_col, fontsize=9, pad=4)
        ax_ape.set_ylabel("ATE [m]", color=text_col, fontsize=7)
    else:
        ax_ape.text(0.5, 0.5, "No ground truth", transform=ax_ape.transAxes,
                    ha="center", va="center", color="#888", fontsize=11)
        ax_ape.set_title("APE vs Ground Truth", color=text_col, fontsize=9, pad=4)
    ax_ape.set_xlabel("time [s]", color=text_col, fontsize=7)
    ax_ape.grid(True, color=grid_col, lw=0.4)

    # ── C: GICP confidence ───────────────────────────────────────────────────
    gicp_valid = ~np.isnan(gicp_conf)
    if gicp_valid.any():
        colors_conf = np.where(accepted[gicp_valid] == 0, "#ef5350",
                      np.where(gicp_conf[gicp_valid] < 0.25, "#ffa726", "#80cbc4"))
        ax_conf.bar(t_rel[gicp_valid], gicp_conf[gicp_valid],
                    width=(t_rel[-1] - t_rel[0]) / max(gicp_valid.sum(), 1) * 0.8,
                    color=colors_conf, zorder=3, linewidth=0)
    ax_conf.axhline(0.25, color="#ef5350", lw=1.2, ls="--", label="Min conf 0.25")
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_title("GICP Confidence", color=text_col, fontsize=9, pad=4)
    ax_conf.set_ylabel("confidence", color=text_col, fontsize=7)
    ax_conf.set_xlabel("time [s]", color=text_col, fontsize=7)
    ax_conf.legend(fontsize=6, facecolor="#111", edgecolor=grid_col, labelcolor=text_col)
    ax_conf.grid(True, color=grid_col, lw=0.4)

    # ── D: chi² (log scale) ──────────────────────────────────────────────────
    chi2_valid = ~np.isnan(chi2)
    if chi2_valid.any():
        c_acc = np.where(accepted[chi2_valid] == 0, "#ef5350", "#80cbc4")
        ax_chi2.scatter(t_rel[chi2_valid], np.maximum(chi2[chi2_valid], 1e-3),
                        c=c_acc, s=6, zorder=3, linewidths=0)
    ax_chi2.axhline(22.46, color="#ef5350", lw=1.2, ls="--", label="Gate 22.46")
    ax_chi2.set_yscale("log")
    ax_chi2.set_title("Chi² Gating", color=text_col, fontsize=9, pad=4)
    ax_chi2.set_ylabel("chi²", color=text_col, fontsize=7)
    ax_chi2.set_xlabel("time [s]", color=text_col, fontsize=7)
    ax_chi2.legend(fontsize=6, facecolor="#111", edgecolor=grid_col, labelcolor=text_col)
    ax_chi2.grid(True, color=grid_col, lw=0.4, which="both")

    # ── E: Motion timeline ───────────────────────────────────────────────────
    motion_colors = {
        "stationary": "#546e7a",
        "smooth":     "#66bb6a",
        "erratic":    "#ef5350",
        "combined":   "#ffa726",
        "variable":   "#ce93d8",
        "unknown":    "#424242",
    }
    dt = np.diff(t_rel, append=t_rel[-1] - t_rel[-2] if len(t_rel) > 1 else 1.0)
    for state, col in motion_colors.items():
        mask = motion == state
        if mask.any():
            ax_mot.bar(t_rel[mask], np.ones(mask.sum()),
                       width=dt[mask], color=col, label=state,
                       align="edge", linewidth=0)
    # Overlay map voxel count
    ax_mot2 = ax_mot.twinx()
    ax_mot2.plot(t_rel, n_map_vox / 1000, color="#b0bec5", lw=0.8,
                 alpha=0.6, label="Map voxels ×1k")
    ax_mot2.set_ylabel("Map voxels (×1k)", color="#b0bec5", fontsize=6)
    ax_mot2.tick_params(axis="y", colors="#b0bec5", labelsize=6)
    ax_mot2.set_facecolor(panel_bg)
    patches = [mpatches.Patch(color=v, label=k) for k, v in motion_colors.items()
               if (motion == k).any()]
    ax_mot.legend(handles=patches, fontsize=6, facecolor="#111",
                  edgecolor=grid_col, labelcolor=text_col, loc="upper left",
                  ncol=2)
    ax_mot.set_ylim(0, 1.4)
    ax_mot.set_title("Motion State + Map Size", color=text_col, fontsize=9, pad=4)
    ax_mot.set_xlabel("time [s]", color=text_col, fontsize=7)
    ax_mot.set_yticks([])
    ax_mot.grid(True, color=grid_col, lw=0.4, axis="x")

    # ── F: Stats table ───────────────────────────────────────────────────────
    ax_stat.axis("off")
    n_gicp = int((accepted >= 0).sum())
    n_rej  = int(rejected_mask.sum())
    n_low  = int(low_conf_mask.sum())
    n_no_gicp = int(no_gicp_mask.sum())
    rej_rate  = n_rej / max(n_gicp, 1) * 100
    duration  = float(t_rel[-1]) if len(t_rel) > 1 else 0.0

    rows_stat = [
        ("Total scans",        f"{n}"),
        ("Duration",           f"{duration:.1f} s"),
        ("GICP ran",           f"{n_gicp}"),
        ("GICP rejected",      f"{n_rej}  ({rej_rate:.1f}%)"),
        ("Low conf (accepted)",f"{n_low}"),
        ("No GICP (stationary)",f"{n_no_gicp}"),
    ]
    if has_gt and ape is not None:
        rows_stat += [
            ("Median APE",     f"{np.median(ape):.3f} m"),
            ("90th pct APE",   f"{np.percentile(ape, 90):.3f} m"),
            ("Max APE",        f"{ape.max():.3f} m"),
            ("RMSE APE",       f"{float(np.sqrt(np.mean(ape**2))):.3f} m"),
        ]
    if not np.isnan(gicp_conf[~np.isnan(gicp_conf)]).all() and (~np.isnan(gicp_conf)).any():
        valid_conf = gicp_conf[~np.isnan(gicp_conf)]
        rows_stat.append(("Median GICP conf", f"{np.median(valid_conf):.3f}"))
    if not np.isnan(chi2[~np.isnan(chi2)]).all() and (~np.isnan(chi2)).any():
        valid_chi2 = chi2[~np.isnan(chi2) & (chi2 < 1e6)]
        if len(valid_chi2):
            rows_stat.append(("Median chi²",  f"{np.median(valid_chi2):.2f}"))

    table_data = [[r[0], r[1]] for r in rows_stat]
    col_colors = [[panel_bg, panel_bg]] * len(table_data)
    tbl = ax_stat.table(
        cellText=table_data,
        cellLoc="left",
        loc="center",
        cellColours=col_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(panel_bg)
        cell.set_edgecolor(grid_col)
        cell.set_text_props(color=text_col)
        if col == 0:
            cell.set_text_props(color="#b0bec5")
    ax_stat.set_title("Summary", color=text_col, fontsize=9, pad=4)

    # ── Overall title ─────────────────────────────────────────────────────────
    prefix = f"{args.title} — " if args.title else ""
    seq    = Path(csv_path).stem
    gt_lbl = f"  |  GT: {Path(gt_path).name}" if has_gt else "  |  No GT"
    fig.suptitle(f"{prefix}{seq}{gt_lbl}  —  {n} scans, {duration:.0f}s",
                 color=text_col, fontsize=11, y=0.97)
    fig.patch.set_facecolor(dark)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, facecolor=dark, bbox_inches="tight")
    print(f"Saved → {out_path}")
    if _has_display:
        plt.show()
    plt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
