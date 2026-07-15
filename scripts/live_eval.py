#!/usr/bin/env python3
"""
live_eval.py  —  Live odometry vs ground-truth analysis for the regnonrep LIO
                 nodes (designed for ros_lio_v3.py + the Tier indoor1_avia set).

Subscribes to an Odometry topic (default /lio/odom), loads a TUM ground-truth
trajectory, continuously Umeyama-aligns the live estimate to GT, and shows a
self-updating figure:

  A  XY trajectory   — estimated (colour = APE) vs ground truth
  B  APE over time   — translational error vs GT, with median / RMSE lines
  C  Live stats      — #poses, path length, median/RMSE/max APE, end drift

This is the live counterpart of debug_plot.py.  The conf / chi² / motion panels
are omitted because the node only publishes pose on /lio/odom — those would need
the node to publish its diagnostics on a separate topic.

Run alongside the bag (3 terminals, or use run_live_indoor1_avia.sh):
    # 1) the LIO node
    ros2 run regnonrep ros_lio_v3.py
    # 2) this live analysis (GUI — keep in the foreground)
    python3 live_eval.py \
        --gt /…/Tier/Livox_avia/indoor1_avia/indoor1_avia.tum
    # 3) play the dataset
    ros2 bag play /…/Tier/Livox_avia/indoor1_avia/indoor1_avia/indoor1_avia.db3

On exit (close window / Ctrl-C) a PNG is saved next to the GT file.
"""

import argparse
import os
import sys
import threading

import numpy as np
import matplotlib
_has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
matplotlib.use("TkAgg" if _has_display else "Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.colors import Normalize   # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

import rclpy                              # noqa: E402
from rclpy.node import Node               # noqa: E402
from nav_msgs.msg import Odometry         # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (shared with debug_plot.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_tum(path: str) -> np.ndarray:
    """Return (N, 8): [t, x, y, z, qx, qy, qz, qw]."""
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


def interp_tum_at(t_gt: np.ndarray, xyz_gt: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    """Linear-interpolate GT xyz at each query stamp (clamped at the ends).

    Vectorized with np.interp (which already clamps to the edge values outside
    the range) so it stays cheap as the live trajectory grows."""
    return np.column_stack([np.interp(stamps, t_gt, xyz_gt[:, k]) for k in range(3)])


def umeyama_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Best-fit rigid transform (4×4, scale=1) mapping src → dst (Umeyama)."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d) / len(src)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


# ─────────────────────────────────────────────────────────────────────────────
# ROS node: just buffers incoming odometry (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
class OdomBuffer(Node):
    def __init__(self, odom_topic: str):
        super().__init__("live_eval")
        self._lock = threading.Lock()
        self._stamps = []
        self._xyz = []
        self.create_subscription(Odometry, odom_topic, self._cb, 100)
        self.get_logger().info(f"live_eval listening on {odom_topic}")

    def _cb(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        with self._lock:
            self._stamps.append(t)
            self._xyz.append((p.x, p.y, p.z))

    def snapshot(self):
        with self._lock:
            return np.array(self._stamps), np.array(self._xyz)


# ─────────────────────────────────────────────────────────────────────────────
# Live figure
# ─────────────────────────────────────────────────────────────────────────────
class LiveFigure:
    def __init__(self, node: OdomBuffer, tum: np.ndarray, t_offset: float,
                 align_min: int, out_path: str):
        self.node = node
        self.t_gt = tum[:, 0]
        self.xyz_gt = tum[:, 1:4]
        self.gt0 = self.xyz_gt[0].copy()
        self.t_offset = t_offset
        self.align_min = align_min
        self.out_path = out_path

        dark, self.panel_bg, self.text_col, self.grid_col = \
            "#0f0f0f", "#1a1a1a", "#e0e0e0", "#2a2a2a"
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.patch.set_facecolor(dark)
        gs = GridSpec(2, 2, figure=self.fig, hspace=0.30, wspace=0.25,
                      left=0.06, right=0.96, top=0.92, bottom=0.08)
        self.ax_traj = self.fig.add_subplot(gs[0:2, 0])   # A
        self.ax_ape = self.fig.add_subplot(gs[0, 1])      # B
        self.ax_stat = self.fig.add_subplot(gs[1, 1])     # C
        for ax in (self.ax_traj, self.ax_ape, self.ax_stat):
            ax.set_facecolor(self.panel_bg)
            ax.tick_params(colors=self.text_col, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(self.grid_col)
        self._cbar = None
        self._scat = None
        # plot the full GT path once (static reference)
        gx = self.xyz_gt[:, 0] - self.gt0[0]
        gy = self.xyz_gt[:, 1] - self.gt0[1]
        self.ax_traj.plot(gx, gy, color="#4fc3f7", lw=1.4, alpha=0.85,
                          zorder=2, label="Ground truth")
        self.ax_traj.plot(0, 0, "o", color="#a5d6a7", markersize=8, zorder=6)
        self.ax_traj.set_aspect("equal")

    def _compute(self):
        stamps, xyz = self.node.snapshot()
        if len(stamps) < self.align_min:
            return None
        if self.t_offset is None:   # AUTO-anchor est clock to GT start
            self.t_offset = float(self.t_gt[0] - stamps[0])
            print(f"[live_eval] auto t-offset = {self.t_offset:+.3f} "
                  f"(anchored est start {stamps[0]:.3f} -> GT start {self.t_gt[0]:.3f})")
        self._time_sanity(stamps)   # one-time clock-overlap check
        gt_interp = interp_tum_at(self.t_gt, self.xyz_gt, stamps + self.t_offset)
        T = umeyama_align(xyz, gt_interp)
        xyz_al = (T[:3, :3] @ xyz.T).T + T[:3, 3]
        ape = np.linalg.norm(xyz_al - gt_interp, axis=1)
        return stamps, xyz_al, gt_interp, ape

    def _time_sanity(self, stamps: np.ndarray):
        """Print once whether the est stamps overlap the GT time range.  A
        non-overlap (or a huge offset) means GT is being sampled at the wrong
        instants — every GT value gets clamped to an endpoint and APE is
        meaningless.  This is the first thing to rule out when the plot looks
        very wrong."""
        if getattr(self, "_sanity_done", False):
            return
        self._sanity_done = True
        e0, e1 = float(stamps[0] + self.t_offset), float(stamps[-1] + self.t_offset)
        g0, g1 = float(self.t_gt[0]), float(self.t_gt[-1])
        overlap = max(0.0, min(e1, g1) - max(e0, g0))
        print(f"[live_eval] est t=[{e0:.3f},{e1:.3f}]  gt t=[{g0:.3f},{g1:.3f}]  "
              f"offset={self.t_offset:+.3f}  overlap={overlap:.2f}s")
        if overlap <= 0.0:
            print("[live_eval] WARNING: est and GT time ranges DO NOT overlap — "
                  "APE will be garbage. Fix with --t-offset (gt_time = est_time + offset).")

    def update(self, _frame):
        res = self._compute()
        if res is None:
            return
        stamps, xyz_al, gt_interp, ape = res
        t_rel = stamps - stamps[0]

        # ── A: trajectory ───────────────────────────────────────────────
        if self._scat is not None:
            self._scat.remove()
        ex = xyz_al[:, 0] - self.gt0[0]
        ey = xyz_al[:, 1] - self.gt0[1]
        norm = Normalize(vmin=0.0, vmax=max(float(np.percentile(ape, 95)), 0.1))
        cmap = plt.cm.RdYlGn_r
        self._scat = self.ax_traj.scatter(ex, ey, c=ape, cmap=cmap, norm=norm,
                                          s=6, zorder=3)
        if self._cbar is None:
            self._cbar = self.fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap), ax=self.ax_traj,
                fraction=0.035, pad=0.02)
            self._cbar.set_label("APE [m]", color=self.text_col, fontsize=8)
            self._cbar.ax.yaxis.set_tick_params(color=self.text_col, labelsize=7)
            plt.setp(plt.getp(self._cbar.ax.axes, "yticklabels"), color=self.text_col)
        else:
            self._cbar.mappable.set_norm(norm)
        self.ax_traj.set_xlabel("x [m]", color=self.text_col, fontsize=8)
        self.ax_traj.set_ylabel("y [m]", color=self.text_col, fontsize=8)
        self.ax_traj.set_title("XY Trajectory (live)", color=self.text_col,
                               fontsize=10, pad=4)
        self.ax_traj.grid(True, color=self.grid_col, lw=0.5)
        self.ax_traj.legend(fontsize=8, facecolor="#111", edgecolor=self.grid_col,
                            labelcolor=self.text_col, loc="upper left")
        self.ax_traj.relim()
        self.ax_traj.autoscale_view()

        # ── B: APE over time ────────────────────────────────────────────
        self.ax_ape.clear()
        self.ax_ape.set_facecolor(self.panel_bg)
        self.ax_ape.plot(t_rel, ape, color="#80cbc4", lw=0.9)
        med, rmse = float(np.median(ape)), float(np.sqrt(np.mean(ape ** 2)))
        self.ax_ape.axhline(med, color="#ffee58", lw=1, ls="--",
                            label=f"Median {med:.3f} m")
        self.ax_ape.axhline(rmse, color="#ef9a9a", lw=1, ls=":",
                            label=f"RMSE {rmse:.3f} m")
        self.ax_ape.set_title("APE vs Ground Truth", color=self.text_col,
                              fontsize=10, pad=4)
        self.ax_ape.set_xlabel("time [s]", color=self.text_col, fontsize=8)
        self.ax_ape.set_ylabel("APE [m]", color=self.text_col, fontsize=8)
        self.ax_ape.tick_params(colors=self.text_col, labelsize=8)
        self.ax_ape.grid(True, color=self.grid_col, lw=0.4)
        self.ax_ape.legend(fontsize=7, facecolor="#111", edgecolor=self.grid_col,
                           labelcolor=self.text_col)

        # ── C: live stats ───────────────────────────────────────────────
        self.ax_stat.clear()
        self.ax_stat.set_facecolor(self.panel_bg)
        self.ax_stat.axis("off")
        path_len = float(np.sum(np.linalg.norm(np.diff(xyz_al, axis=0), axis=1)))
        end_drift = float(np.linalg.norm(xyz_al[-1] - gt_interp[-1]))
        drift_pct = (end_drift / path_len * 100.0) if path_len > 1e-6 else 0.0
        lines = [
            ("Poses received",  f"{len(stamps)}"),
            ("Elapsed",         f"{t_rel[-1]:.1f} s"),
            ("Path length",     f"{path_len:.2f} m"),
            ("Median APE",      f"{med:.3f} m"),
            ("RMSE APE",        f"{rmse:.3f} m"),
            ("Max APE",         f"{ape.max():.3f} m"),
            ("End drift",       f"{end_drift:.3f} m ({drift_pct:.2f}% of path)"),
        ]
        y = 0.92
        for k, v in lines:
            self.ax_stat.text(0.04, y, k, color="#b0bec5", fontsize=10,
                              transform=self.ax_stat.transAxes)
            self.ax_stat.text(0.55, y, v, color=self.text_col, fontsize=10,
                              transform=self.ax_stat.transAxes)
            y -= 0.13
        self.ax_stat.set_title("Live Stats", color=self.text_col, fontsize=10, pad=4)

        self.fig.suptitle(
            f"Live odom vs GT — {len(stamps)} poses, {t_rel[-1]:.0f}s   "
            f"RMSE {rmse:.3f} m | drift {drift_pct:.2f}%",
            color=self.text_col, fontsize=12, y=0.97)

    def save(self):
        try:
            self.fig.savefig(self.out_path, dpi=150, facecolor="#0f0f0f",
                             bbox_inches="tight")
            print(f"Saved → {self.out_path}")
        except Exception as e:           # noqa: BLE001
            print(f"Could not save figure: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="Ground-truth TUM file")
    ap.add_argument("--odom-topic", default="/lio/odom",
                    help="Odometry topic to subscribe to (default /lio/odom)")
    ap.add_argument("--t-offset", type=float, default=None,
                    help="GT clock offset (gt_time = est_time + offset). "
                         "Default: AUTO-anchor first est pose to GT start — needed "
                         "because the odom uses the sensor clock while the GT uses "
                         "wall-clock epoch (they differ by ~1.6e9 s).")
    ap.add_argument("--align-min", type=int, default=10,
                    help="Min poses before aligning/plotting (default 10)")
    ap.add_argument("--interval", type=int, default=500,
                    help="Redraw interval in ms (default 500; raise to lower CPU)")
    ap.add_argument("--out", default="",
                    help="PNG saved on exit (default: <gt>_live.png)")
    args = ap.parse_args()

    if not os.path.exists(args.gt):
        print(f"ERROR: GT not found: {args.gt}", file=sys.stderr)
        return 1
    tum = load_tum(args.gt)
    out_path = args.out or os.path.splitext(args.gt)[0] + "_live.png"

    rclpy.init()
    node = OdomBuffer(args.odom_topic)
    # spin ROS in a background thread; matplotlib owns the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    live = LiveFigure(node, tum, args.t_offset, args.align_min, out_path)
    # keep a handle so the animation isn't garbage-collected
    _ani = FuncAnimation(live.fig, live.update, interval=args.interval,
                         cache_frame_data=False)
    try:
        if _has_display:
            plt.show()
        else:
            print("No display detected — running headless; "
                  "press Ctrl-C to stop and save a snapshot.")
            spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        live.update(None)   # final refresh with everything received
        live.save()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
