#!/usr/bin/env python3
"""
live_viz_bridge.py — live trajectory + LiDAR + map bridge for the web bench.

Subscribes to the regnonrep LIO node's world-frame outputs:
  * /lio/odom         (nav_msgs/Odometry)   → live trajectory + current pose
  * /lio/cloud_world  (sensor_msgs/PointCloud2, world frame, deskewed)
                                            → current scan + accumulated MAP
and writes a compact JSON snapshot (trajectory polyline, decimated map, current
scan, pose, bounds) to a shared-memory file that bench_web.py serves at /api/live
for a top-down canvas view.  No RViz, no ROS on the web side.

Multi-core by design so it keeps up with fast bag playback:
  * process 1 — this rclpy node: receive + decode clouds, enqueue (light).
  * process 2 — accumulator: owns the map/trajectory, writes snapshots.
  * a worker Pool (LIVE_WORKERS, default 3) voxel-hashes scans in parallel.

A new sequence is detected automatically from an odom timestamp reset/gap, which
clears the map + trajectory so runs don't overlay each other.

    LIVE_SNAP=/dev/shm/regnonrep_live.json python3 live_viz_bridge.py \
        --map-voxel 0.2 --workers 3
"""
import argparse
import json
import os
import struct
import tempfile
import time
from multiprocessing import Process, Queue, Pool, cpu_count

import numpy as np

# ---- snapshot location (shared with bench_web.py) ----------------------------
def default_snap():
    if os.environ.get("LIVE_SNAP"):
        return os.environ["LIVE_SNAP"]
    shm = "/dev/shm"
    base = shm if os.path.isdir(shm) and os.access(shm, os.W_OK) else tempfile.gettempdir()
    return os.path.join(base, "regnonrep_live.json")


# ---- voxel packing (vectorised, collision-free over ±200 km at 0.2 m) --------
_BITS = 21
_OFF = 1 << (_BITS - 1)          # index offset → non-negative
_MASK = (1 << _BITS) - 1


def _pack(ix, iy, iz):
    return ((ix + _OFF) << (2 * _BITS)) | ((iy + _OFF) << _BITS) | (iz + _OFF)


def _unpack(keys):
    iz = (keys & _MASK) - _OFF
    iy = ((keys >> _BITS) & _MASK) - _OFF
    ix = ((keys >> (2 * _BITS)) & _MASK) - _OFF
    return ix, iy, iz


def voxelize(args):
    """Worker: unique occupied-voxel keys of one world-frame scan. Runs in a Pool
    process so several scans hash in parallel under fast playback."""
    xyz, vox = args
    q = np.floor(xyz / vox).astype(np.int64)
    keys = _pack(q[:, 0], q[:, 1], q[:, 2])
    return np.unique(keys)


# ---- accumulator process: owns the map + trajectory, writes snapshots --------
def accumulator(q, snap_path, vox, workers, caps):
    map_cap, traj_cap, scan_cap = caps
    pool = Pool(workers) if workers > 1 else None
    map_set = set()
    map_chunks = []                 # list of (N,3) float32 arrays of NEW voxel centres
    map_n = 0
    traj = []                       # [x,y,z] poses
    scan = None                     # last world scan (subsampled) for display
    pose = [0.0, 0.0, 0.0]
    seg = 0
    last_t = None
    last_write = 0.0
    tmp_path = snap_path + ".tmp"

    def reset():
        nonlocal map_set, map_chunks, map_n, traj, scan, seg
        map_set = set(); map_chunks = []; map_n = 0; traj = []; scan = None
        seg += 1

    def add_scans(clouds):
        nonlocal map_n
        if not clouds:
            return
        if pool is not None and len(clouds) > 1:
            uniqs = pool.map(voxelize, [(c, vox) for c in clouds])
        else:
            uniqs = [voxelize((c, vox)) for c in clouds]
        for uk in uniqs:
            if map_n >= map_cap:
                break
            new = [k for k in uk.tolist() if k not in map_set]   # C-set membership: fast
            if not new:
                continue
            map_set.update(new)
            ix, iy, iz = _unpack(np.array(new, dtype=np.int64))
            centres = (np.stack([ix, iy, iz], axis=1).astype(np.float32) + 0.5) * vox
            map_chunks.append(centres)
            map_n += centres.shape[0]

    def write_snapshot():
        # decimate map for display
        if map_chunks:
            allc = np.concatenate(map_chunks, axis=0)
        else:
            allc = np.zeros((0, 3), np.float32)
        if allc.shape[0] > 9000:
            allc = allc[np.linspace(0, allc.shape[0] - 1, 9000).astype(int)]
        tr = np.asarray(traj, np.float32)
        if tr.ndim != 2 or tr.shape[1] < 2:            # empty trajectory → well-shaped
            tr = np.zeros((0, 3), np.float32)
        elif tr.shape[0] > traj_cap:
            tr = tr[np.linspace(0, tr.shape[0] - 1, traj_cap).astype(int)]
        sc = scan if (scan is not None and scan.ndim == 2) else np.zeros((0, 3), np.float32)
        # bounds from map ∪ trajectory (fall back to scan)
        pts_for_bounds = [a for a in (allc, tr, sc) if a.shape[0]]
        if pts_for_bounds:
            allb = np.concatenate([p[:, :2] for p in pts_for_bounds], axis=0)
            xmin, ymin = allb.min(axis=0); xmax, ymax = allb.max(axis=0)
        else:
            xmin = ymin = -1.0; xmax = ymax = 1.0
        snap = {
            "seg": seg, "t": round(last_t or 0.0, 3),
            "pose": [round(float(v), 3) for v in pose],
            "n_traj": len(traj), "n_map": map_n, "n_scan": int(sc.shape[0]),
            "bounds": [float(xmin), float(xmax), float(ymin), float(ymax)],
            "traj": [[round(float(x), 2), round(float(y), 2)] for x, y in tr[:, :2]],
            "map": [[round(float(x), 2), round(float(y), 2), round(float(z), 2)]
                    for x, y, z in allc],
            "scan": [[round(float(x), 2), round(float(y), 2), round(float(z), 2)]
                     for x, y, z in sc],
        }
        try:
            with open(tmp_path, "w") as f:
                json.dump(snap, f)
            os.replace(tmp_path, snap_path)
        except OSError:
            pass

    # prime an empty snapshot so the UI shows "waiting"
    write_snapshot()
    while True:
        item = q.get()
        if item is None:                       # shutdown
            break
        kind, stamp, payload = item
        # segment reset on timestamp regression / large gap (new sequence)
        if stamp is not None:
            if last_t is not None and (stamp < last_t - 0.5 or stamp - last_t > 5.0):
                reset()
            last_t = stamp
        clouds = []
        if kind == "odom":
            pose = payload
            traj.append(payload)
        elif kind == "cloud":
            clouds.append(payload)
            scan = payload if payload.shape[0] <= scan_cap else \
                payload[np.linspace(0, payload.shape[0] - 1, scan_cap).astype(int)]
        # opportunistically drain a burst so the Pool hashes several scans at once
        for _ in range(16):
            try:
                nxt = q.get_nowait()
            except Exception:
                break
            if nxt is None:
                q.put(None); break
            k2, s2, p2 = nxt
            if s2 is not None:
                if last_t is not None and (s2 < last_t - 0.5 or s2 - last_t > 5.0):
                    reset(); clouds = []
                last_t = s2
            if k2 == "odom":
                pose = p2; traj.append(p2)
            elif k2 == "cloud":
                clouds.append(p2)
                scan = p2 if p2.shape[0] <= scan_cap else \
                    p2[np.linspace(0, p2.shape[0] - 1, scan_cap).astype(int)]
        add_scans(clouds)
        now = time.time()
        if now - last_write >= 0.25:
            write_snapshot(); last_write = now
    if pool is not None:
        pool.close(); pool.join()
    write_snapshot()


# ---- ROS node: decode + enqueue (kept light) ---------------------------------
def run_ros(q, decim):
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2

    def decode_xyz(msg):
        n = msg.width * msg.height
        if n == 0:
            return None
        fo = {f.name: f.offset for f in msg.fields}
        if not all(k in fo for k in ("x", "y", "z")):
            return None
        b = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
        def col(k):
            o = fo[k]
            return b[:, o:o + 4].copy().ravel().view(np.float32).astype(np.float32)
        xyz = np.column_stack([col("x"), col("y"), col("z")])
        return xyz[np.isfinite(xyz).all(axis=1)]

    class Bridge(Node):
        def __init__(self):
            super().__init__("live_viz_bridge")
            self._i = 0
            self.create_subscription(Odometry, "/lio/odom", self.on_odom, 50)
            self.create_subscription(PointCloud2, "/lio/cloud_world", self.on_cloud, 10)
            self.get_logger().info("live_viz_bridge: subscribed /lio/odom + /lio/cloud_world")

        def _stamp(self, h):
            return h.stamp.sec + h.stamp.nanosec * 1e-9

        def on_odom(self, m):
            p = m.pose.pose.position
            q.put(("odom", self._stamp(m.header),         # always deliver (complete track)
                   [float(p.x), float(p.y), float(p.z)]))

        def on_cloud(self, m):
            self._i += 1
            if decim > 1 and (self._i % decim):           # optional display decimation
                return
            xyz = decode_xyz(m)
            if xyz is not None and xyz.shape[0]:
                try:                                       # drop (never stall ROS) if backlogged
                    q.put_nowait(("cloud", self._stamp(m.header), xyz))
                except Exception:
                    pass

    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):        # incl. ExternalShutdownException on kill
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default=default_snap())
    ap.add_argument("--map-voxel", type=float, default=0.2)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("LIVE_WORKERS", min(3, max(1, cpu_count() - 2)))))
    ap.add_argument("--decimate", type=int, default=1,
                    help="use every Nth cloud (1 = all)")
    ap.add_argument("--map-cap", type=int, default=400000)
    ap.add_argument("--traj-cap", type=int, default=3000)
    ap.add_argument("--scan-cap", type=int, default=4000)
    a = ap.parse_args()

    q = Queue(maxsize=256)
    # NOT daemon: the accumulator itself spawns a worker Pool, and daemonic
    # processes are not allowed to have children.  bench_web launches us in our
    # own session and kills the whole group, so this stays contained.
    acc = Process(target=accumulator, daemon=False,
                  args=(q, a.snap, a.map_voxel, a.workers,
                        (a.map_cap, a.traj_cap, a.scan_cap)))
    acc.start()
    print(f"live_viz_bridge → {a.snap}  (map_voxel={a.map_voxel} m, workers={a.workers})",
          flush=True)
    try:
        run_ros(q, a.decimate)
    finally:
        q.put(None)
        acc.join(timeout=3)


if __name__ == "__main__":
    main()
