#!/usr/bin/env python3
"""
map_saver.py — accumulate the LIO's world-frame cloud into a voxel map and save
it as a .pcd, for offline map evaluation (JokerJohn/Cloud_Map_Evaluation → MME).

Subscribes to /lio/cloud_world (the deskewed, world-frame scan the regnonrep LIO
publishes on demand), voxel-accumulates it full-resolution (centroid per cell),
and writes a binary .pcd:
  * periodically (checkpoint, overwrite) so a hard SIGKILL loses ≤ one interval,
  * and once more on SIGINT/SIGTERM shutdown.

The map is exactly what the LIO built (same poses, same deskew) — no re-accumulation
from trajectory, no GT needed.  MME measures local map sharpness/consistency, which
is meaningful without a reference map.

    ros2 run regnonrep map_saver.py --ros-args -p out:=/path/map.pcd -p voxel:=0.05
"""
import signal
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

_OFF = 1 << 20
_BITS = 21


def _pack(q):
    return (((q[:, 0] + _OFF) << (2 * _BITS))
            | ((q[:, 1] + _OFF) << _BITS) | (q[:, 2] + _OFF))


def write_pcd_binary(path, pts, intensity=None):
    """Minimal binary PCD — read fine by PCL/Open3D/MapEval.  Writes
    'x y z intensity' when intensity is given, else 'x y z'."""
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    n = pts.shape[0]
    if intensity is not None:
        data = np.empty((n, 4), np.float32)
        data[:, :3] = pts
        data[:, 3] = np.asarray(intensity, np.float32)
        fields, size, typ, count = "x y z intensity", "4 4 4 4", "F F F F", "1 1 1 1"
    else:
        data = pts
        fields, size, typ, count = "x y z", "4 4 4", "F F F", "1 1 1"
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        f"VERSION 0.7\nFIELDS {fields}\nSIZE {size}\nTYPE {typ}\nCOUNT {count}\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())


class MapSaver(Node):
    def __init__(self):
        super().__init__("map_saver")
        self.out = self.declare_parameter("out", "map.pcd").value
        self.voxel = float(self.declare_parameter("voxel", 0.05).value)
        period = float(self.declare_parameter("save_period_s", 15.0).value)
        # key -> [sum_x, sum_y, sum_z, sum_intensity, count]
        self._cells = {}
        self._n_scans = 0
        self._has_intensity = False
        self.create_subscription(PointCloud2, "/lio/cloud_world", self.cb, 10)
        self.create_timer(period, self.save)
        self.get_logger().info(
            f"map_saver: /lio/cloud_world → {self.out}  "
            f"(voxel={self.voxel} m, checkpoint every {period:.0f}s)")

    def cb(self, msg):
        n = msg.width * msg.height
        if n == 0:
            return
        fo = {f.name: f.offset for f in msg.fields}
        if not all(k in fo for k in ("x", "y", "z")):
            return
        b = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
        xyz = np.column_stack([
            b[:, fo[k]:fo[k] + 4].copy().ravel().view(np.float32) for k in ("x", "y", "z")
        ]).astype(np.float64)
        inten = None
        if "intensity" in fo:
            io = fo["intensity"]
            inten = b[:, io:io + 4].copy().ravel().view(np.float32).astype(np.float64)
        mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[mask]
        if xyz.shape[0] == 0:
            return
        if inten is not None:
            inten = inten[mask]
            self._has_intensity = True
        q = np.floor(xyz / self.voxel).astype(np.int64)
        uk, inv = np.unique(_pack(q), return_inverse=True)
        sums = np.zeros((uk.shape[0], 3))
        np.add.at(sums, inv, xyz)
        cnts = np.bincount(inv, minlength=uk.shape[0])
        if inten is not None:
            si = np.zeros(uk.shape[0])
            np.add.at(si, inv, inten)
        cells = self._cells
        for i, k in enumerate(uk.tolist()):
            ii = si[i] if inten is not None else 0.0
            e = cells.get(k)
            if e is None:
                cells[k] = [sums[i, 0], sums[i, 1], sums[i, 2], ii, float(cnts[i])]
            else:
                e[0] += sums[i, 0]; e[1] += sums[i, 1]; e[2] += sums[i, 2]
                e[3] += ii; e[4] += cnts[i]
        self._n_scans += 1

    def points(self):
        if not self._cells:
            return np.zeros((0, 3)), None
        a = np.array(list(self._cells.values()))       # (M, 5): sx sy sz si n
        cnt = a[:, 4:5]
        xyz = a[:, :3] / cnt
        inten = (a[:, 3] / a[:, 4]) if self._has_intensity else None
        return xyz, inten

    def save(self):
        pts, inten = self.points()
        if pts.shape[0] == 0:
            return
        try:
            write_pcd_binary(self.out + ".tmp", pts, inten)
            import os
            os.replace(self.out + ".tmp", self.out)
            self.get_logger().info(
                f"map_saver: saved {pts.shape[0]:,} map pts "
                f"({self._n_scans} scans) → {self.out}")
        except OSError as e:
            self.get_logger().warn(f"map_saver: save failed: {e}")


def main():
    rclpy.init()
    node = MapSaver()
    signal.signal(signal.SIGTERM,
                  lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()                      # final full-map save on shutdown
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
