#!/usr/bin/env python3.10
"""
livox_to_velodyne.py  —  republish a Livox-format PointCloud2 as a
Velodyne/VELO16-format PointCloud2 so the velodyne-handler LIO packages
(faster_lio, fast_lio mid360, point_lio mid360, super_lio VELO32, …) can consume
the Tier/iilab Livox bags.

Input  fields (Livox):  x,y,z,intensity, tag,line, (offset_time uint32 ns  |
                        timestamp float64 abs ns)
Output fields (Velodyne): x,y,z (f32), intensity (f32),
                          time (f32, seconds relative to scan start),
                          ring (uint16, from Livox 'line')

Params:
    input_topic   (default /livox/points)
    output_topic  (default /velodyne_points)
    time_field    (auto | offset_time | timestamp)  — usually auto
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# Velodyne-style output point: x,y,z,intensity,time(f32 s), ring(u16)
_OUT_DTYPE = np.dtype({
    "names": ["x", "y", "z", "intensity", "time", "ring"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
    "offsets": [0, 4, 8, 12, 16, 20],
    "itemsize": 22,
})
_OUT_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="time", offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name="ring", offset=20, datatype=PointField.UINT16, count=1),
]


class LivoxToVelodyne(Node):
    def __init__(self):
        super().__init__("livox_to_velodyne")
        self.in_topic = self.declare_parameter("input_topic", "/livox/points").value
        self.out_topic = self.declare_parameter("output_topic", "/velodyne_points").value
        self.time_field = self.declare_parameter("time_field", "auto").value
        # filtering (mirrors lio_base: drop near/far + Livox noise-tag points)
        self.blind = float(self.declare_parameter("blind", 0.5).value)
        self.max_range = float(self.declare_parameter("max_range", 100.0).value)
        self.tag_filter = bool(self.declare_parameter("tag_filter", True).value)
        # RELIABLE + deep queue so we do not silently DROP scans under load
        # (best-effort was dropping ~70% of clouds at bag-rate >0.5, starving the
        # downstream LIO nodes into IMU-only divergence).  Reliable is compatible
        # with the bags' default-reliable publisher.
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=200)
        # publish RELIABLE so it satisfies both reliable subscribers (dlio) and
        # best-effort ones (a reliable publisher is compatible with both).
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=200)
        self.pub = self.create_publisher(PointCloud2, self.out_topic, pub_qos)
        self.create_subscription(PointCloud2, self.in_topic, self.cb, sub_qos)
        self.get_logger().info(
            f"livox_to_velodyne: {self.in_topic} -> {self.out_topic} "
            f"(time_field={self.time_field})")

    def cb(self, msg: PointCloud2):
        n = msg.width * msg.height
        if n == 0:
            return
        fo = {f.name: f for f in msg.fields}
        if not ({"x", "y", "z"} <= set(fo)):
            return
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)

        def colf32(name):
            o = fo[name].offset
            return raw[:, o:o + 4].copy().ravel().view(np.float32).astype(np.float32)

        x = colf32("x"); y = colf32("y"); z = colf32("z")
        intensity = colf32("intensity") if "intensity" in fo else np.zeros(n, np.float32)

        # per-point time → seconds relative to scan start
        tf = self.time_field
        if tf == "auto":
            tf = ("offset_time" if "offset_time" in fo else
                  "timestamp" if "timestamp" in fo else
                  "t" if "t" in fo else None)
        if tf == "offset_time" and "offset_time" in fo:
            o = fo["offset_time"].offset
            t = raw[:, o:o + 4].copy().ravel().view(np.uint32).astype(np.float64) * 1e-9
        elif tf == "timestamp" and "timestamp" in fo:
            o = fo["timestamp"].offset
            ts = raw[:, o:o + 8].copy().ravel().view(np.float64)
            t = (ts - ts.min()) * 1e-9
        elif tf == "t" and "t" in fo:
            o = fo["t"].offset
            t = raw[:, o:o + 4].copy().ravel().view(np.uint32).astype(np.float64) * 1e-9
        else:
            t = np.zeros(n, dtype=np.float64)

        ring = (raw[:, fo["line"].offset].astype(np.uint16) if "line" in fo
                else np.zeros(n, np.uint16))

        # ── filter: finite, blind<range<max_range, drop Livox noise tags ──────
        d2 = x * x + y * y + z * z
        good = np.isfinite(d2) & (d2 > self.blind ** 2) & (d2 < self.max_range ** 2)
        if self.tag_filter and "tag" in fo:
            tag = raw[:, fo["tag"].offset]
            good &= ((tag & 0x30) == 0x10) | ((tag & 0x30) == 0x00)
        ng = int(good.sum())
        if ng < 10:
            return

        out = np.zeros(ng, dtype=_OUT_DTYPE)
        out["x"] = x[good]; out["y"] = y[good]; out["z"] = z[good]
        out["intensity"] = intensity[good]
        out["time"] = t[good].astype(np.float32)
        out["ring"] = ring[good]

        m = PointCloud2()
        m.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        m.height = 1
        m.width = ng
        m.fields = _OUT_FIELDS
        m.is_bigendian = False
        m.point_step = _OUT_DTYPE.itemsize
        m.row_step = m.point_step * ng
        m.is_dense = True
        m.data = out.tobytes()
        self.pub.publish(m)


def main():
    rclpy.init()
    node = LivoxToVelodyne()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
