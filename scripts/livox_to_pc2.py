#!/usr/bin/env python3.10
"""
Convert livox_ros_driver2/msg/CustomMsg  →  sensor_msgs/msg/PointCloud2

Subscribes : /avia/livox/lidar  (CustomMsg)
Publishes  : /avia/livox/points (PointCloud2, fields: x y z intensity timestamp)

Per-point timestamps are written as uint64 nanoseconds
(timebase + offset_time) so that ros_lio.py's deskewing via
pointcloud2_to_xyz_i_stamps() works without modification.

Only returns with tag & 0x30 == 0x00 (strongest) or 0x10 (second) are
forwarded; noise/specular returns (0x20, 0x30) are dropped.
"""

import struct
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

try:
    from livox_ros_driver2.msg import CustomMsg
except ImportError as e:
    raise SystemExit(
        "livox_ros_driver2 not found — source the livox workspace first.\n"
        f"  {e}"
    )

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# PointCloud2 point layout (24 bytes per point):
#   x         float32  offset  0
#   y         float32  offset  4
#   z         float32  offset  8
#   intensity float32  offset 12
#   timestamp uint64   offset 16   (absolute nanoseconds)
_POINT_STEP = 24
_FIELDS = [
    PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="timestamp", offset=16, datatype=8,  count=1),  # 8 = UINT64 (ros_lio convention)
]
_PACK_FMT = "<ffffQ"   # little-endian: 4×float32, 1×uint64 = 24 bytes
_VALID_TAGS = {0x00, 0x10}


class LivoxToPC2(Node):
    def __init__(self):
        super().__init__("livox_to_pc2")

        in_topic  = self.declare_parameter("input_topic",  "/avia/livox/lidar").value
        out_topic = self.declare_parameter("output_topic", "/avia/livox/points").value

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(
            CustomMsg, in_topic, self._cb, sensor_qos)
        self.pub = self.create_publisher(PointCloud2, out_topic, 10)

        self.get_logger().info(
            f"livox_to_pc2: {in_topic} → {out_topic}  "
            f"(fields: x y z intensity timestamp[uint64 ns])"
        )

    def _cb(self, msg: CustomMsg) -> None:
        timebase: int = int(msg.timebase)
        rows = []
        for p in msg.points:
            if (p.tag & 0x30) not in _VALID_TAGS:
                continue
            abs_ns = timebase + int(p.offset_time)
            rows.append(struct.pack(
                _PACK_FMT,
                p.x, p.y, p.z,
                float(p.reflectivity),
                abs_ns,
            ))
        if not rows:
            return

        data = b"".join(rows)
        n_pts = len(rows)

        pc2_msg = PointCloud2()
        pc2_msg.header         = msg.header
        pc2_msg.height         = 1
        pc2_msg.width          = n_pts
        pc2_msg.fields         = _FIELDS
        pc2_msg.is_bigendian   = False
        pc2_msg.point_step     = _POINT_STEP
        pc2_msg.row_step       = _POINT_STEP * n_pts
        pc2_msg.data           = data
        pc2_msg.is_dense       = True
        self.pub.publish(pc2_msg)


def main():
    rclpy.init()
    node = LivoxToPC2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
