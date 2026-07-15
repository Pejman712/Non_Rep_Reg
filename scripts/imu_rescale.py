#!/usr/bin/env python3.10
"""
imu_rescale.py — republish a sensor_msgs/Imu with its linear acceleration rescaled.

The Tier Livox bags (Avia / Horizon) report the built-in IMU acceleration in *g*
(~1.0 at rest), but the external LIO packages (fast_lio, point_lio, super_lio, …)
assume m/s² with gravity ≈ 9.81.  Feeding them g-units makes the filter see a
phantom ~8.8 m/s² and diverge (mostly in Z).  This node multiplies the accel by
`accel_scale` (default 9.81, g→m/s²) and republishes, so every consumer gets
standard m/s² acceleration.  iilab's Xsens is already m/s² (use accel_scale=1.0).

Params:
    input_topic   (default /livox/imu)
    output_topic  (default /bench/imu)
    accel_scale   (default 9.81)   multiply linear_acceleration by this
    gyro_scale    (default 1.0)    multiply angular_velocity by this (usually 1)

Dependency-free (rclpy only).  Reliable QoS + deep queue so it neither drops nor
lets the bag outrun the consumers (matches livox_to_velodyne.py).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu


class ImuRescale(Node):
    def __init__(self):
        super().__init__("imu_rescale")
        self.in_topic = self.declare_parameter("input_topic", "/livox/imu").value
        self.out_topic = self.declare_parameter("output_topic", "/bench/imu").value
        self.accel_scale = float(self.declare_parameter("accel_scale", 9.81).value)
        self.gyro_scale = float(self.declare_parameter("gyro_scale", 1.0).value)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=2000)
        self.pub = self.create_publisher(Imu, self.out_topic, qos)
        self.create_subscription(Imu, self.in_topic, self.cb, qos)
        self.get_logger().info(
            f"imu_rescale: {self.in_topic} -> {self.out_topic} "
            f"(accel x{self.accel_scale}, gyro x{self.gyro_scale})")

    def cb(self, msg: Imu):
        a = msg.linear_acceleration
        a.x *= self.accel_scale
        a.y *= self.accel_scale
        a.z *= self.accel_scale
        if self.gyro_scale != 1.0:
            g = msg.angular_velocity
            g.x *= self.gyro_scale
            g.y *= self.gyro_scale
            g.z *= self.gyro_scale
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ImuRescale()
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
