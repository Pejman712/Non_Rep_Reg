#!/usr/bin/env python3
"""
Live Matplotlib visualization for nav_msgs/Odometry.

Default topic:
    /lio/odom

Displays:
    - XY trajectory
    - X, Y, Z position versus elapsed time

Run:
    python3 live_odom_plot.py

Optional ROS parameters:
    python3 live_odom_plot.py --ros-args \
        -p odom_topic:=/lio/odom \
        -p max_points:=5000
"""

import threading
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class LiveOdomPlotter(Node):
    def __init__(self) -> None:
        super().__init__("live_odom_plotter")

        self.declare_parameter("odom_topic", "/lio/odom")
        self.declare_parameter("max_points", 5000)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.max_points = max(10, int(self.get_parameter("max_points").value))

        self._lock = threading.Lock()
        self._start_time = None

        self.times = deque(maxlen=self.max_points)
        self.xs = deque(maxlen=self.max_points)
        self.ys = deque(maxlen=self.max_points)
        self.zs = deque(maxlen=self.max_points)

        self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            50,
        )

        self.get_logger().info(f"Plotting odometry from {self.odom_topic}")

    def _odom_callback(self, msg: Odometry) -> None:
        stamp = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1.0e-9
        )

        if self._start_time is None:
            self._start_time = stamp

        elapsed = stamp - self._start_time
        position = msg.pose.pose.position

        with self._lock:
            self.times.append(elapsed)
            self.xs.append(float(position.x))
            self.ys.append(float(position.y))
            self.zs.append(float(position.z))

    def snapshot(self):
        with self._lock:
            return (
                list(self.times),
                list(self.xs),
                list(self.ys),
                list(self.zs),
            )


def main() -> None:
    rclpy.init()
    node = LiveOdomPlotter()

    executor_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )
    executor_thread.start()

    fig = plt.figure(figsize=(11, 5))

    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xyz = fig.add_subplot(1, 2, 2)

    xy_line, = ax_xy.plot([], [], linewidth=1.5)
    current_point, = ax_xy.plot([], [], marker="o", linestyle="None")

    x_line, = ax_xyz.plot([], [], label="x")
    y_line, = ax_xyz.plot([], [], label="y")
    z_line, = ax_xyz.plot([], [], label="z")

    ax_xy.set_title("LIO trajectory")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.grid(True)
    ax_xy.set_aspect("equal", adjustable="datalim")

    ax_xyz.set_title("Position versus time")
    ax_xyz.set_xlabel("Elapsed time [s]")
    ax_xyz.set_ylabel("Position [m]")
    ax_xyz.grid(True)
    ax_xyz.legend()

    status_text = fig.text(
        0.5,
        0.01,
        f"Waiting for {node.odom_topic} ...",
        ha="center",
    )

    def update(_frame):
        times, xs, ys, zs = node.snapshot()

        if not times:
            return (
                xy_line,
                current_point,
                x_line,
                y_line,
                z_line,
                status_text,
            )

        xy_line.set_data(xs, ys)
        current_point.set_data([xs[-1]], [ys[-1]])

        x_line.set_data(times, xs)
        y_line.set_data(times, ys)
        z_line.set_data(times, zs)

        ax_xy.relim()
        ax_xy.autoscale_view()

        ax_xyz.relim()
        ax_xyz.autoscale_view()

        status_text.set_text(
            f"Samples: {len(times)} | "
            f"x={xs[-1]:.3f} m, "
            f"y={ys[-1]:.3f} m, "
            f"z={zs[-1]:.3f} m"
        )

        return (
            xy_line,
            current_point,
            x_line,
            y_line,
            z_line,
            status_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        interval=100,
        blit=False,
        cache_frame_data=False,
    )

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

    try:
        plt.show()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        executor_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
