from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("regnonrep")
    params = os.path.join(pkg_share, "config", "lio_v2.yaml")

    return LaunchDescription([
        Node(
            package="regnonrep",
            executable="ros_lio_v2.py",
            name="lio_node_v2",
            parameters=[params],
            output="screen",
        ),
        Node(
            package="regnonrep",
            executable="odom_to_tum.py",
            name="odom_to_tum_v2_fused",
            parameters=[{
                "odom_to_tum": {
                    "enabled": True,
                    "odom_topic": "/lio_v2/odom",
                    "output_path": "/u/97/habibip1/unix/ros2_ws/src/regnonrep/tum/lio_v2_odom.tum",
                    "flush_every_n": 10,
                    "use_msg_time": True,
                    "append": False,
                }
            }],
            output="screen",
        ),
    ])
