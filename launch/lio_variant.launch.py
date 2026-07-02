from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Generic launch for any lio_base.py variant.
    #   exe : the variant executable, e.g. lio_nonrep_gicp.py
    #   cfg : full path to the dataset config YAML (keyed under `lio_node:`)
    # The node is remapped to the name `lio_node` so the YAML params bind to it
    # regardless of the variant's own NODE_NAME.
    exe = LaunchConfiguration("exe")
    cfg = LaunchConfiguration("cfg")

    return LaunchDescription([
        DeclareLaunchArgument("exe"),
        DeclareLaunchArgument("cfg"),
        Node(
            package="regnonrep",
            executable=exe,
            name="lio_node",
            parameters=[cfg],
            output="screen",
        ),
        Node(
            package="regnonrep",
            executable="odom_to_tum.py",
            name="odom_to_tum_fused",
            parameters=[{
                "odom_to_tum": {
                    "enabled": True,
                    "odom_topic": "/lio/odom",
                    "output_path": "/u/97/habibip1/unix/ros2_ws/src/regnonrep/tum/lio_odom.tum",
                    "flush_every_n": 10,
                    "use_msg_time": True,
                    "append": False,
                }
            }],
            output="screen",
        ),
    ])
