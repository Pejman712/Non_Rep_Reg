from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("regnonrep")
    params = os.path.join(pkg_share, "config", "lio_tier_horizen.yaml")

    return LaunchDescription([
        # Convert livox_ros_driver2/CustomMsg → PointCloud2 for the Horizon bag.
        Node(
            package="regnonrep",
            executable="livox_to_pc2.py",
            name="livox_to_pc2",
            parameters=[{
                "input_topic":  "/livox/lidar",
                "output_topic": "/livox/points",
            }],
            output="screen",
        ),
        Node(
            package="regnonrep",
            executable="ros_lio.py",
            name="lio_node",
            parameters=[params],
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
        Node(
            package="regnonrep",
            executable="odom_to_tum.py",
            name="odom_to_tum_imu_only",
            parameters=[{
                "odom_to_tum": {
                    "enabled": True,
                    "odom_topic": "/lio/odom_imu_only",
                    "output_path": "/u/97/habibip1/unix/ros2_ws/src/regnonrep/tum/lio_imuonly.tum",
                    "flush_every_n": 10,
                    "use_msg_time": True,
                    "append": False,
                }
            }],
            output="screen",
        ),
    ])
