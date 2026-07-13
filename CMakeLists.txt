cmake_minimum_required(VERSION 3.8)
project(regnonrep)

# Default to C++17 (safe even if unused)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

# Install Python node
install(
  PROGRAMS
    scripts/ros_lio.py
    scripts/lio_base.py
    scripts/lio_p2p.py
    scripts/lio_nonrep_gicp.py
    scripts/lio_nonrep_ndt.py
    scripts/lio_fused_gated_ndt.py
    scripts/lio_fused_gated_p2p_gicp_ndt.py
    scripts/lio_nonrep_gicp_p2p.py
    scripts/lio_nonrep_gicp_p2p_degen.py
    scripts/lio_nonrep_fused_degen.py
    scripts/lio_gyro_gicp.py
    scripts/lio_gyro_gicp_degen.py
    scripts/lio_gyro_slowrot.py
    scripts/lio_fused_gated.py
    scripts/lio_gyro_gicp_p2p.py
    scripts/lio_degen_reuse.py
    scripts/pose_graph_se2.py
    scripts/lio_pgo.py
    scripts/lio_tsvd.py
    scripts/lio_gen_lio.py
    scripts/lio_gen_lio_tier1.py
    scripts/lio_gen_lio_tier2.py
    scripts/lio_gen_lio_tier3.py
    scripts/lio_gen_lio_tier4.py
    scripts/lio_gen_lio_intensity.py
    scripts/lio_nrlio.py
    scripts/lio_nrlio_optimized.py
    scripts/lio_nrlio_optA.py
    scripts/lio_nrlio_optB.py
    scripts/lio_nrlio_op_den.py
    scripts/lio_nrlio_plus.py
    scripts/map_saver.py
    scripts/mme.py
    scripts/map_report.py
    scripts/degeneracy.py
    scripts/Pctools.py
    scripts/odom_to_tum.py
    scripts/livox_to_pc2.py
    scripts/livox_to_velodyne.py
    scripts/imu_rescale.py
    scripts/debug_plot.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
