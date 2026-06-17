#!/usr/bin/env python3.10
"""Offline driver: replay the indoor1_avia db3 through ros_lio_v3's real
callbacks (no ros2 bag play / no rate limiting), then ATE-RMSE vs GT."""
import os, sys, time, sqlite3
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2, Imu

BAG = ("/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/"
       "Tier/Livox_avia/indoor1_avia/indoor1_avia/indoor1_avia.db3")
GT = ("/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/"
      "Tier/Livox_avia/indoor1_avia/indoor1_avia.tum")
CSV = "/tmp/v3_indoor1.csv"
MAX_T = float(sys.argv[1]) if len(sys.argv) > 1 else 1e18  # optional time cap (s)

rclpy.init(args=["--ros-args", "-p", f"debug_csv:={CSV}", "-p", "publish_cloud:=false"])
from ros_lio_v3 import SuperLioV3
node = SuperLioV3()

con = sqlite3.connect(BAG); cur = con.cursor()
topics = {tid: name for tid, name in cur.execute("select id,name from topics")}
rows = cur.execute("select topic_id,timestamp,data from messages order by timestamp").fetchall()
t0 = None
t_start = time.time()
n_imu = n_pc = 0
for tid, ts, data in rows:
    name = topics[tid]
    if t0 is None:
        t0 = ts
    if (ts - t0) * 1e-9 > MAX_T:
        break
    if name == node.imu_topic:
        node.cb_imu(deserialize_message(data, Imu)); n_imu += 1
    elif name == node.lidar_topic:
        node.cb_lidar(deserialize_message(data, PointCloud2)); n_pc += 1
con.close()
node.shutdown()
print(f"replayed imu={n_imu} pc={n_pc} scans_processed={node._scan_counter} "
      f"in {time.time()-t_start:.1f}s wall")

# ---- ATE-RMSE ----
from tune_lio_v2 import ate_rmse_csv
rmse, n, off = ate_rmse_csv(CSV, GT, max_diff=0.05)
print(f"ATE-RMSE={rmse:.4f} m  matched={n}  offset={off:.3f}s")
