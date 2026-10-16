"""
Extract a stable-transfer Gazebo bag into a compact .npz (run INSIDE the container):
    python3 /work/scripts/extract_bag.py /work/bags/stable_dave_20km_run2

Writes <bag>/../<bag_name>_extracted.npz with:
  metrics (T, F) + metric_names, metrics_wall (receive time, s)
  action (N, 2) + action_wall, obs_raw (N, 27) + obs_wall
  gps_sim, gps_wall, gps_lat, gps_lon   (downsampled to ~1 Hz of sim time)
  imu_sim, imu_yaw, imu_yawrate         (downsampled to ~1 Hz of sim time)
"""
import math
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32MultiArray

bag = Path(sys.argv[1])
reader = rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"))

out = {k: [] for k in ["metrics", "metrics_wall", "action", "action_wall", "obs", "obs_wall",
                       "gps_sim", "gps_wall", "gps_lat", "gps_lon", "imu_sim", "imu_yaw", "imu_yawrate"]}
names = None
next_gps = next_imu = -1.0
n = 0
while reader.has_next():
    topic, data, t_ns = reader.read_next()
    n += 1
    wall = t_ns * 1e-9
    if topic == "/asv_stable/metrics":
        m = deserialize_message(data, Float32MultiArray)
        names = m.layout.dim[0].label
        out["metrics"].append(list(m.data))
        out["metrics_wall"].append(wall)
    elif topic == "/asv_stable/action":
        m = deserialize_message(data, Float32MultiArray)
        out["action"].append(list(m.data))
        out["action_wall"].append(wall)
    elif topic == "/asv_stable/obs_raw":
        m = deserialize_message(data, Float32MultiArray)
        out["obs"].append(list(m.data))
        out["obs_wall"].append(wall)
    elif topic == "/wamv/sensors/gps/gps/fix":
        if wall < next_gps:          # cheap wall-clock pre-filter before deserializing
            continue
        m = deserialize_message(data, NavSatFix)
        sim = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        out["gps_sim"].append(sim)
        out["gps_wall"].append(wall)
        out["gps_lat"].append(m.latitude)
        out["gps_lon"].append(m.longitude)
        next_gps = wall + 0.5
    elif topic == "/wamv/sensors/imu/imu/data":
        if wall < next_imu:
            continue
        m = deserialize_message(data, Imu)
        q = m.orientation
        out["imu_sim"].append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        out["imu_yaw"].append(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        out["imu_yawrate"].append(m.angular_velocity.z)
        next_imu = wall + 0.5
    if n % 2_000_000 == 0:
        print(f"  {n:,} messages read", flush=True)

dst = bag.parent / f"{bag.name}_extracted.npz"
np.savez(dst, metric_names=np.array(names.split(",")), **{k: np.asarray(v) for k, v in out.items()})
print(f"{n:,} messages -> {dst}")
for k, v in out.items():
    print(f"  {k:14s} {np.asarray(v).shape}")
