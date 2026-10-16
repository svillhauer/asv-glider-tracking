#!/usr/bin/env bash
# End-to-end headless check with currents (run inside the container):
#   bash /work/scripts/test_current_e2e.sh
# 2 km scale (30 s ticks) so a few ticks run quickly. Compares the WAM-V's velocity over ground,
# minus the water current, with the policy's commanded surge along the heading.
source /opt/ros/jazzy/setup.bash
source /work/vrx_ws/install/setup.bash
source /work/ros_ws/install/setup.bash
export GZ_SIM_RESOURCE_PATH=/work/models:/work/worlds:/work/vrx_ws/install/wamv_description/share:/work/vrx_ws/install/wamv_gazebo/share:/work/vrx_ws/install/vrx_gazebo/share
ros2 launch vrx_gz vrx_environment.launch.py world:=/work/worlds/open_ocean.sdf sim_mode:=full \
    config_file:=/work/config/wamv_spawn.yaml headless:=true > /tmp/vrx.log 2>&1 &
sleep 45
ros2 launch asv_bridge_stable stable_nondim.launch.py sep_m:=2000.0 current_max:=0.08 max_ticks:=8 seed:=907 ${EXTRA_ARGS} \
    > /tmp/stable.log 2>&1 &
python3 - <<'PY'
import math, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray

class C(Node):
    def __init__(s):
        super().__init__("cur_e2e"); s.fix, s.yaw, s.cur, s.act = [], [], None, None
        s.create_subscription(NavSatFix, "/wamv/sensors/gps/gps/fix", lambda m: s.fix.append(
            (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.latitude, m.longitude, s.yaw[-1] if s.yaw else 0.0,
             s.cur, s.act)), 50)
        s.create_subscription(Imu, "/wamv/sensors/imu/imu/data", lambda m: s.yaw.append(math.atan2(
            2 * (m.orientation.w * m.orientation.z), 1 - 2 * m.orientation.z ** 2)), 50)
        s.create_subscription(Vector3, "/asv_stable/water_current", lambda m: setattr(s, "cur", (m.x, m.y)), 10)
        s.create_subscription(Float32MultiArray, "/asv_stable/action", lambda m: setattr(s, "act", tuple(m.data)), 10)

rclpy.init(); n = C(); t0 = time.time()
while time.time() - t0 < 260: rclpy.spin_once(n, timeout_sec=0.1)
f = [x for x in n.fix if x[4] is not None and x[5] is not None]
lat0, lon0 = f[0][1], f[0][2]
t = np.array([x[0] for x in f])
xy = np.array([[(x[2] - lon0) * math.pi / 180 * 6371000 * math.cos(math.radians(lat0)),
                (x[1] - lat0) * math.pi / 180 * 6371000] for x in f])
rows = []
for i in range(0, len(f) - 40, 40):                      # ~1 s windows
    j = i + 40
    v = (xy[j] - xy[i]) / (t[j] - t[i])
    cur = np.array(f[j][4]); yaw = f[j][3]; a = f[j][5]
    head = np.array([math.cos(yaw), math.sin(yaw)])
    rows.append((t[j] - t[0], np.linalg.norm(v), cur, v - cur, (a[0] + 1) / 2 * 2.5, (v - cur) @ head))
rows = rows[len(rows) // 3:]                              # skip the start-up transient
cur = np.mean([r[2] for r in rows], axis=0)
print(f"current at the WAM-V: ({cur[0]:+.3f}, {cur[1]:+.3f}) m/s  |current| {np.linalg.norm(cur):.3f}")
print(f"commanded surge (median)        {np.median([r[4] for r in rows]):.3f} m/s")
print(f"through-water speed along hdg   {np.median([r[5] for r in rows]):.3f} m/s   <- should match the command")
print(f"speed over ground (median)      {np.median([r[1] for r in rows]):.3f} m/s   <- differs by the current")
PY
grep -m3 "attached\|Episode started\|current_max" /tmp/stable.log | cut -c1-200
grep -ci "traceback\|error" /tmp/stable.log
