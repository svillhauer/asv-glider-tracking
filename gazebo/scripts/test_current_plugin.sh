#!/usr/bin/env bash
# Headless check of the WAM-V current plugin (run inside the container):
#   bash /work/scripts/test_current_plugin.sh
# Zero thrust; measures the WAM-V's drift from GPS with no current, then with a 0.2/0.1 m/s current.
source /opt/ros/jazzy/setup.bash
source /work/vrx_ws/install/setup.bash
source /work/ros_ws/install/setup.bash
export GZ_SIM_RESOURCE_PATH=/work/models:/work/worlds:/work/vrx_ws/install/wamv_description/share:/work/vrx_ws/install/wamv_gazebo/share:/work/vrx_ws/install/vrx_gazebo/share
ros2 launch vrx_gz vrx_environment.launch.py world:=/work/worlds/open_ocean.sdf sim_mode:=full \
    config_file:=/work/config/wamv_spawn.yaml headless:=true > /tmp/vrx.log 2>&1 &
sleep 45
python3 /work/ros_ws/src/asv_bridge_stable/asv_bridge_stable/attach_current_node.py 2>&1 | grep -v "^$"
python3 - <<'PY'
import math, subprocess, threading, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

class G(Node):
    def __init__(s):
        super().__init__("cur_test"); s.fix = []
        s.create_subscription(NavSatFix, "/wamv/sensors/gps/gps/fix", s.cb, 50)
    def cb(s, m):
        s.fix.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.latitude, m.longitude))

rclpy.init(); n = G()
def spin_for(sec):
    t0 = time.time()
    while time.time() - t0 < sec: rclpy.spin_once(n, timeout_sec=0.1)
def vel(fixes):
    (t0, la0, lo0), (t1, la1, lo1) = fixes[0], fixes[-1]
    dx = (lo1 - lo0) * math.pi / 180 * 6371000 * math.cos(math.radians(la0))
    dy = (la1 - la0) * math.pi / 180 * 6371000
    return dx / (t1 - t0), dy / (t1 - t0), t1 - t0

def publish_current(x, y, sec):
    end = time.time() + sec
    while time.time() < end:
        subprocess.run(["gz", "topic", "-t", "/vrx/wamv/water_current", "-m", "gz.msgs.Vector3d",
                        "-p", f"x: {x}, y: {y}"], capture_output=True)
        time.sleep(1.0)

for x, y in ((0.0, 0.0), (0.2, 0.1), (-0.15, 0.05)):
    th = threading.Thread(target=publish_current, args=(x, y, 70)); th.start()
    spin_for(15)                      # let the boat reach the drift speed
    n.fix.clear(); spin_for(50)
    vx, vy, dt = vel(n.fix)
    print(f"current ({x:+.2f}, {y:+.2f}) m/s -> WAM-V drift ({vx:+.3f}, {vy:+.3f}) m/s over {dt:.0f} s sim")
    th.join()
PY
