"""
Attach the WamvCurrentCorrection Gazebo system to the running WAM-V (VRX is not modified).

Waits until the WAM-V model exists, then calls Gazebo's /world/<world>/entity/system/add
service once. The plugin then reads the water current from the gz topic /asv_stable/water_current,
which the launch file bridges from the ROS topic of the same name (published by the bridge node).
Attaching again (e.g. relaunching without restarting Gazebo) is harmless: only the first
instance in the Gazebo process applies forces.
"""

import re
import subprocess
import time

import rclpy
from rclpy.node import Node

PLUGIN_LIB = "/work/ros_ws/install/wamv_current/lib/libWamvCurrentCorrection.so"
# Coefficients of the WAM-V's vrx::SimpleHydrodynamics (wamv_gazebo_dynamics_plugin.xacro).
INNER_XML = ("<link_name>wamv/base_link</link_name><topic>/asv_stable/water_current</topic>"
             "<xDotU>0.0</xDotU><yDotV>0.0</yDotV><xU>100.0</xU><xUU>150.0</xUU>"
             "<yV>100.0</yV><yVV>100.0</yVV>")


class AttachCurrent(Node):
    def __init__(self):
        super().__init__("asv_stable_attach_current")
        self.declare_parameter("world_name", "open_ocean")
        self.declare_parameter("model_name", "wamv")
        world = self.get_parameter("world_name").value
        model = self.get_parameter("model_name").value

        model_id = None
        for _ in range(120):
            r = subprocess.run(["gz", "model", "-m", model], capture_output=True, text=True, timeout=30)
            m = re.search(r"Model: \[(\d+)\]", r.stdout)
            if m:
                model_id = int(m.group(1))
                break
            self.get_logger().info(f"waiting for model '{model}' ...", throttle_duration_sec=10.0)
            time.sleep(2.0)
        if model_id is None:
            self.get_logger().error(f"model '{model}' not found; current NOT applied to the WAM-V")
            return

        xml = INNER_XML.replace('"', '\\"')
        req = (f'entity {{ id: {model_id} }} plugins {{ name: "stable::WamvCurrentCorrection" '
               f'filename: "{PLUGIN_LIB}" innerxml: "{xml}" }}')
        r = subprocess.run(["gz", "service", "-s", f"/world/{world}/entity/system/add",
                            "--reqtype", "gz.msgs.EntityPlugin_V", "--reptype", "gz.msgs.Boolean",
                            "--timeout", "5000", "--req", req], capture_output=True, text=True, timeout=30)
        ok = r.returncode == 0 and "true" in r.stdout
        (self.get_logger().info if ok else self.get_logger().error)(
            f"WamvCurrentCorrection on model {model} (id {model_id}): "
            f"{'attached' if ok else 'FAILED ' + (r.stderr.strip() or r.stdout.strip())}")


def main(args=None):
    rclpy.init(args=args)
    node = AttachCurrent()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
