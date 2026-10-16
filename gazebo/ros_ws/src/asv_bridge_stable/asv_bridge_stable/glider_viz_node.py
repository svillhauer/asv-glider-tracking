"""
Stable-reference glider visualization: spawns two Slocum models and moves them
every tick to the TRUE glider positions from the bridge, including the dive.

Subscribes /asv_stable/glider_state (PoseArray [true g1, true g2, est g1, est g2]).
Poses are already in the Gazebo frame [m]; z = -depth, orientation = heading plus
a +/-26 deg glide pitch (nose down descending, nose up ascending).
The Slocum model is static and visual-only, so it is simply teleported.
"""

import subprocess

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

_LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                          history=HistoryPolicy.KEEP_LAST)


class StableGliderVizNode(Node):

    def __init__(self):
        super().__init__("asv_stable_glider_viz")
        self.declare_parameter("world_name", "open_ocean")
        self.declare_parameter("model_sdf", "/work/models/glider_slocum/model.sdf")
        self._world = self.get_parameter("world_name").value
        self._sdf = self.get_parameter("model_sdf").value
        self._names = ["glider1", "glider2"]
        self._spawned = False
        self.create_subscription(PoseArray, "/asv_stable/glider_state", self._cb, _LATCHED_QOS)
        self.get_logger().info(f"Glider viz ready  world={self._world}")

    def _cb(self, msg):
        if len(msg.poses) < 2:
            return
        for name, p in zip(self._names, msg.poses[:2], strict=False):
            pose = (f'position {{ x: {p.position.x:.3f} y: {p.position.y:.3f} z: {p.position.z:.3f} }} '
                    f'orientation {{ w: {p.orientation.w:.6f} x: {p.orientation.x:.6f} '
                    f'y: {p.orientation.y:.6f} z: {p.orientation.z:.6f} }}')
            if not self._spawned:
                self._call(f"/world/{self._world}/create", "gz.msgs.EntityFactory",
                           f'sdf_filename: "{self._sdf}" name: "{name}" pose {{ {pose} }}')
            else:
                self._call(f"/world/{self._world}/set_pose", "gz.msgs.Pose", f'name: "{name}" {pose}')
        if not self._spawned:
            self._spawned = True
            self.get_logger().info("Gliders spawned.")

    def _call(self, service, reqtype, req):
        cmd = ["gz", "service", "-s", service, "--reqtype", reqtype, "--reptype", "gz.msgs.Boolean",
               "--timeout", "5000", "--req", req]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                self.get_logger().warn(f"{service}: {r.stderr.strip() or r.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.get_logger().warn(f"{service}: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = StableGliderVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
