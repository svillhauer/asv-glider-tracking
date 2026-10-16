"""
Stable-reference policy node: numpy SAC actor + WAM-V inner-loop velocity controller.

Differences from the old asv_bridge policy_node, both from the stable env:
  * Surge is forward-only: action[0] in [-1, 1] -> surge = (a + 1) / 2 * max_surge
    (the old node mapped a < 0 to reverse thrust).
  * Yaw cap is 12 nondimensional (not 400): full yaw action = 12 / tau rad/s,
    i.e. 0.015 rad/s and at most 0.45 rad (26 deg) of heading change per tick at sep_m = 2000.

Outer loop (one env tick, triggered by /asv_stable/obs_normalized):
  actor -> [surge_cmd (m/s), yaw_rate_cmd (rad/s)] -> target heading = heading + yaw_rate_cmd * tick.
  The env updates the heading first and then moves the ASV along the new heading for
  the whole tick, so the inner loop turns to the target heading and then holds it.
Inner loop (10 Hz): PI on surge (GPS-derived) + heading P -> yaw-rate P -> differential thrust.
Gains are the old node's, which were tuned on the same WAM-V.

Publishes /asv_stable/action and /wamv/thrusters/{left,right}/thrust.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32MultiArray, Float64, MultiArrayDimension

from asv_bridge_stable.ekf_core_stable import physical_scales
from asv_bridge_stable.policy_numpy import SACActorNumpy

_DYN = ParameterDescriptor(dynamic_typing=True)   # accept 2000 as well as 2000.0 from the launch file
_R_EARTH = 6_371_000.0


class StablePolicyNode(Node):

    KP_SURGE = 500.0            # N / (m/s)
    KI_SURGE = 100.0            # N / (m/s * s)
    KP_YAW = 1000.0             # N*m / (rad/s)
    KP_HEADING = 2.0            # (rad/s) / rad
    MAX_YAW_RATE_INNER = 0.5    # rad/s cap on the heading controller output
    MAX_THRUST = 1500.0         # N per thruster (VRX WAM-V)
    CTRL_HZ = 10.0

    def __init__(self):
        super().__init__("asv_stable_policy")
        self.declare_parameter("actor_weights", "/work/policy/actor_weights.npz")
        self.declare_parameter("sep_m", 2000.0, _DYN)

        sc = physical_scales(float(self.get_parameter("sep_m").value))
        self.MAX_SURGE = sc["max_surge_m_s"]
        self.MAX_YAW_RATE = sc["max_yaw_rate_rad_s"]
        self.DT_TICK = sc["tick_s"]

        self._actor = SACActorNumpy(self.get_parameter("actor_weights").value)
        self.get_logger().info(
            f"Actor loaded (obs_dim={self._actor.obs_dim})  max_surge={self.MAX_SURGE:.2f} m/s  "
            f"max_yaw_rate={self.MAX_YAW_RATE:.4f} rad/s  tick={self.DT_TICK:.1f} s")

        self._pub_action = self.create_publisher(Float32MultiArray, "/asv_stable/action", 10)
        self._pub_left = self.create_publisher(Float64, "/wamv/thrusters/left/thrust", 10)
        self._pub_right = self.create_publisher(Float64, "/wamv/thrusters/right/thrust", 10)
        self.create_subscription(Float32MultiArray, "/asv_stable/obs_normalized", self._obs_cb, 10)
        self.create_subscription(Imu, "/wamv/sensors/imu/imu/data", self._imu_cb, 10)
        self.create_subscription(NavSatFix, "/wamv/sensors/gps/gps/fix", self._gps_cb, 10)
        # water current at the boat from the ASV's ADCP (noisy when the bridge's adcp_noise is on);
        # the policy's surge command is a speed through the water, as in the env,
        # so the inner loop tracks GPS speed minus current
        self._current = (0.0, 0.0)
        self.create_subscription(Vector3, "/asv_stable/adcp_current", self._current_cb, 10)

        self._surge_cmd = 0.0
        self._yaw_cmd = 0.0
        self._target_heading = None
        self._yaw_rate = 0.0
        self._surge_meas = 0.0
        self._heading = 0.0
        self._gps_ref = self._gps_pos = self._gps_time = None
        self._surge_int = 0.0
        self._int_limit = self.MAX_THRUST / self.KI_SURGE
        self._step = 0
        self._ctrl_count = 0
        self.create_timer(1.0 / self.CTRL_HZ, self._ctrl_step)

    # Outer loop
    def _obs_cb(self, msg):
        obs = np.array(msg.data, dtype=np.float32)
        if obs.shape[0] != self._actor.obs_dim:
            self.get_logger().error(f"obs has {obs.shape[0]} values, actor expects {self._actor.obs_dim}")
            return
        action = np.clip(self._actor.predict(obs), -1.0, 1.0).astype(np.float32)

        self._surge_cmd = (float(action[0]) + 1.0) / 2.0 * self.MAX_SURGE   # forward-only
        self._yaw_cmd = float(action[1]) * self.MAX_YAW_RATE
        target = self._heading + self._yaw_cmd * self.DT_TICK
        self._target_heading = (target + math.pi) % (2 * math.pi) - math.pi
        self._step += 1

        out = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label, dim.size, dim.stride = "action", 2, 2
        out.layout.dim = [dim]
        out.data = [float(action[0]), float(action[1])]
        self._pub_action.publish(out)
        self.get_logger().info(
            f"step={self._step:4d}  action=[{action[0]:+.3f}, {action[1]:+.3f}]  "
            f"cmd=[{self._surge_cmd:.2f} m/s, {self._yaw_cmd:+.4f} rad/s]  "
            f"tgt_hdg={math.degrees(self._target_heading):.1f} deg")

    # Sensors
    def _imu_cb(self, msg):
        self._yaw_rate = msg.angular_velocity.z
        q = msg.orientation
        self._heading = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _current_cb(self, msg):
        self._current = (msg.x, msg.y)

    def _gps_cb(self, msg):
        if self._gps_ref is None:
            self._gps_ref = (msg.latitude, msg.longitude)
        lat0, lon0 = self._gps_ref
        x = (msg.longitude - lon0) * math.pi / 180.0 * _R_EARTH * math.cos(math.radians(lat0))
        y = (msg.latitude - lat0) * math.pi / 180.0 * _R_EARTH
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._gps_pos is not None and now - self._gps_time > 0.01:
            dt = now - self._gps_time
            c, s = math.cos(self._heading), math.sin(self._heading)
            vx = (x - self._gps_pos[0]) / dt - self._current[0]     # through-water velocity
            vy = (y - self._gps_pos[1]) / dt - self._current[1]
            raw = vx * c + vy * s
            self._surge_meas = 0.3 * raw + 0.7 * self._surge_meas
        self._gps_pos, self._gps_time = (x, y), now

    # Inner loop
    def _ctrl_step(self):
        dt = 1.0 / self.CTRL_HZ
        err = self._surge_cmd - self._surge_meas
        self._surge_int = float(np.clip(self._surge_int + err * dt, -self._int_limit, self._int_limit))
        f_surge = self.KP_SURGE * err + self.KI_SURGE * self._surge_int

        if self._target_heading is not None:
            h_err = (self._target_heading - self._heading + math.pi) % (2 * math.pi) - math.pi
            des_yaw = float(np.clip(self.KP_HEADING * h_err, -self.MAX_YAW_RATE_INNER, self.MAX_YAW_RATE_INNER))
        else:
            h_err, des_yaw = 0.0, 0.0
        f_yaw = self.KP_YAW * (des_yaw - self._yaw_rate)

        left = float(np.clip(f_surge - f_yaw, -self.MAX_THRUST, self.MAX_THRUST))
        right = float(np.clip(f_surge + f_yaw, -self.MAX_THRUST, self.MAX_THRUST))
        self._pub_left.publish(Float64(data=left))
        self._pub_right.publish(Float64(data=right))

        self._ctrl_count += 1
        if self._ctrl_count % 50 == 0:
            self.get_logger().info(
                f"ctrl: surge cmd/meas={self._surge_cmd:.2f}/{self._surge_meas:.2f} m/s  "
                f"hdg={math.degrees(self._heading):.1f} deg  hdg_err={math.degrees(h_err):.1f} deg  "
                f"L={left:.0f} R={right:.0f} N")


def main(args=None):
    rclpy.init(args=args)
    node = StablePolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
