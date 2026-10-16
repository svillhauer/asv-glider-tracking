"""
Stable-reference bridge node.

Runs the training env's glider, dive, DAC and EKF model (EkfCoreStable) once per
env tick and turns the WAM-V's measured pose into the 27-D observation the
stable reference policy was trained on.

Gazebo mode (default): ASV pose from GPS + IMU.
Kinematic mode (gazebo_mode:=False): the core integrates the ASV itself, which is
the training env exactly; use it to test the ROS plumbing without Gazebo.

Tick timing: the reset observation is published once GPS and IMU are ready
(with the heading overwritten by the measured spawn heading). The policy acts on
it, the WAM-V executes that action for one tick, and at the end of the tick the
core steps with the measured pose and publishes the next observation.

Publishes
  /asv_stable/obs_raw          Float32MultiArray  27-D raw observation
  /asv_stable/obs_normalized   Float32MultiArray  VecNormalize-applied observation
  /asv_stable/glider_state     PoseArray          [true g1, true g2, est g1, est g2], Gazebo frame [m],
                                                  z = -depth, orientation = heading + dive pitch
  /asv_stable/metrics          Float32MultiArray  per-tick metrics, names in layout.dim[0].label
Subscribes
  /asv_stable/action           Float32MultiArray  from stable_policy_node
  /wamv/sensors/gps/gps/fix    NavSatFix          (gazebo mode)
  /wamv/sensors/imu/imu/data   Imu                (gazebo mode)
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, Vector3
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from asv_bridge_stable.ekf_core_stable import PHYSICAL_MAX_SPEED, EkfCoreStable, physical_scales
from asv_bridge_stable.policy_numpy import VecNormalizeNumpy

_LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                          history=HistoryPolicy.KEEP_LAST)
_DYN = ParameterDescriptor(dynamic_typing=True)   # accept 2000 as well as 2000.0 from the launch file
_R_EARTH = 6_371_000.0   # metres

METRIC_NAMES = [
    "step", "reward", "r_unc", "r_comm", "r_ctrl", "in_range_frac", "depth_m",
    "slant_range1_m", "slant_range2_m", "det_P1", "det_P2",
    "asv_x_m", "asv_y_m", "asv_heading",
    "g1_x_m", "g1_y_m", "g2_x_m", "g2_y_m",
    "g1_est_x_m", "g1_est_y_m", "g2_est_x_m", "g2_est_y_m",
    "action_surge", "action_yaw", "asv_current_x", "asv_current_y",
    "received1", "received2", "adcp_obs_x", "adcp_obs_y",
]


def _f32(arr, label):
    msg = Float32MultiArray()
    dim = MultiArrayDimension()
    dim.label, dim.size, dim.stride = label, len(arr), len(arr)
    msg.layout.dim = [dim]
    msg.data = [float(v) for v in arr]
    return msg


def _quat_from_yaw_pitch(yaw, pitch):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return cp * cy, -sp * sy, sp * cy, cp * sy   # w, x, y, z  (ZYX, roll = 0)


class StableBridgeNode(Node):

    def __init__(self):
        super().__init__("asv_stable_bridge")
        self.declare_parameter("seed", 900, _DYN)
        self.declare_parameter("sep_m", 2000.0, _DYN)
        self.declare_parameter("current_max", 0.08, _DYN)
        self.declare_parameter("gazebo_mode", True)
        self.declare_parameter("tick_period", 0.0, _DYN)   # 0 = physical tick from sep_m; set small for kinematic tests
        self.declare_parameter("max_ticks", 0, _DYN)       # 0 = full training episode (2880 ticks)
        self.declare_parameter("glider_model", "dave")     # "dave" = DAVE flight model, "env" = training model
        self.declare_parameter("dave_max_depth_m", 1000.0, _DYN)
        self.declare_parameter("dave_surface_time_s", 600.0, _DYN)
        self.declare_parameter("glider_update_s", 10.0, _DYN)
        self.declare_parameter("acoustic_model", "ideal")   # "ideal" (training) or "realistic"
        self.declare_parameter("adcp_noise", False)          # noisy, biased ADCP at the ASV
        self.declare_parameter("current_profile", "uniform")  # "uniform" (training) or "decay" (weaker at depth)
        self.declare_parameter("glider_nav_error", False)     # glider compass/speed/GPS errors in its DAC
        self.declare_parameter("vecnorm_npz",
                               "/work/policy/best_vec_normalize_arrays.npz")

        seed = int(self.get_parameter("seed").value)
        self._sep_m = float(self.get_parameter("sep_m").value)
        current_max = float(self.get_parameter("current_max").value)
        self._gazebo = bool(self.get_parameter("gazebo_mode").value)

        glider_model = str(self.get_parameter("glider_model").value)
        self._core = EkfCoreStable(
            config={"current_max": current_max,
                    "dave_max_depth_m": float(self.get_parameter("dave_max_depth_m").value),
                    "dave_surface_time_s": float(self.get_parameter("dave_surface_time_s").value),
                    "acoustic_model": str(self.get_parameter("acoustic_model").value),
                    "adcp_noise": bool(self.get_parameter("adcp_noise").value),
                    "current_profile": str(self.get_parameter("current_profile").value),
                    "glider_nav_error": bool(self.get_parameter("glider_nav_error").value)},
            sep_m=self._sep_m, glider_model=glider_model)
        self._obs = self._core.reset_core(seed=seed)
        self._vecnorm = VecNormalizeNumpy(self.get_parameter("vecnorm_npz").value)
        self._max_ticks = int(self.get_parameter("max_ticks").value) or self._core.max_steps

        sc = physical_scales(self._sep_m)
        if glider_model == "dave":
            cs = self._core.dave[0].cycle_stats()
            dive = (f"DAVE gliders: {self._core.dave_cfg['dave_max_depth_m']:.0f} m yo-yo, "
                    f"{cs['u_descend']:.2f} m/s horizontal, {-cs['w_descend']:.3f} m/s vertical, "
                    f"glide {cs['glide_angle_deg']:.1f} deg, pump {self._core.dave_cfg['dave_pump_cc']:.0f} cc, "
                    f"cycle {cs['period_s'] / 3600:.2f} h")
        else:
            dive = f"env gliders: dive {sc['dive_depth_m']:.0f} m / {sc['dive_period_s'] / 3600:.2f} h"
        self.get_logger().info(
            f"Stable reference  seed={seed}  sep_m={self._sep_m:.0f}  current_max={current_max}  "
            f"tick={sc['tick_s']:.1f} s  comm_range={sc['comm_range_m']:.0f} m  "
            f"episode={self._max_ticks} ticks ({self._max_ticks * sc['tick_s'] / 3600:.1f} h sim)  "
            f"gazebo_mode={self._gazebo}")
        self.get_logger().info(dive)
        rc = self._core.real_cfg
        self.get_logger().info(
            f"Sensing: acoustic_model={rc['acoustic_model']}"
            + (f" (p_max {rc['ac_p_max']}, 50% at {rc['ac_r50_frac']:.2f} R_comm, outliers {rc['ac_outlier_p']:.0%}, "
               f"sound-speed scale {self._core.sound_speed_scale:.4f}, DAC loss {rc['ac_dac_loss_p']:.0%})"
               if rc["acoustic_model"] == "realistic" else "")
            + f"  adcp_noise={rc['adcp_noise']}"
            + (f" (bias {(self._core.adcp_bias * PHYSICAL_MAX_SPEED).round(3)} m/s,"
               f" noise {rc['adcp_noise_std_mps']} m/s)"
               if rc["adcp_noise"] else ""))
        s1000, a1000 = self._core.current_profile_factor(1000.0)
        nav = ""
        if rc["glider_nav_error"] and glider_model == "dave":
            nav = "  (" + ", ".join(
                f"G{i + 1}: compass {math.degrees(g.nav_heading_bias):+.1f} deg, speed {g.nav_speed_scale - 1:+.1%}"
                for i, g in enumerate(self._core.dave)) + f", GPS {rc['gnav_gps_std_m']} m)"
        self.get_logger().info(
            f"Ocean/gliders: current_profile={rc['current_profile']}"
            + (f" (e-fold {rc['current_efold_m']:.0f} m, at 1000 m x{s1000:.2f} rotated {math.degrees(a1000):+.0f} deg)"
               if rc["current_profile"] == "decay" else "")
            + f"  glider_nav_error={rc['glider_nav_error']}" + nav)

        self._pub_raw = self.create_publisher(Float32MultiArray, "/asv_stable/obs_raw", 10)
        self._pub_norm = self.create_publisher(Float32MultiArray, "/asv_stable/obs_normalized", 10)
        self._pub_metrics = self.create_publisher(Float32MultiArray, "/asv_stable/metrics", 10)
        self._pub_gliders = self.create_publisher(PoseArray, "/asv_stable/glider_state", _LATCHED_QOS)
        # Water current at the WAM-V [m/s, ENU], from the same current field the env and gliders use.
        # Drives the WamvCurrentCorrection Gazebo plugin (via ros_gz_bridge) and the policy node's
        # through-water speed estimate (the ASV's ADCP in the training env).
        self._pub_current = self.create_publisher(Vector3, "/asv_stable/water_current", 10)
        # What the ASV's ADCP reports (true current, or with bias and noise when adcp_noise is on);
        # the policy node's speed loop uses this, the hull plugin uses the true current above.
        self._pub_adcp = self.create_publisher(Vector3, "/asv_stable/adcp_current", 10)

        self._action = np.zeros(2, dtype=np.float32)
        self.create_subscription(Float32MultiArray, "/asv_stable/action", self._action_cb, 10)

        self._gps_ref = None
        self._pos_m = np.zeros(2)
        self._yaw = 0.0
        self._gps_ready = self._imu_ready = False
        if self._gazebo:
            self.create_subscription(NavSatFix, "/wamv/sensors/gps/gps/fix", self._gps_cb, 10)
            self.create_subscription(Imu, "/wamv/sensors/imu/imu/data", self._imu_cb, 10)

        if self._gazebo:
            self.create_timer(0.1, self._publish_current)
        self._started = False
        self._return = 0.0
        self._done = False
        self._tick_period = float(self.get_parameter("tick_period").value) or sc["tick_s"]
        # Poll every second until GPS/IMU are up, publish the reset observation right away,
        # then start the env-tick timer (so the first action doesn't wait a whole tick).
        self._start_timer = self.create_timer(1.0, self._try_start)
        self._tick_start = None
        if glider_model == "dave":
            # Gliders move continuously between ticks (visuals and glider_state); decisions,
            # pings, EKF and metrics stay once per tick, as in training.
            self.create_timer(float(self.get_parameter("glider_update_s").value), self._glider_update)
        self._pub_gliders.publish(self._glider_msg())   # lets the viz node spawn the gliders right away

    # Callbacks
    def _action_cb(self, msg):
        self._action = np.clip(np.array(msg.data, dtype=np.float32), -1.0, 1.0)

    def _gps_cb(self, msg: NavSatFix):
        if self._gps_ref is None:
            self._gps_ref = (msg.latitude, msg.longitude)
        lat0, lon0 = self._gps_ref
        self._pos_m[0] = (msg.longitude - lon0) * math.pi / 180.0 * _R_EARTH * math.cos(math.radians(lat0))
        self._pos_m[1] = (msg.latitude - lat0) * math.pi / 180.0 * _R_EARTH
        self._gps_ready = True

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        self._yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._imu_ready = True

    # Tick
    def _try_start(self):
        if self._gazebo and not (self._gps_ready and self._imu_ready):
            self.get_logger().warn(f"Waiting for GPS ({self._gps_ready}) and IMU ({self._imu_ready}) ...",
                                   throttle_duration_sec=5.0)
            return
        # Give a bag recorder up to 10 polls to subscribe, then one more poll for the match to settle,
        # so the reset observation and the first action are not lost from the bag.
        self._start_polls = getattr(self, "_start_polls", 0) + 1
        if self._start_polls <= 10 and self._pub_raw.get_subscription_count() == 0:
            return
        if not getattr(self, "_settled", False) and self._start_polls <= 11:
            self._settled = True
            return
        # Episode start: the Gazebo origin is the WAM-V's position at its first GPS fix, and the
        # reset observation uses the measured heading instead of the env's random draw.
        if self._gazebo:
            self._core.asv_heading = float(self._yaw)
            self._obs = self._core._get_obs()
        self._started = True
        self._start_timer.cancel()
        self._publish(self._obs, metrics=False)
        self.create_timer(self._tick_period, self._tick)
        self._tick_start = self.get_clock().now()
        self.get_logger().info(f"Episode started: published reset observation (heading {self._core.asv_heading!r} rad, "
                               f"{self._pub_raw.get_subscription_count()} obs_raw subscriber(s)); "
                               f"ticking every {self._tick_period:.1f} s.")

    def _current_at_asv(self):
        c = self._core._get_current(self._core.gazebo_to_nd(self._pos_m))
        return np.asarray(c, dtype=float) * PHYSICAL_MAX_SPEED

    def _publish_current(self):
        if not self._gps_ready:
            return
        c = self._current_at_asv()
        self._pub_current.publish(Vector3(x=float(c[0]), y=float(c[1]), z=0.0))
        a = self._core.adcp_sample_mps(self._core.gazebo_to_nd(self._pos_m))
        self._pub_adcp.publish(Vector3(x=float(a[0]), y=float(a[1]), z=0.0))

    def _glider_update(self):
        if not self._started or self._done or self._tick_start is None:
            return
        elapsed = (self.get_clock().now() - self._tick_start).nanoseconds * 1e-9
        self._core.advance_gliders_to(elapsed)
        self._pub_gliders.publish(self._glider_msg())

    def _tick(self):
        if self._done:
            return
        self._tick_start = self.get_clock().now()

        action = self._action.copy()
        if self._gazebo:
            self._obs = self._core.step_gazebo(self._core.gazebo_to_nd(self._pos_m), self._yaw, action)
        else:
            self._obs = self._core.step_kinematic(action)
        self._return += self._core.last_info["reward"]
        self._publish(self._obs, metrics=True, action=action)

        k = self._core.step_count
        if k % 10 == 0 or k == 1:
            li = self._core.last_info
            self.get_logger().info(
                f"step={k:4d}  reward={li['reward']:+.2f}  in_range={li['in_range_frac']:.2f}  "
                f"depth={li['depth'] * self._sep_m:.0f} m  "
                f"slant=[{li['dist_each'][0] * self._sep_m:.0f}, {li['dist_each'][1] * self._sep_m:.0f}] m  "
                f"gliders={'/'.join(self._core.glider_modes())}")
        if k >= self._max_ticks:
            self._done = True
            self.get_logger().info(f"Episode finished after {k} steps.  return={self._return:,.0f}")

    def _publish(self, obs, metrics, action=None):
        self._pub_raw.publish(_f32(obs, "obs"))
        self._pub_norm.publish(_f32(self._vecnorm(obs), "obs"))
        self._pub_gliders.publish(self._glider_msg())
        if metrics:
            c, li, s = self._core, self._core.last_info, self._sep_m
            true_xy, est_xy, depth_m, _, _ = c.glider_state_gazebo()
            asv_xy = c.nd_to_gazebo(c.asv_pos)
            row = [c.step_count, li["reward"], li["r_unc"], li["r_comm"], li["r_ctrl"],
                   li["in_range_frac"], depth_m,
                   li["dist_each"][0] * s, li["dist_each"][1] * s,
                   li["det_P_each"][0], li["det_P_each"][1],
                   asv_xy[0], asv_xy[1], c.asv_heading,
                   *true_xy[0], *true_xy[1], *est_xy[0], *est_xy[1],
                   action[0], action[1],
                   *(np.asarray(c._get_current(c.asv_pos), dtype=float) * PHYSICAL_MAX_SPEED),
                   *li.get("received", np.zeros(2)),
                   *(np.asarray(c.adcp_reading, dtype=float) * PHYSICAL_MAX_SPEED)]
            self._pub_metrics.publish(_f32(row, ",".join(METRIC_NAMES)))

    def _glider_msg(self):
        true_xy, est_xy, depth_m, heading, pitch = self._core.glider_state_gazebo()
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for xy, yaw, z, pit in [(true_xy[0], heading[0], -depth_m, pitch[0]),
                                (true_xy[1], heading[1], -depth_m, pitch[1]),
                                (est_xy[0], heading[0], -depth_m, 0.0),
                                (est_xy[1], heading[1], -depth_m, 0.0)]:
            p = Pose()
            p.position.x, p.position.y, p.position.z = float(xy[0]), float(xy[1]), float(z)
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = _quat_from_yaw_pitch(yaw, pit)
            msg.poses.append(p)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = StableBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
