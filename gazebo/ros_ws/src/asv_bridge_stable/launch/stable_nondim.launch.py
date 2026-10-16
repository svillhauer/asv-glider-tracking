"""
Stable reference policy (nglider_knowndepth_flat_fixed2_stable_s0) on the VRX WAM-V.

Defaults: gliders 20 km apart (sep_m), moved by Project DAVE's glider flight model
(1000 m yo-yo, 0.30 m/s glide, 10 min at the surface, ~4.1 h per dive).
Physical scale at sep_m = 20000: tick 300 s, yaw cap 0.0015 rad/s, comm range 8000 m.
A full 2880-tick episode is 240 h of sim time; use max_ticks for shorter runs.

Terminal 1 (inside the container):
  ros2 launch vrx_gz vrx_environment.launch.py \\
      world:=/work/worlds/open_ocean.sdf sim_mode:=full config_file:=/work/config/wamv_spawn.yaml
Terminal 2 (docker exec into the same container):
  source /work/ros_ws/install/setup.bash
  ros2 launch asv_bridge_stable stable_nondim.launch.py current_max:=0.0 max_ticks:=300

Training glider model instead of DAVE, or the old 2 km scale:
  ... glider_model:=env      ... sep_m:=2000.0

Without Gazebo (checks the ROS plumbing; the ASV follows the training kinematics):
  ros2 launch asv_bridge_stable stable_nondim.launch.py gazebo_mode:=false use_sim_time:=false tick_period:=0.02
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

REF = "/work/policy"


def generate_launch_description():
    sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription([
        DeclareLaunchArgument("seed", default_value="900"),
        DeclareLaunchArgument("sep_m", default_value="20000.0"),
        DeclareLaunchArgument("glider_model", default_value="dave",
                              description="dave = DAVE glider flight model, env = training glider model"),
        DeclareLaunchArgument("dave_max_depth_m", default_value="1000.0"),
        DeclareLaunchArgument("dave_surface_time_s", default_value="600.0"),
        DeclareLaunchArgument("acoustic_model", default_value="ideal",
                              description="ideal = training acoustic channel, realistic = range-dependent "
                                          "reception, delay, sound-speed error, multipath, DAC loss"),
        DeclareLaunchArgument("adcp_noise", default_value="false",
                              description="true = biased, noisy ADCP current at the ASV"),
        DeclareLaunchArgument("current_profile", default_value="uniform",
                              description="uniform = training current at every depth, decay = weaker and "
                                          "veered with depth (gliders only)"),
        DeclareLaunchArgument("glider_nav_error", default_value="false",
                              description="true = glider compass bias, speed error and GPS noise in its DAC"),
        DeclareLaunchArgument("glider_update_s", default_value="10.0",
                              description="sim seconds between DAVE glider position updates in Gazebo"),
        DeclareLaunchArgument("max_ticks", default_value="0",
                              description="0 = full 2880-tick episode; e.g. 300 for a shorter run"),
        DeclareLaunchArgument("current_max", default_value="0.08",
                              description="0.08 = training value; 0.0 matches the current-free Gazebo world"),
        DeclareLaunchArgument("gazebo_mode", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("tick_period", default_value="0.0",
                              description="0 = physical tick from sep_m; e.g. 0.05 for a fast kinematic test"),
        DeclareLaunchArgument("world_name", default_value="open_ocean"),

        Node(package="asv_bridge_stable", executable="stable_bridge_node", name="asv_stable_bridge",
             output="screen",
             parameters=[{"seed": LaunchConfiguration("seed"),
                          "sep_m": LaunchConfiguration("sep_m"),
                          "current_max": LaunchConfiguration("current_max"),
                          "gazebo_mode": LaunchConfiguration("gazebo_mode"),
                          "tick_period": LaunchConfiguration("tick_period"),
                          "max_ticks": LaunchConfiguration("max_ticks"),
                          "glider_model": LaunchConfiguration("glider_model"),
                          "dave_max_depth_m": LaunchConfiguration("dave_max_depth_m"),
                          "dave_surface_time_s": LaunchConfiguration("dave_surface_time_s"),
                          "glider_update_s": LaunchConfiguration("glider_update_s"),
                          "acoustic_model": LaunchConfiguration("acoustic_model"),
                          "adcp_noise": LaunchConfiguration("adcp_noise"),
                          "current_profile": LaunchConfiguration("current_profile"),
                          "glider_nav_error": LaunchConfiguration("glider_nav_error"),
                          "vecnorm_npz": f"{REF}/best_vec_normalize_arrays.npz",
                          "use_sim_time": sim_time}]),

        Node(package="asv_bridge_stable", executable="stable_policy_node", name="asv_stable_policy",
             output="screen",
             parameters=[{"actor_weights": f"{REF}/actor_weights.npz",
                          "sep_m": LaunchConfiguration("sep_m"),
                          "use_sim_time": sim_time}]),

        # Ocean current on the WAM-V: bridge the current to Gazebo and attach the hull-current plugin.
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="asv_stable_current_bridge",
             output="screen", condition=IfCondition(LaunchConfiguration("gazebo_mode")),
             arguments=["/asv_stable/water_current@geometry_msgs/msg/Vector3]gz.msgs.Vector3d"],
             parameters=[{"use_sim_time": sim_time}]),
        Node(package="asv_bridge_stable", executable="stable_attach_current", name="asv_stable_attach_current",
             output="screen", condition=IfCondition(LaunchConfiguration("gazebo_mode")),
             parameters=[{"world_name": LaunchConfiguration("world_name")}]),

        Node(package="asv_bridge_stable", executable="stable_glider_viz_node", name="asv_stable_glider_viz",
             output="screen", condition=IfCondition(LaunchConfiguration("gazebo_mode")),
             parameters=[{"world_name": LaunchConfiguration("world_name"),
                          "model_sdf": "/work/models/glider_slocum/model.sdf",
                          "use_sim_time": sim_time}]),
    ])
