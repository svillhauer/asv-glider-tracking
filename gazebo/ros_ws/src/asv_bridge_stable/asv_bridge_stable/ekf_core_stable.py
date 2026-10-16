"""
EkfCoreStable: Gazebo port of the stable reference environment
(``AsvNGliderCurrentsADCPEnvKnownDepthFlatStable``, run
``nglider_knowndepth_flat_fixed2_stable_s0``).

The core subclasses the training env itself (``stable_env.py`` is an unmodified
copy of ``nglider_currents_adcp_nondim_knowndepth_flat_stable.py``), so reset(),
the observation builder, the EKF and the reward are the training code, not a
re-implementation.

New code:
  * ``step_gazebo``: ``step`` with the ASV kinematics replaced by the measured
    WAM-V pose. Everything after the ASV update is copied line for line from
    ``step`` and consumes the RNG in the same order, so feeding it the env's own
    ASV positions reproduces the env exactly (see ``validate_core.py``).
  * ``glider_model``: "env" moves the gliders exactly as in training (constant
    velocity, prescribed synchronized sawtooth dive, DAC every 120 ticks).
    "dave" moves them with Project DAVE's glider flight model (dave_glider.py):
    physical dive depth and glide speeds, a surface interval, and the DAC from
    the glider's own dead reckoning at each actual surfacing. The ASV, EKF,
    observation and reward code is identical in both cases; only where the
    gliders are and how deep they are changes.

Units: the core runs in nondimensional units (length = glider separation sep_m,
speed = ASV max speed 2.5 m/s). Gazebo positions arrive in metres relative to
the WAM-V spawn point; ``gazebo_to_nd`` / ``nd_to_gazebo`` convert using sep_m
and the ASV's reset position, so the env frame (and the current-stripe field)
is never shifted.
"""

import math

import numpy as np

from asv_bridge_stable.dave_glider import DaveGlider, pump_volume_for_speed
from asv_bridge_stable.stable_env import AsvNGliderCurrentsADCPEnvKnownDepthFlatStable

# Evaluation config of the reference run (compare_algo_trajectories_nglider_stable.ipynb),
# with the coverage ratio pinned to rho = 0.4 as in every thesis evaluation.
STABLE_CONFIG = dict(
    dt=0.0375, max_steps=2880, max_cmd=1.0, max_yaw_rate=12.0, world_size=2.5,
    glider_speed=0.12, glider_dive_depth=0.12, glider_separation=1.0, dive_cycle_steps=120,
    comm_range=0.4, comm_range_min=0.4, comm_range_max=0.4,
    r_noise_std=0.0075, q_pos_std=7.5e-5, q_vel_std=8e-4,
    max_pos_std=1.0, obs_unc_scale=0.5,
    spawn_range=0.1, ekf_init_err=0.05, spawn_center_frac=0.25,
    w_unc=1.0, w_comm=1.0, w_ctrl=0.1,
    current_max=0.08, current_stripe_width=0.5, current_num_stripes=200,
    n_gliders_ref_lo=2, n_gliders_ref_hi=5,
    n_gliders_min=2, n_gliders_max=2, n_gliders_cap=2, reward_agg="min",
)

# DAVE glider mission defaults (glider_model="dave").
DAVE_DEFAULTS = dict(
    dave_max_depth_m=1000.0,     # yo-yo depth: deep-rated Slocum limit
    dave_pitch_deg=26.0,         # glide pitch (typical Slocum)
    dave_pump_cc=None,           # None: chosen so the glide speed equals the training glider speed
    dave_surface_time_s=600.0,   # time at the surface per dive (GPS fix, satellite call)
    dave_substep_s=5.0,          # flight-model integration step inside one env tick
)

# Sensing realism (off by default = the training models). All draws come from separate random
# streams seeded from the episode seed, so the mission (current field, glider layout, glider motion)
# is identical with and without them; only the measurements change.
REALISM_DEFAULTS = dict(
    acoustic_model="ideal",       # "ideal" = training channel; "realistic" = the model below
    ac_p_max=0.9,                 # reception probability well inside range
    ac_r50_frac=1.0,              # range (x R_comm) where reception probability is p_max / 2
    ac_width_frac=0.05,           # width (x R_comm) of the reception roll-off
    ac_outlier_p=0.05,            # probability that a received range is a multipath arrival
    ac_outlier_max_frac=0.05,     # multipath excess path: uniform in [0, this x R_comm]
    ac_sound_speed_rel_std=0.005, # per-episode sound-speed error (relative), scales every range
    ac_sound_speed_mps=1500.0,    # assumed sound speed, for the travel-time delay
    ac_dac_loss_p=0.2,            # probability the DAC message is lost on a contact
    adcp_noise=False,             # True = noisy, biased ADCP current at the ASV
    adcp_bias_std_mps=0.02,       # per-episode bias, per axis
    adcp_noise_std_mps=0.03,      # white noise per reading, per axis
    # Current vs depth (DAVE gliders only; the ASV, its hull plugin and the ADCP stay at the surface).
    # "uniform" = training (same current at every depth); "decay" = the surface field scaled by
    # deep_frac + (1 - deep_frac) exp(-z / efold) and rotated by veer (1 - exp(-z / efold)).
    current_profile="uniform",
    current_efold_m=250.0,        # e-folding depth of the surface-intensified part [m]
    current_deep_frac=0.25,       # fraction of the surface current left at depth
    current_veer_deg=30.0,        # rotation of the deep current relative to the surface (+ = CCW)
    # Glider navigation errors (DAVE gliders only): dead reckoning with a compass bias and a speed
    # error, a noisy GPS fix at surfacing and again before each dive, DAC from the dive only.
    glider_nav_error=False,
    gnav_compass_bias_std_deg=2.0,   # per-glider constant compass bias
    gnav_speed_err_std=0.05,         # per-glider relative error of the flight-model speed
    gnav_gps_std_m=5.0,              # GPS fix error, per axis
)

PHYSICAL_MAX_SPEED = 2.5   # WAM-V max surge [m/s] = the speed unit V


def physical_scales(sep_m: float, cfg: dict = STABLE_CONFIG) -> dict:
    """Physical time step, yaw cap, (env-model) dive depth and dive period for a given separation."""
    tau = sep_m / PHYSICAL_MAX_SPEED                     # time unit [s]
    return {
        "tau_s": tau,
        "tick_s": cfg["dt"] * tau,                        # one env step [s]
        "max_yaw_rate_rad_s": cfg["max_yaw_rate"] / tau,  # full yaw action [rad/s]
        "max_surge_m_s": cfg["max_cmd"] * PHYSICAL_MAX_SPEED,
        "dive_depth_m": cfg["glider_dive_depth"] * sep_m,
        "dive_period_s": cfg["dive_cycle_steps"] * cfg["dt"] * tau,
        "comm_range_m": cfg["comm_range"] * sep_m,
    }


class EkfCoreStable(AsvNGliderCurrentsADCPEnvKnownDepthFlatStable):

    def __init__(self, config: dict | None = None, sep_m: float = 2000.0, glider_model: str = "env"):
        cfg = {**STABLE_CONFIG, **DAVE_DEFAULTS, **REALISM_DEFAULTS, **(config or {})}
        super().__init__(cfg)
        self.sep_m = float(sep_m)
        self.tick_s = self.dt * self.sep_m / PHYSICAL_MAX_SPEED
        self.glider_model = glider_model
        if glider_model not in ("env", "dave"):
            raise ValueError("glider_model must be 'env' or 'dave'")
        self.dave_cfg = {k: cfg[k] for k in DAVE_DEFAULTS}
        self.real_cfg = {k: cfg[k] for k in REALISM_DEFAULTS}
        if self.real_cfg["acoustic_model"] not in ("ideal", "realistic"):
            raise ValueError("acoustic_model must be 'ideal' or 'realistic'")
        if self.real_cfg["current_profile"] not in ("uniform", "decay"):
            raise ValueError("current_profile must be 'uniform' or 'decay'")
        self.adcp_reading = np.zeros(2, dtype=np.float32)   # last ADCP current in the observation (nd)
        if self.dave_cfg["dave_pump_cc"] is None:
            self.dave_cfg["dave_pump_cc"] = pump_volume_for_speed(
                self.glider_speed_mean * PHYSICAL_MAX_SPEED, math.radians(self.dave_cfg["dave_pitch_deg"]))
        self.dave = []
        # Continuous glider motion: the DAVE model runs on a fixed substep grid inside each tick.
        # advance_gliders_to() moves the gliders part-way through a tick (for display); the tick
        # itself (_advance_gliders) finishes the remaining substeps, so the result is identical
        # to integrating the whole tick at once.
        self._sub_h = self.tick_s / max(1, int(round(self.tick_s / self.dave_cfg["dave_substep_s"])))
        self._subs_per_tick = int(round(self.tick_s / self._sub_h))
        self._subs_done = 0
        self._surfaced_pending = []
        self.origin_nd = np.zeros(2, dtype=np.float32)   # env-frame position of the Gazebo origin
        self.last_info = {}

    # Frames
    def reset_core(self, seed=None):
        """Reset the env and anchor the Gazebo origin (WAM-V spawn) at the ASV's reset position."""
        obs, _ = self.reset(seed=seed)
        self.origin_nd = self.asv_pos.copy()
        self.last_info = {}
        # separate streams: acoustic channel + per-tick ADCP readings, and the 10 Hz ADCP stream
        base = None if seed is None else int(seed)
        self._real_rng = np.random.default_rng(None if base is None else base + 7919)
        self._adcp_stream_rng = np.random.default_rng(None if base is None else base + 104729)
        rc = self.real_cfg
        self.sound_speed_scale = float(1.0 + self._real_rng.normal(0.0, rc["ac_sound_speed_rel_std"]))
        self.adcp_bias = (self._real_rng.normal(0.0, rc["adcp_bias_std_mps"], size=2)
                          / PHYSICAL_MAX_SPEED).astype(np.float32)   # nd
        self._asv_prev = self.asv_pos.copy()
        if rc["adcp_noise"]:
            obs = self._get_obs()
        if self.glider_model == "dave":
            c = self.dave_cfg
            self.dave = [DaveGlider(self.g_pos[i].astype(np.float64) * self.sep_m,
                                    math.atan2(self.g_vel[i, 1], self.g_vel[i, 0]),
                                    c["dave_max_depth_m"], math.radians(c["dave_pitch_deg"]),
                                    c["dave_pump_cc"], c["dave_surface_time_s"])
                         for i in range(self.n_gliders)]
            if rc["glider_nav_error"]:
                # own stream: the mission and the other realism draws are unchanged
                nav_rng = np.random.default_rng(None if base is None else base + 15485863)
                for g in self.dave:
                    g.nav_heading_bias = math.radians(nav_rng.normal(0.0, rc["gnav_compass_bias_std_deg"]))
                    g.nav_speed_scale = 1.0 + nav_rng.normal(0.0, rc["gnav_speed_err_std"])
                    g.gps_std_m = rc["gnav_gps_std_m"]
                    g.nav_rng = nav_rng
                    g.realistic_nav = True
                    g.dive_start_fix()
            self._depth = 0.0   # both gliders start at the surface, as in the env
            self._subs_done = 0
            self._surfaced_pending = [False] * self.n_gliders
            obs = self._get_obs()
        return obs

    def gazebo_to_nd(self, pos_m):
        return (np.asarray(pos_m, dtype=np.float64) / self.sep_m + self.origin_nd).astype(np.float32)

    def nd_to_gazebo(self, pos_nd):
        return (np.asarray(pos_nd, dtype=np.float64) - self.origin_nd) * self.sep_m

    # ASV kinematics of the training env (kinematic / no-Gazebo mode)
    def kinematic_asv(self, action):
        """The env's ASV update, returned instead of applied: (new position nd, new heading)."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        surge = ((action[0] + 1.0) / 2.0) * self.max_cmd
        yaw_rate = action[1] * self.max_yaw_rate
        heading = self.asv_heading + yaw_rate * self.dt
        heading = (heading + np.pi) % (2 * np.pi) - np.pi
        vx = surge * np.cos(heading)
        vy = surge * np.sin(heading)
        asv_current = self._get_current(self.asv_pos)
        pos = self.asv_pos + np.array([vx + asv_current[0], vy + asv_current[1]], dtype=np.float32) * self.dt
        return pos, heading

    # Glider motion
    def _advance_gliders(self):
        n = self.n_gliders
        if self.glider_model == "env":
            # identical to AsvNGliderCurrentsADCPEnvKnownDepthFlatStable.step
            for i in range(n):
                gc = self._get_current(self.g_pos[i])
                self.g_pos[i] = self.g_pos[i] + (self.g_vel[i] + gc) * self.dt
            self._depth = self._glider_depth_at(self.step_count + 1)

            self.g_dr_pos = self.g_dr_pos + self.g_vel * self.dt

            if self.step_count > 0 and self.step_count % self.dive_cycle_steps == 0:
                dive_time = self.dive_cycle_steps * self.dt
                self.g_internal_dac = (self.g_pos - self.g_dr_pos) / dive_time
                self.g_dr_pos = self.g_pos.copy()
            return

        # DAVE flight model: finish this tick's remaining substeps
        self._integrate_dave(self._subs_per_tick)
        for i, g in enumerate(self.dave):
            if self._surfaced_pending[i] and g.dac is not None:
                # The glider's own DAC estimate, handed to the EKF at the next acoustic contact (as in the env).
                self.g_internal_dac[i] = (g.dac / PHYSICAL_MAX_SPEED).astype(np.float32)
        self._surfaced_pending = [False] * n
        self._subs_done = 0

    def _integrate_dave(self, target_subs):
        """Run DAVE substeps until target_subs of this tick are done; update g_pos and depth."""
        while self._subs_done < min(target_subs, self._subs_per_tick):
            for i, g in enumerate(self.dave):
                cur_mps = self.current_at_depth_mps(g.pos / self.sep_m, g.depth)
                g.advance(self._sub_h, cur_mps)
                self._surfaced_pending[i] |= g.surfaced_event
            self._subs_done += 1
        for i, g in enumerate(self.dave):
            self.g_pos[i] = (g.pos / self.sep_m).astype(np.float32)
        depths = [g.depth for g in self.dave]
        # The observation and EKF use one known depth for the fleet (the env's gliders dive in sync).
        # Both DAVE gliders get identical commands from the surface, so their depths stay identical.
        assert max(depths) - min(depths) < 1e-6, f"glider depths diverged: {depths}"
        self._depth = depths[0] / self.sep_m

    def current_profile_factor(self, depth_m):
        """(scale, rotation [rad]) of the current at depth_m relative to the surface current."""
        rc = self.real_cfg
        if rc["current_profile"] == "uniform":
            return 1.0, 0.0
        e = math.exp(-max(float(depth_m), 0.0) / rc["current_efold_m"])
        return (rc["current_deep_frac"] + (1.0 - rc["current_deep_frac"]) * e,
                math.radians(rc["current_veer_deg"]) * (1.0 - e))

    def current_at_depth_mps(self, pos_nd, depth_m):
        """Horizontal ocean current [m/s, ENU] at pos_nd and depth_m (the surface field for 'uniform')."""
        c = np.asarray(self._get_current(pos_nd), dtype=np.float64) * PHYSICAL_MAX_SPEED
        if self.real_cfg["current_profile"] == "uniform":
            return c
        s, a = self.current_profile_factor(depth_m)
        ca, sa = math.cos(a), math.sin(a)
        return s * np.array([ca * c[0] - sa * c[1], sa * c[0] + ca * c[1]])

    def advance_gliders_to(self, seconds_into_tick):
        """Move the DAVE gliders to `seconds_into_tick` of the current tick (between env ticks).

        Only the glider motion advances; the EKF, observation and reward still update once per
        tick. Positions are snapped to the substep grid, and the tick completes the rest.
        """
        if self.glider_model != "dave":
            return
        self._integrate_dave(int(seconds_into_tick // self._sub_h))

    # Gazebo step
    def step_gazebo(self, asv_pos_nd, asv_heading, action):
        """``step`` with the ASV pose taken from Gazebo instead of integrated.

        asv_pos_nd: ASV position in the env frame (nondimensional), e.g. from gazebo_to_nd().
        asv_heading: ASV yaw [rad], ENU (0 = +x).
        action: the action the policy issued for this tick (goes into last_cmd and r_ctrl).
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.last_cmd = action.copy()
        n = self.n_gliders

        # Replaces the env's ASV kinematics
        self.asv_pos = np.asarray(asv_pos_nd, dtype=np.float32).copy()
        self.asv_heading = float((asv_heading + np.pi) % (2 * np.pi) - np.pi)

        self._advance_gliders()

        # From here on: identical to AsvNGliderCurrentsADCPEnvKnownDepthFlatStable.step
        for i in range(n):
            self.g_x[i], self.g_P[i] = self._ekf_predict(
                self.g_x[i], self.g_P[i], self.g_last_known_current[i])

        true_dist = np.array(
            [np.sqrt(self._distance(self.asv_pos, self.g_pos[i]) ** 2 + self._depth ** 2)
             for i in range(n)], dtype=np.float32)
        received = np.zeros(n, dtype=bool)
        if self.real_cfg["acoustic_model"] == "ideal":
            for i in range(n):
                if true_dist[i] <= self.comm_range:
                    self.g_x[i], self.g_P[i] = self._ekf_update(
                        self.g_x[i], self.g_P[i], self.g_pos[i], self._depth)
                    self.g_last_known_current[i] = self.g_internal_dac[i].copy()
                    received[i] = True
        else:
            received = self._acoustic_contacts(true_dist)
        self._asv_prev = self.asv_pos.copy()

        self._sync_glider_estimates()
        self.t += self.dt
        self.step_count += 1

        det_P = np.array([max(float(np.linalg.det(self.g_P[i, 0:2, 0:2])), 1e-18)
                          for i in range(n)])
        r_unc_each = -np.log(det_P)
        r_unc = self._agg_unc(r_unc_each)
        r_ctrl = -float(np.sum(action ** 2))
        excess = np.clip((true_dist - self.comm_range) / self.comm_range, 0.0, None)
        r_comm = -float(np.mean(excess))
        reward = self.w_unc * r_unc + self.w_comm * r_comm + self.w_ctrl * r_ctrl
        # end of copied block
        self.last_info = {
            "reward": float(reward), "r_unc": r_unc, "r_comm": r_comm, "r_ctrl": r_ctrl,
            "dist_each": true_dist, "det_P_each": det_P.astype(np.float32),
            "depth": float(self._depth),
            "in_range_frac": float(np.mean(true_dist <= self.comm_range)),
            "received": received.astype(np.float32),
        }
        return self._get_obs()

    # Sensing realism
    def reception_probability(self, slant_nd):
        """Probability that a ping/reply exchange succeeds at slant range `slant_nd` (realistic channel)."""
        rc = self.real_cfg
        r50 = rc["ac_r50_frac"] * self.comm_range
        w = max(rc["ac_width_frac"] * self.comm_range, 1e-9)
        return rc["ac_p_max"] / (1.0 + np.exp((np.asarray(slant_nd) - r50) / w))

    def _acoustic_contacts(self, true_dist):
        """Realistic acoustic channel: range-dependent reception, two-way travel-time delay, sound-speed
        error, multipath outliers, and lost DAC messages. Returns which gliders were received."""
        rc, rng = self.real_cfg, self._real_rng
        n = self.n_gliders
        received = np.zeros(n, dtype=bool)
        asv_vel = (self.asv_pos - self._asv_prev) / self.dt          # nd velocity over the last tick
        for i in range(n):
            if rng.random() >= self.reception_probability(true_dist[i]):
                continue
            # Two-way ranging completed at the tick: the range corresponds to the ASV ~one travel time ago.
            delay = true_dist[i] * PHYSICAL_MAX_SPEED / rc["ac_sound_speed_mps"]      # nd time
            asv_meas = self.asv_pos - asv_vel * delay
            slant = np.sqrt(self._distance(asv_meas, self.g_pos[i]) ** 2 + self._depth ** 2)
            z = slant * self.sound_speed_scale + self._rng.normal(0, self.r_noise_std)
            if rng.random() < rc["ac_outlier_p"]:
                z += rng.uniform(0.0, rc["ac_outlier_max_frac"] * self.comm_range)
            self.g_x[i], self.g_P[i] = self._ekf_update_z(self.g_x[i], self.g_P[i], z, self._depth)
            if rng.random() >= rc["ac_dac_loss_p"]:
                self.g_last_known_current[i] = self.g_internal_dac[i].copy()
            received[i] = True
        return received

    def _ekf_update_z(self, x, P, z, depth):
        """The env's _ekf_update with the measured range z supplied (same EKF math)."""
        dx = x[0] - self.asv_pos[0]
        dy = x[1] - self.asv_pos[1]
        z_hat = np.sqrt(dx ** 2 + dy ** 2 + depth ** 2)
        z_hat_safe = max(z_hat, 1e-6)
        H = np.zeros((1, 4), dtype=np.float32)
        H[0, 0] = dx / z_hat_safe
        H[0, 1] = dy / z_hat_safe
        y = np.array([z - z_hat_safe], dtype=np.float32)
        S = H @ P @ H.T + self.R
        K = P @ H.T @ np.linalg.inv(S)
        x_new = x + (K @ y).flatten()
        I_KH = np.eye(4, dtype=np.float32) - K @ H
        P_new = I_KH @ P @ I_KH.T + K @ self.R @ K.T
        P_new = 0.5 * (P_new + P_new.T)
        for j in range(4):
            if P_new[j, j] < 1e-9:
                P_new[j, j] = 1e-9
        return x_new, P_new

    def _get_obs(self):
        obs = super()._get_obs()
        true_c = np.asarray(self._get_current(self.asv_pos), dtype=np.float32)
        if hasattr(self, "real_cfg") and self.real_cfg["adcp_noise"] and hasattr(self, "_real_rng"):
            noise = self._real_rng.normal(0.0, self.real_cfg["adcp_noise_std_mps"], size=2) / PHYSICAL_MAX_SPEED
            self.adcp_reading = (true_c + self.adcp_bias + noise).astype(np.float32)
            nf = max(self.current_max, 1e-6)
            obs[3] = np.clip(self.adcp_reading[0] / nf, -1.0, 1.0)
            obs[4] = np.clip(self.adcp_reading[1] / nf, -1.0, 1.0)
        else:
            self.adcp_reading = true_c
        return obs

    def adcp_sample_mps(self, pos_nd):
        """One ADCP reading [m/s, ENU] at pos_nd for the 10 Hz speed loop (its own random stream)."""
        c = np.asarray(self._get_current(pos_nd), dtype=float) * PHYSICAL_MAX_SPEED
        if not self.real_cfg["adcp_noise"]:
            return c
        return (c + self.adcp_bias * PHYSICAL_MAX_SPEED
                + self._adcp_stream_rng.normal(0.0, self.real_cfg["adcp_noise_std_mps"], size=2))

    def step_kinematic(self, action):
        """Training-env ASV kinematics + this core's glider model (no Gazebo)."""
        pos, heading = self.kinematic_asv(action)
        return self.step_gazebo(pos, heading, action)

    # Convenience for the ROS nodes
    def glider_state_gazebo(self):
        """True and estimated glider positions in the Gazebo frame [m], depth [m], headings, pitches."""
        depth_m = self._depth * self.sep_m
        true_xy = np.array([self.nd_to_gazebo(p) for p in self.g_pos])
        est_xy = np.array([self.nd_to_gazebo(p) for p in self.g_est_pos])
        if self.glider_model == "dave":
            heading = np.array([g.heading for g in self.dave])
            pitch = np.array([g.pitch for g in self.dave])
        else:
            heading = np.array([np.arctan2(v[1], v[0]) for v in self.g_vel])
            phase = (self.step_count % self.dive_cycle_steps) / self.dive_cycle_steps
            pitch = np.full(self.n_gliders, 0.45 if phase < 0.5 else -0.45)   # REP-103: + = nose down
        return true_xy, est_xy, depth_m, heading, pitch

    def glider_modes(self):
        return [g.mode for g in self.dave] if self.glider_model == "dave" else ["env"] * self.n_gliders
