"""
Stability-focused variant of ``nglider_currents_adcp_nondim_knowndepth_flat.py``,
built to remove a specific failure mode diagnosed in the reference SAC policy
(``twoglider_currents_nondim_3d_knowndepth_v1``) on held-out seed 907: once both
gliders are simultaneously out of acoustic range for long enough that both EKF
covariances saturate at the environment's clamp, the trained policy's yaw
command pins near its maximum magnitude, and because ``max_yaw_rate`` is
enormous relative to any heading correction this task ever needs, that
produces multiple full rotations per single environment step (a literal spin,
not a gentle loop), while a persistent negative-surge (reverse-thrust) habit
was already present for hundreds of steps before the collapse became visible.
None of the three non-learned baseline controllers compared against this
policy can enter this state, structurally, because their steering law is
hard-capped to a tiny, physically sane fraction of ``max_yaw_rate`` and their
surge is always full-forward. This file makes the same guarantees available
to a trained policy by construction, not just as a habit training happens to
learn:

  1. ``max_yaw_rate`` default lowered from 400.0 to 12.0. At dt=0.0375 this
     caps the maximum possible heading change to 12*0.0375 = 0.45 rad
     (~26 degrees) per step at full yaw command, comfortably above the
     non-learned baselines' own effective operating cap (their pursuit law
     never exceeds +/-3.0 rad per unit ND time) while making a >360-degree
     per-step spin mathematically impossible regardless of what the policy
     outputs. This also means the full [-1, 1] yaw action range now maps to
     physically meaningful turn rates throughout, rather than the reference
     env where 99%+ of that range corresponded to turn rates no sane
     controller would ever use.
  2. Surge restricted to forward-only. ``action[0] in [-1, 1]`` now maps to
     ``surge in [0, max_cmd]`` (linearly, -1 -> stop, +1 -> full forward)
     instead of ``[-max_cmd, max_cmd]``. Reverse thrust has no legitimate use
     in a pursuit task and was observed active for hundreds of steps before
     the diagnosed collapse became visible; removing it removes an entire
     dimension of pathological behaviour rather than hoping training learns
     not to use it.
  3. Observation saturation loosened for the two features that ran away
     during the diagnosed failure (distance and covariance-derived
     uncertainty). The reference env hard-clips both once they exceed a
     fixed scale, so a policy that ends up in a badly-lost state receives
     literally the same observation value regardless of how much worse
     things get from there, with no gradient at all pointing back toward
     recovery. This file replaces the hard clip with a smooth
     ``tanh``-based squashing (still bounded to [-1, 1], never changes the
     value near the operating point) that keeps a small but nonzero
     gradient all the way to the extreme tail, so "very lost" and
     "catastrophically lost" are no longer observationally identical.

Physics, EKF, and reward are otherwise byte-for-byte identical to
``nglider_currents_adcp_nondim_knowndepth_flat.py`` (see that file's
docstring for the full description shared with
``AsvTwoGliderCurrentsADCPEnv3DKnownDepth``).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

GLIDER_FEATURES = 9   # d_est, d_rate, los_x, los_y, los_z, cov_major, cov_minor, cov_angle, est_heading
GLOBAL_FEATURES = 7   # asv_heading, last_cmd(2), asv_current(2), rho_norm, n_gliders_norm


class AsvNGliderCurrentsADCPEnvKnownDepthFlatStable(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}

        # --- World (nondimensional defaults) ---
        self.dt = float(cfg.get("dt", 0.0375))
        self.max_steps = int(cfg.get("max_steps", 2880))
        self.world_size = float(cfg.get("world_size", 2.5))

        # --- ASV Kinematics ---
        self.max_cmd = float(cfg.get("max_cmd", 1.0))
        self.max_yaw_rate = float(cfg.get("max_yaw_rate", 12.0))   # was 400.0

        # --- Fleet size (see module docstring: fixed_n vs variable_n) ---
        _n_default = int(cfg.get("n_gliders", 2))
        self.n_gliders_min = int(cfg.get("n_gliders_min", _n_default))
        self.n_gliders_max = int(cfg.get("n_gliders_max", _n_default))
        self.n_gliders_cap = int(cfg.get("n_gliders_cap", self.n_gliders_max))
        self.n_ref_lo = int(cfg.get("n_gliders_ref_lo", 2))
        self.n_ref_hi = int(cfg.get("n_gliders_ref_hi", 5))
        self.n_gliders = min(_n_default, self.n_gliders_max)   # resampled every reset()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        flat_width = GLOBAL_FEATURES + self.n_gliders_max * GLIDER_FEATURES + self.n_gliders_max
        self.observation_space = spaces.Box(-1.0, 1.0, (flat_width,), dtype=np.float32)

        # --- Glider parameters ---
        self.glider_speed_mean = float(cfg.get("glider_speed", 0.12))
        self.glider_separation = float(cfg.get("glider_separation", 1.0))

        # --- Dive cycle: sawtooth surface -> glider_dive_depth -> surface,
        #     ALL gliders in sync (matches knowndepth_v1's g1/g2 always diving
        #     together) ---
        self.dive_cycle_steps = int(cfg.get("dive_cycle_steps", 120))
        self.glider_dive_depth = float(cfg.get("glider_dive_depth",
                                               cfg.get("glider_depth", 0.12)))

        # --- Acoustic / coverage (randomized per episode) ---
        self.comm_range = float(cfg.get("comm_range", 0.4))
        self.comm_range_min = float(cfg.get("comm_range_min", self.comm_range))
        self.comm_range_max = float(cfg.get("comm_range_max", self.comm_range))

        # --- EKF & noise (horizontal only -- depth is known) ---
        self.q_pos_std = float(cfg.get("q_pos_std", 7.5e-5))
        self.q_vel_std = float(cfg.get("q_vel_std", 8.0e-4))
        self.r_noise_std = float(cfg.get("r_noise_std", 0.0075))
        self.max_pos_std = float(cfg.get("max_pos_std", 1.0))
        self.obs_unc_scale = float(cfg.get("obs_unc_scale", 0.5))

        # --- Reward weights + aggregation ---
        self.w_unc = float(cfg.get("w_unc", 1.0))
        self.w_comm = float(cfg.get("w_comm", 1.0))
        self.w_ctrl = float(cfg.get("w_ctrl", 0.1))
        self.reward_agg = str(cfg.get("reward_agg", "min")).lower()
        self.softmin_beta = float(cfg.get("softmin_beta", 1.0))

        # --- Spawn ---
        self.spawn_range = float(cfg.get("spawn_range", 0.1))
        self.ekf_init_err = float(cfg.get("ekf_init_err", 0.05))
        self.spawn_center_frac = float(cfg.get("spawn_center_frac", 0.25))

        # --- Ocean Currents (2-D horizontal stripe field) ---
        self.current_max = float(cfg.get("current_max", 0.08))
        self.current_stripe_width = float(cfg.get("current_stripe_width", 0.5))
        self.current_num_stripes = int(cfg.get("current_num_stripes", 200))
        self.current_velocities = np.zeros((self.current_num_stripes, 2), dtype=np.float32)

        # --- ASV state (z pinned to 0: surface vehicle) ---
        self.asv_pos = np.zeros(2, dtype=np.float32)
        self.asv_heading = 0.0
        self.last_cmd = np.zeros(2, dtype=np.float32)

        # --- Per-glider arrays (allocated in reset() to the episode's N) ---
        self.g_pos = None
        self.g_vel = None
        self.g_dr_pos = None
        self.g_internal_dac = None
        self.g_last_known_current = None
        self.g_x = None
        self.g_P = None
        self.g_est_pos = None
        self.g_est_vel = None
        self.g_unc = None
        self.prev_d_est = None

        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt
        self.Q = np.diag([
            self.q_pos_std ** 2, self.q_pos_std ** 2,
            self.q_vel_std ** 2, self.q_vel_std ** 2,
        ]).astype(np.float32)
        self.R = np.array([[self.r_noise_std ** 2]], dtype=np.float32)

        self._rng = np.random.default_rng()
        self.t = 0.0
        self.step_count = 0
        self._depth = 0.0

    # Utilities (identical to the Dict/Deep-Sets version)
    @staticmethod
    def _distance(a, b):
        return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))

    def _glider_depth_at(self, step_count):
        if self.dive_cycle_steps <= 0:
            return 0.0
        phase = (step_count % self.dive_cycle_steps) / self.dive_cycle_steps
        return float(self.glider_dive_depth * (1.0 - abs(2.0 * phase - 1.0)))

    def _get_current(self, pos):
        half_world = self.current_num_stripes * self.current_stripe_width / 2.0
        stripe_idx = int((pos[0] + half_world) / self.current_stripe_width)
        stripe_idx = np.clip(stripe_idx, 0, self.current_num_stripes - 1)
        return self.current_velocities[stripe_idx]

    def _clamp_covariance(self, P):
        max_var = self.max_pos_std ** 2
        P_clamped = P.copy()
        for i in range(2):
            if P_clamped[i, i] > max_var:
                P_clamped[i, :] = 0.0
                P_clamped[:, i] = 0.0
                P_clamped[i, i] = max_var
        return P_clamped

    def _ekf_predict(self, x, P, known_current):
        x_new = self.F @ x
        x_new[0] += known_current[0] * self.dt
        x_new[1] += known_current[1] * self.dt
        P_new = self._clamp_covariance(self.F @ P @ self.F.T + self.Q)
        return x_new, P_new

    def _ekf_update(self, x, P, glider_true_xy, glider_true_depth):
        true_slant = np.sqrt(self._distance(self.asv_pos, glider_true_xy) ** 2
                             + glider_true_depth ** 2)
        z = true_slant + self._rng.normal(0, self.r_noise_std)

        dx = x[0] - self.asv_pos[0]
        dy = x[1] - self.asv_pos[1]
        dz_known = glider_true_depth
        z_hat = np.sqrt(dx ** 2 + dy ** 2 + dz_known ** 2)
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
        for i in range(4):
            if P_new[i, i] < 1e-9:
                P_new[i, i] = 1e-9
        return x_new, P_new

    def _sync_glider_estimates(self):
        for i in range(self.n_gliders):
            self.g_est_pos[i] = self.g_x[i, 0:2]
            self.g_est_vel[i] = self.g_x[i, 2:4]
            self.g_unc[i] = np.sqrt(max(self.g_P[i, 0, 0] + self.g_P[i, 1, 1], 0.0))

    def _cov_to_obs(self, P):
        # Same eigen-decomposition as the reference env, but squashed with
        # tanh instead of hard-clipped, so uncertainty far beyond
        # obs_unc_scale is still distinguishable from "just over" it instead
        # of both reading exactly 1.0.
        P_pos = P[0:2, 0:2]
        eigvals, eigvecs = np.linalg.eigh(P_pos)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        major_std = np.sqrt(max(eigvals[0], 0.0))
        minor_std = np.sqrt(max(eigvals[1], 0.0))
        major_angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        major_norm = np.tanh(major_std / self.obs_unc_scale)
        minor_norm = np.tanh(minor_std / self.obs_unc_scale)
        return major_norm * 2.0 - 1.0, minor_norm * 2.0 - 1.0, major_angle / np.pi

    def _agg_unc(self, r):
        r = np.asarray(r, dtype=np.float64)
        if self.reward_agg == "min" or self.softmin_beta <= 0:
            return float(r.min())
        b = self.softmin_beta
        return float(-np.logaddexp.reduce(-b * r) / b + np.log(len(r)) / b)

    # Reset (identical to the Dict/Deep-Sets version)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        hi = int(np.clip(self.n_gliders_cap, self.n_gliders_min, self.n_gliders_max))
        if hi > self.n_gliders_min:
            self.n_gliders = int(self._rng.integers(self.n_gliders_min, hi + 1))
        else:
            self.n_gliders = int(self.n_gliders_min)
        n = self.n_gliders

        if self.comm_range_max > self.comm_range_min:
            self.comm_range = float(self._rng.uniform(self.comm_range_min, self.comm_range_max))

        self.current_velocities = np.zeros((self.current_num_stripes, 2), dtype=np.float32)
        if self.current_max > 0:
            current = self._rng.uniform(-self.current_max, self.current_max, size=2)
            for i in range(self.current_num_stripes):
                self.current_velocities[i] = current
                stp = self._rng.normal(0, self.current_max * 0.3, size=2)
                current = np.clip(current + stp, -self.current_max, self.current_max)

        cr = self.spawn_center_frac * self.glider_separation
        center = self._rng.uniform(-cr, cr, size=2).astype(np.float32)
        base_angle = self._rng.uniform(0, 2 * np.pi)
        heading_vec = np.array([np.cos(base_angle), np.sin(base_angle)], dtype=np.float32)
        perp_vec = np.array([-np.sin(base_angle), np.cos(base_angle)], dtype=np.float32)
        if n == 1:
            offsets = np.array([0.0], dtype=np.float32)
        else:
            offsets = np.linspace(-0.5, 0.5, n).astype(np.float32) * self.glider_separation

        self.g_pos = np.stack([center + perp_vec * o for o in offsets]).astype(np.float32)
        self._depth = self._glider_depth_at(0)
        self.g_vel = np.tile(heading_vec * self.glider_speed_mean, (n, 1)).astype(np.float32)
        self.g_dr_pos = self.g_pos.copy()

        asv_xy = center + self._rng.uniform(-self.spawn_range, self.spawn_range, size=2)
        self.asv_pos = asv_xy.astype(np.float32)
        self.asv_heading = float(self._rng.uniform(-np.pi, np.pi))

        vel_var = (0.1 * self.glider_speed_mean) ** 2
        self.g_x = np.zeros((n, 4), dtype=np.float32)
        self.g_P = np.zeros((n, 4, 4), dtype=np.float32)
        for i in range(n):
            err = self._rng.normal(0, self.ekf_init_err, size=2)
            self.g_x[i] = [self.g_pos[i, 0] + err[0], self.g_pos[i, 1] + err[1],
                           self.g_vel[i, 0], self.g_vel[i, 1]]
            self.g_P[i] = np.diag([self.ekf_init_err ** 2, self.ekf_init_err ** 2,
                                   vel_var, vel_var])

        init_current = self._get_current(self.asv_pos).copy()
        self.g_internal_dac = np.tile(init_current, (n, 1)).astype(np.float32)
        self.g_last_known_current = np.tile(init_current, (n, 1)).astype(np.float32)

        self.g_est_pos = np.zeros((n, 2), dtype=np.float32)
        self.g_est_vel = np.zeros((n, 2), dtype=np.float32)
        self.g_unc = np.zeros(n, dtype=np.float32)
        self._sync_glider_estimates()

        self.prev_d_est = np.array(
            [self._distance(self.asv_pos, self.g_est_pos[i]) for i in range(n)], dtype=np.float32)
        self.last_cmd = np.zeros(2, dtype=np.float32)
        self.t = 0.0
        self.step_count = 0
        return self._get_obs(), {}

    # Observation -- FLAT, zero-padded
    def _get_obs(self):
        n = self.n_gliders
        nmax = self.n_gliders_max
        max_dist = self.world_size
        max_rate = self.max_cmd + self.glider_speed_mean + self.current_max
        depth_scale = max(self.glider_dive_depth, 1e-6)

        gliders = np.zeros((nmax, GLIDER_FEATURES), dtype=np.float32)
        mask = np.zeros(nmax, dtype=np.float32)

        for i in range(n):
            d_est = np.sqrt(self._distance(self.asv_pos, self.g_est_pos[i]) ** 2
                            + self._depth ** 2)
            d_rate = (d_est - self.prev_d_est[i]) / self.dt if self.step_count > 0 else 0.0
            self.prev_d_est[i] = d_est
            los_xy = self.g_est_pos[i] - self.asv_pos
            cmaj, cmin, cang = self._cov_to_obs(self.g_P[i])
            ev = self.g_est_vel[i]
            est_head = np.arctan2(ev[1], ev[0]) if np.linalg.norm(ev) > 1e-6 else 0.0
            # Distance and line-of-sight components: tanh-squashed instead of
            # hard-clipped, same reasoning as _cov_to_obs above -- keeps a
            # small gradient pointing back toward the target even far beyond
            # max_dist, instead of every "very far" state reading identically.
            gliders[i] = [
                np.tanh(d_est / max_dist) * 2.0 - 1.0,
                np.clip(d_rate / max_rate, -1, 1),
                np.tanh(los_xy[0] / max_dist),
                np.tanh(los_xy[1] / max_dist),
                np.clip(self._depth / depth_scale, -1, 1),
                cmaj, cmin, cang,
                est_head / np.pi,
            ]
            mask[i] = 1.0

        asv_current = self._get_current(self.asv_pos)
        norm_factor = max(self.current_max, 1e-6)
        rho = self.comm_range / max(self.glider_separation, 1e-9)
        n_span = max(self.n_ref_hi - self.n_ref_lo, 1)
        n_norm = np.clip((n - self.n_ref_lo) / n_span * 2.0 - 1.0, -1.0, 1.0)

        glob = np.array([
            self.asv_heading / np.pi,
            self.last_cmd[0],
            self.last_cmd[1],
            np.clip(asv_current[0] / norm_factor, -1.0, 1.0),
            np.clip(asv_current[1] / norm_factor, -1.0, 1.0),
            float(np.clip((rho - 0.45) / 0.20, -1.0, 1.0)),
            float(n_norm),
        ], dtype=np.float32)

        flat = np.concatenate([
            np.clip(glob, -1.0, 1.0),
            np.clip(gliders, -1.0, 1.0).reshape(-1),
            mask,
        ]).astype(np.float32)
        return np.clip(flat, -1.0, 1.0)

    # Step (identical physics to the Dict/Deep-Sets version, except the
    #     ASV kinematics below: forward-only surge, capped yaw rate)
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.last_cmd = action.copy()
        n = self.n_gliders

        # Forward-only surge: -1 -> stop, +1 -> full forward. Reverse thrust
        # has no legitimate use in this pursuit task (see module docstring).
        surge = ((action[0] + 1.0) / 2.0) * self.max_cmd
        yaw_rate = action[1] * self.max_yaw_rate
        self.asv_heading += yaw_rate * self.dt
        self.asv_heading = (self.asv_heading + np.pi) % (2 * np.pi) - np.pi
        vx = surge * np.cos(self.asv_heading)
        vy = surge * np.sin(self.asv_heading)
        asv_current = self._get_current(self.asv_pos)
        self.asv_pos = self.asv_pos + np.array(
            [vx + asv_current[0], vy + asv_current[1]], dtype=np.float32) * self.dt

        for i in range(n):
            gc = self._get_current(self.g_pos[i])
            self.g_pos[i] = self.g_pos[i] + (self.g_vel[i] + gc) * self.dt
        self._depth = self._glider_depth_at(self.step_count + 1)

        self.g_dr_pos = self.g_dr_pos + self.g_vel * self.dt

        if self.step_count > 0 and self.step_count % self.dive_cycle_steps == 0:
            dive_time = self.dive_cycle_steps * self.dt
            self.g_internal_dac = (self.g_pos - self.g_dr_pos) / dive_time
            self.g_dr_pos = self.g_pos.copy()

        for i in range(n):
            self.g_x[i], self.g_P[i] = self._ekf_predict(
                self.g_x[i], self.g_P[i], self.g_last_known_current[i])

        true_dist = np.array(
            [np.sqrt(self._distance(self.asv_pos, self.g_pos[i]) ** 2 + self._depth ** 2)
             for i in range(n)], dtype=np.float32)
        for i in range(n):
            if true_dist[i] <= self.comm_range:
                self.g_x[i], self.g_P[i] = self._ekf_update(
                    self.g_x[i], self.g_P[i], self.g_pos[i], self._depth)
                self.g_last_known_current[i] = self.g_internal_dac[i].copy()

        self._sync_glider_estimates()
        self.t += self.dt
        self.step_count += 1

        det_P = np.array([max(float(np.linalg.det(self.g_P[i, 0:2, 0:2])), 1e-18)
                          for i in range(n)])
        r_unc_each = -np.log(det_P)
        r_unc = self._agg_unc(r_unc_each)
        r_unc_min = float(r_unc_each.min())
        r_ctrl = -float(np.sum(action ** 2))
        excess = np.clip((true_dist - self.comm_range) / self.comm_range, 0.0, None)
        r_comm = -float(np.mean(excess))

        reward = self.w_unc * r_unc + self.w_comm * r_comm + self.w_ctrl * r_ctrl
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), False, truncated, {
            "n_gliders": n,
            "r_unc": r_unc, "r_unc_min": r_unc_min,
            "r_unc_each": r_unc_each.astype(np.float32),
            "r_comm": r_comm, "r_ctrl": r_ctrl,
            "unc_each": self.g_unc.copy(),
            "det_P_each": det_P.astype(np.float32),
            "dist_each": true_dist,
            "depth": float(self._depth),
            "in_range_frac": float(np.mean(true_dist <= self.comm_range)),
            "comm_range": self.comm_range,
            "rho": self.comm_range / max(self.glider_separation, 1e-9),
            "reward_agg": self.reward_agg,
        }

    def get_positions(self):
        return {
            "asv": self.asv_pos.copy(),
            "n": self.n_gliders,
            "depth": self._depth,
            "gliders": self.g_pos.copy(),
            "gliders_est": self.g_est_pos.copy(),
        }

    def get_current_field(self):
        return {
            "velocities": self.current_velocities.copy(),
            "stripe_width": self.current_stripe_width,
            "num_stripes": self.current_num_stripes,
            "max_speed": self.current_max,
        }
