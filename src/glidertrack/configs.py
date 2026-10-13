"""Environment configs for the two reference cases and the physical reference scale."""
from __future__ import annotations

# reference scale: 20 km fleet span, 2.5 m/s ASV top speed
LENGTH_M = 20_000.0
SPEED_MPS = 2.5
TIME_S = LENGTH_M / SPEED_MPS  # 8000 s, so one 0.0375 step is 300 s

_BASE = dict(
    dt=0.0375,
    max_steps=2880,  # 240 h
    max_cmd=1.0,
    max_yaw_rate=12.0,
    world_size=2.5,
    glider_speed=0.12,  # 0.30 m/s
    glider_dive_depth=0.12,  # 2400 m
    glider_separation=1.0,
    dive_cycle_steps=120,  # 10 h
    comm_range=0.4,
    comm_range_min=0.30,
    comm_range_max=0.60,
    r_noise_std=0.0075,  # 150 m
    q_pos_std=7.5e-5,
    q_vel_std=8.0e-4,
    max_pos_std=1.0,
    obs_unc_scale=0.5,
    spawn_range=0.1,
    ekf_init_err=0.05,  # 1 km
    spawn_center_frac=0.25,
    w_unc=1.0,
    w_comm=1.0,
    w_ctrl=0.1,
    current_max=0.08,  # 0.2 m/s
    current_stripe_width=0.5,
    current_num_stripes=200,
    n_gliders_ref_lo=2,
    n_gliders_ref_hi=5,
    obs_slot_assignment="canonical",
)

# two gliders, reward on the worse-tracked one
TWO_GLIDER = dict(_BASE, n_gliders_min=2, n_gliders_max=2, n_gliders_cap=2, reward_agg="min")

# 2-5 gliders over the same span, softmin reward over the fleet
MULTI_GLIDER = dict(_BASE, n_gliders_min=2, n_gliders_max=5, n_gliders_cap=5, reward_agg="softmin", softmin_beta=1.0)

CASES = {"two_glider": TWO_GLIDER, "multi_glider": MULTI_GLIDER}


def eval_config(case: str, rho: float = 0.4, n_gliders: int | None = None, **overrides) -> dict:
    """Fixed coverage ratio (and optionally fleet size) for evaluation."""
    cfg = dict(CASES[case], comm_range=rho, comm_range_min=rho, comm_range_max=rho)
    if n_gliders is not None:
        cfg.update(n_gliders_min=n_gliders, n_gliders_cap=n_gliders)
    cfg.update(overrides)
    return cfg
