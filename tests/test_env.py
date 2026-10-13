import numpy as np
import pytest

from glidertrack.configs import CASES, eval_config
from glidertrack.env import GLIDER_FEATURES, GLOBAL_FEATURES, GliderTrackingEnv


def rollout(env, seed, steps, action_seed=0):
    rng = np.random.default_rng(action_seed)
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for _ in range(steps):
        obs, *_ = env.step(rng.uniform(-1, 1, 2))
        out.append(obs)
    return np.array(out)


@pytest.mark.parametrize("case, width", [("two_glider", 27), ("multi_glider", 57)])
def test_observation_shape_and_bounds(case, width):
    env = GliderTrackingEnv(CASES[case])
    obs = rollout(env, seed=0, steps=300)
    assert obs.shape[1] == width == env.observation_space.shape[0]
    assert np.all(np.abs(obs) <= 1.0)


def test_seeded_reset_is_deterministic():
    a = rollout(GliderTrackingEnv(CASES["multi_glider"]), seed=5, steps=200)
    b = rollout(GliderTrackingEnv(CASES["multi_glider"]), seed=5, steps=200)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n, slots", [(2, [0, 4]), (3, [0, 2, 4]), (4, [0, 1, 3, 4]), (5, [0, 1, 2, 3, 4])])
def test_fleet_slots_keep_line_ends_in_end_slots(n, slots):
    env = GliderTrackingEnv(eval_config("multi_glider", n_gliders=n))
    obs, _ = env.reset(seed=1)
    mask = obs[GLOBAL_FEATURES + 5 * GLIDER_FEATURES:]
    assert env.n_gliders == n
    assert list(np.flatnonzero(mask)) == slots


def test_fleet_size_is_sampled_over_full_range():
    env = GliderTrackingEnv(CASES["multi_glider"])
    env.reset(seed=0)
    sizes = set()
    for _ in range(60):
        env.reset()
        sizes.add(env.n_gliders)
    assert sizes == {2, 3, 4, 5}


def test_gliders_span_one_length_unit():
    for n in (2, 3, 5):
        env = GliderTrackingEnv(eval_config("multi_glider", n_gliders=n))
        env.reset(seed=3)
        assert np.linalg.norm(env.g_pos[-1] - env.g_pos[0]) == pytest.approx(1.0, abs=1e-5)


def test_softmin_sits_between_min_and_mean():
    env = GliderTrackingEnv(CASES["multi_glider"])
    r = np.array([2.0, 5.0, 9.0])
    assert r.min() <= env._agg_unc(r) <= r.mean()
    assert env._agg_unc(np.full(4, 3.0)) == pytest.approx(3.0)


def test_uncertainty_reward_is_floored_at_zero_when_never_heard():
    env = GliderTrackingEnv(eval_config("two_glider", rho=1e-3, max_steps=2000))
    env.reset(seed=0)
    for _ in range(2000):
        *_, info = env.step(np.array([-1.0, 0.0]))
    assert np.all(info["r_unc_each"] >= -1e-6)


def test_range_update_shrinks_uncertainty():
    env = GliderTrackingEnv(eval_config("two_glider", r_noise_std=1e-3))
    env.reset(seed=0)
    before = env.g_unc[0]
    env.asv_pos = env.g_pos[0].copy()  # sit right above glider 1
    env.step(np.array([-1.0, 0.0]))
    assert env.g_unc[0] < before
