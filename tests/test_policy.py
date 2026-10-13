import numpy as np
import pytest

from glidertrack.configs import eval_config
from glidertrack.env import GliderTrackingEnv
from glidertrack.policy import Policy
from glidertrack.rollout import run_episode


@pytest.mark.parametrize("case", ["two_glider", "multi_glider"])
def test_shipped_policy_runs(case, request):
    cfg = request.getfixturevalue("short_two" if case == "two_glider" else "short_multi")
    ep = run_episode(GliderTrackingEnv(cfg), Policy.load(case), seed=900)
    assert len(ep.rewards) == 200
    assert np.isfinite(ep.rewards).all()


def test_policy_actions_are_bounded(short_multi):
    env, policy = GliderTrackingEnv(short_multi), Policy.load("multi_glider")
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        a = policy(obs)
        assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)
        obs, *_ = env.step(a)


def test_reference_mission_is_reproduced():
    # the representative two-glider mission reported in the thesis
    ep = run_episode(GliderTrackingEnv(eval_config("two_glider")), Policy.load("two_glider"), seed=77)
    assert ep.rewards.sum() == pytest.approx(53_410, abs=1)
