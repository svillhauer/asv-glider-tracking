from dataclasses import replace

from glidertrack.policy import Policy
from glidertrack.training import RECIPES, train


def test_training_smoke(tmp_path):
    cfg = replace(RECIPES["multi_glider"], total_steps=400, learning_starts=100, buffer_size=1_000,
                  horizon_start=40, horizon_end=80, eval_every=200, eval_episodes=1, eval_fleet_sizes=[2, 5],
                  eval_max_steps=40)
    train(cfg, tmp_path)
    for part in ("best", "final"):
        assert (tmp_path / part / "policy.zip").exists()
        assert (tmp_path / part / "obs_norm.npz").exists()
    assert (tmp_path / "eval_history.json").exists()
    Policy.load("best", models_dir=tmp_path)  # same layout as models/
