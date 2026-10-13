"""SAC training recipe shared by both reference cases."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .configs import CASES, eval_config
from .env import GliderTrackingEnv
from .policy import Policy
from .rollout import run_episode


@dataclass
class TrainConfig:
    case: str = "two_glider"
    total_steps: int = 30_000_000
    seed: int = 0
    normalize_obs: bool = True
    # episodes grow from horizon_start to horizon_end over the first anneal_frac of training
    horizon_start: int = 500
    horizon_end: int = 2880
    anneal_frac: float = 0.95
    eval_every: int = 25_000
    eval_episodes: int = 15
    eval_fleet_sizes: list[int] = field(default_factory=lambda: [2])
    eval_rho: float = 0.4
    eval_max_steps: int = 2880
    learning_rate: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 3_000_000
    batch_size: int = 256
    tau: float = 0.005
    learning_starts: int = 10_000
    train_freq: int = 4
    net_arch: list[int] = field(default_factory=lambda: [256, 256])


# the recipes behind the two shipped policies
RECIPES = {
    "two_glider": TrainConfig(case="two_glider"),
    "multi_glider": TrainConfig(case="multi_glider", normalize_obs=False, eval_every=500_000,
                                eval_episodes=6, eval_fleet_sizes=[2, 3, 4, 5]),
}


class HorizonCurriculum(BaseCallback):
    """Linearly lengthen training episodes."""

    def __init__(self, start: int, end: int, anneal_steps: int):
        super().__init__()
        self.start, self.end, self.anneal_steps = start, end, max(anneal_steps, 1)

    def _on_step(self) -> bool:
        frac = min(self.num_timesteps / self.anneal_steps, 1.0)
        horizon = int(self.start + frac * (self.end - self.start))
        for env in _base_envs(self.training_env):
            env.max_steps = horizon
        return True


def _base_envs(vec_env) -> list[GliderTrackingEnv]:
    # unwrap VecNormalize / DummyVecEnv / Monitor down to the envs themselves
    while hasattr(vec_env, "venv"):
        vec_env = vec_env.venv
    return [e.unwrapped for e in vec_env.envs]


class FleetEval(BaseCallback):
    """Deterministic evaluation per fleet size; keeps the checkpoint with the best mean return."""

    def __init__(self, cfg: TrainConfig, out_dir: Path):
        super().__init__()
        self.cfg, self.out_dir = cfg, out_dir
        self.best = -np.inf
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.cfg.eval_every:
            return True
        policy = policy_from_model(self.model)
        per_n = {}
        for n in self.cfg.eval_fleet_sizes:
            env = GliderTrackingEnv(eval_config(self.cfg.case, rho=self.cfg.eval_rho, n_gliders=n,
                                                max_steps=self.cfg.eval_max_steps))
            returns = [run_episode(env, policy, seed=50_000 + k).rewards.sum() for k in range(self.cfg.eval_episodes)]
            per_n[n] = float(np.mean(returns))
        score = float(np.mean(list(per_n.values())))
        self.history.append({"step": self.num_timesteps, "score": score, **{f"N{n}": v for n, v in per_n.items()}})
        (self.out_dir / "eval_history.json").write_text(json.dumps(self.history, indent=1))
        self.logger.record("eval/mean_return", score)
        if score > self.best:
            self.best = score
            save_policy(policy, self.out_dir / "best")
        return True


def policy_from_model(model: SAC) -> Policy:
    """Freeze the current observation statistics into a standalone Policy."""
    venv = model.get_vec_normalize_env()
    if venv is not None and venv.norm_obs:
        norm = dict(enabled=True, mean=venv.obs_rms.mean.copy(), var=venv.obs_rms.var.copy(),
                    clip=venv.clip_obs, epsilon=venv.epsilon)
    else:
        dim = model.observation_space.shape[0]
        norm = dict(enabled=False, mean=np.zeros(dim), var=np.ones(dim), clip=10.0, epsilon=1e-8)
    return Policy(model, norm)


def save_policy(policy: Policy, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    policy.model.save(out / "policy.zip")
    np.savez(out / "obs_norm.npz", enabled=np.array(policy.normalize), mean=policy.mean, var=policy.var,
             clip=np.array(policy.clip), epsilon=np.array(policy.epsilon))


def train(cfg: TrainConfig, out_dir: Path | str) -> SAC:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=1))

    env_cfg = dict(CASES[cfg.case], max_steps=cfg.horizon_start)
    venv = DummyVecEnv([lambda: Monitor(GliderTrackingEnv(env_cfg), str(out_dir / "monitor.csv"))])
    venv = VecNormalize(venv, norm_obs=cfg.normalize_obs, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
                        gamma=cfg.gamma)

    model = SAC("MlpPolicy", venv, learning_rate=cfg.learning_rate, gamma=cfg.gamma, buffer_size=cfg.buffer_size,
                learning_starts=cfg.learning_starts, batch_size=cfg.batch_size, tau=cfg.tau, ent_coef="auto",
                train_freq=cfg.train_freq, gradient_steps=1, policy_kwargs=dict(net_arch=cfg.net_arch),
                seed=cfg.seed, verbose=0)
    callbacks = CallbackList([
        HorizonCurriculum(cfg.horizon_start, cfg.horizon_end, int(cfg.anneal_frac * cfg.total_steps)),
        FleetEval(cfg, out_dir),
    ])
    model.learn(total_timesteps=cfg.total_steps, callback=callbacks, progress_bar=False)
    save_policy(policy_from_model(model), out_dir / "final")
    return model
