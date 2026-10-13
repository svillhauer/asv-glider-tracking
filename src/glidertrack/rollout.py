"""Run one episode and record what happened."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .env import GliderTrackingEnv


@dataclass
class Episode:
    """Per-step arrays from one episode, in nondimensional units."""

    rewards: np.ndarray  # (T,)
    asv: np.ndarray  # (T, 2)
    heading: np.ndarray  # (T,)
    gliders: np.ndarray  # (T, N, 2) true positions
    estimates: np.ndarray  # (T, N, 2) EKF positions
    covariances: np.ndarray  # (T, N, 2, 2) EKF position covariance
    depth: np.ndarray  # (T,)
    in_range: np.ndarray  # (T, N) bool
    comm_range: float
    max_depth: float

    @property
    def n_gliders(self) -> int:
        return self.gliders.shape[1]

    @property
    def true_error(self) -> np.ndarray:
        return np.linalg.norm(self.gliders - self.estimates, axis=-1)

    @property
    def ekf_std(self) -> np.ndarray:
        # sqrt(Pxx + Pyy) per glider
        return np.sqrt(np.trace(self.covariances, axis1=-2, axis2=-1))


def run_episode(env: GliderTrackingEnv, policy: Callable[[np.ndarray], np.ndarray], seed: int) -> Episode:
    obs, _ = env.reset(seed=seed)
    rec = {k: [] for k in ("rewards", "asv", "heading", "gliders", "estimates", "covariances", "depth", "in_range")}
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        done = terminated or truncated
        rec["rewards"].append(reward)
        rec["asv"].append(env.asv_pos.copy())
        rec["heading"].append(env.asv_heading)
        rec["gliders"].append(env.g_pos.copy())
        rec["estimates"].append(env.g_est_pos.copy())
        rec["covariances"].append(env.g_P[:, :2, :2].copy())
        rec["depth"].append(env._depth)
        rec["in_range"].append(info["dist_each"] <= info["comm_range"])
    return Episode(**{k: np.asarray(v) for k, v in rec.items()},
                   comm_range=env.comm_range, max_depth=env.glider_dive_depth)
