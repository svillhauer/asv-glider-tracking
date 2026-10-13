"""Load a trained SAC policy together with its observation normalization."""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class Policy:
    """Deterministic SAC actor with the observation normalization it was trained with."""

    def __init__(self, model, obs_norm: dict):
        self.model = model
        self.normalize = bool(obs_norm["enabled"])
        self.mean = np.asarray(obs_norm["mean"], dtype=np.float64)
        self.var = np.asarray(obs_norm["var"], dtype=np.float64)
        self.clip = float(obs_norm["clip"])
        self.epsilon = float(obs_norm["epsilon"])

    @classmethod
    def load(cls, case: str, models_dir: Path | str = MODELS_DIR, device: str = "cpu") -> Policy:
        from gymnasium import spaces
        from stable_baselines3 import SAC

        d = Path(models_dir) / case
        with np.load(d / "obs_norm.npz") as f:
            obs_norm = {k: f[k] for k in f.files}
        # rebuild the spaces instead of unpickling them, so models load across numpy/gymnasium versions
        dim = len(obs_norm["mean"])
        custom = {
            "observation_space": spaces.Box(-1.0, 1.0, (dim,), dtype=np.float32),
            "action_space": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            "lr_schedule": lambda _: 0.0,
            "learning_rate": 0.0,
        }
        model = SAC.load(d / "policy.zip", device=device, custom_objects=custom)
        return cls(model, obs_norm)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(obs, dtype=np.float32)
        z = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(self.normalize_obs(obs), deterministic=True)
        return action
