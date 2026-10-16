"""Plain-numpy SAC actor and VecNormalize, so the container needs no SB3 at run time."""

import numpy as np

_ACT_FNS = {
    "ReLU": lambda x: np.maximum(0.0, x),
    "Tanh": np.tanh,
    "ELU": lambda x: np.where(x > 0, x, np.exp(x) - 1.0),
}


class SACActorNumpy:
    """Deterministic SAC actor: MLP trunk -> mu head -> tanh (what SB3 predict(deterministic=True) does)."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        acts = list(d["act_names"])
        self._layers = [(d[f"mlp_w{i}"].astype(np.float32), d[f"mlp_b{i}"].astype(np.float32),
                         _ACT_FNS[acts[i]]) for i in range(int(d["n_layers"]))]
        self._mu_w = d["mu_w"].astype(np.float32)
        self._mu_b = d["mu_b"].astype(np.float32)
        self.obs_dim = self._layers[0][0].shape[1]

    def predict(self, obs_norm: np.ndarray) -> np.ndarray:
        x = np.asarray(obs_norm, dtype=np.float32)
        for w, b, act in self._layers:
            x = act(x @ w.T + b)
        return np.tanh(x @ self._mu_w.T + self._mu_b)


class VecNormalizeNumpy:
    """Observation half of SB3 VecNormalize (training=False)."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.mean = d["obs_mean"].astype(np.float64)
        self.var = d["obs_var"].astype(np.float64)
        self.clip = float(d["clip_obs"])
        self.eps = float(d["epsilon"])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + self.eps),
                       -self.clip, self.clip).astype(np.float32)
