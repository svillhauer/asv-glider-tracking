"""
Run on the host from the repository root (needs stable-baselines3):
    python3 gazebo/tools/extract_policy.py

Exports the stable reference policy (nglider_knowndepth_flat_fixed2_stable_s0, best_model)
as plain numpy files the container can load without SB3:
  gazebo/policy/actor_weights.npz              SAC actor MLP + mu head
  gazebo/policy/best_vec_normalize_arrays.npz  VecNormalize obs mean / var / clip / epsilon

Same format as the old pipeline's extract_actor.py / extract_vecnorm.py.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

REF = Path(__file__).resolve().parents[1] / "policy"
MODEL = Path(__file__).resolve().parents[2] / "models" / "two_glider" / "policy.zip"
PKL = REF / "best_vec_normalize.pkl"

# Actor
model = SAC.load(str(MODEL), device="cpu")
actor = model.policy.actor
weights, biases, acts = [], [], []
for m in actor.latent_pi:
    if isinstance(m, torch.nn.Linear):
        weights.append(m.weight.detach().float().numpy())
        biases.append(m.bias.detach().float().numpy())
    else:
        acts.append(type(m).__name__)
arrays = {"n_layers": np.array(len(weights)), "act_names": np.array(acts),
          "mu_w": actor.mu.weight.detach().float().numpy(),
          "mu_b": actor.mu.bias.detach().float().numpy()}
for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
    arrays[f"mlp_w{i}"], arrays[f"mlp_b{i}"] = w, b
np.savez(REF / "actor_weights.npz", **arrays)
print(f"actor: obs_dim={weights[0].shape[1]}  layers={[w.shape[0] for w in weights]}  acts={acts}")

# VecNormalize
with open(PKL, "rb") as f:
    vn = pickle.load(f)
assert vn.norm_obs, "expected an observation-normalizing VecNormalize"
np.savez(REF / "best_vec_normalize_arrays.npz",
         obs_mean=vn.obs_rms.mean.astype(np.float32),
         obs_var=vn.obs_rms.var.astype(np.float32),
         clip_obs=np.float32(vn.clip_obs),
         epsilon=np.float32(vn.epsilon))
print(f"vecnorm: obs_dim={vn.obs_rms.mean.shape[0]}  clip={vn.clip_obs}  eps={vn.epsilon}")
