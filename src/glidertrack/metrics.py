"""Episode and evaluation metrics."""
from __future__ import annotations

import numpy as np

from .configs import LENGTH_M
from .rollout import Episode

# a glider counts as lost when its estimate is ever off by more than 20% of the span
LOST_THRESHOLD = 0.20


def episode_metrics(ep: Episode) -> dict:
    err = ep.true_error
    return {
        "n_gliders": ep.n_gliders,
        "reward_per_step": float(ep.rewards.mean()),
        "episode_return": float(ep.rewards.sum()),
        "time_in_range": float(ep.in_range.mean()),
        "least_served_in_range": float(ep.in_range.mean(axis=0).min()),
        "true_error_pct": float(100 * err.mean()),
        "true_error_m": float(LENGTH_M * err.mean()),
        "ekf_std_m": float(LENGTH_M * ep.ekf_std.mean()),
        "lost": bool((err > LOST_THRESHOLD).any()),
    }


def summarize(rows: list[dict]) -> dict:
    """Median over episodes, plus the number of episodes that lost a glider."""
    keys = [k for k in rows[0] if k not in ("n_gliders", "lost")]
    out = {k: float(np.median([r[k] for r in rows])) for k in keys}
    out["lost"] = int(sum(r["lost"] for r in rows))
    out["episodes"] = len(rows)
    return out
