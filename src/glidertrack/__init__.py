"""Reinforcement-learning guidance for an ASV tracking underwater gliders with range-only acoustics."""
from .configs import CASES, MULTI_GLIDER, TWO_GLIDER, eval_config
from .env import GliderTrackingEnv
from .policy import Policy
from .rollout import Episode, run_episode

__all__ = ["CASES", "MULTI_GLIDER", "TWO_GLIDER", "Episode", "GliderTrackingEnv", "Policy", "eval_config",
           "run_episode"]
__version__ = "0.1.0"
