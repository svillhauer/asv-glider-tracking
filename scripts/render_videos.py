"""Render the mission videos linked from the README and the paper.

    python scripts/render_videos.py            # all of them, into docs/videos/
    python scripts/render_videos.py reference  # just one
"""
from __future__ import annotations

import sys
from pathlib import Path

from glidertrack.configs import eval_config
from glidertrack.env import GliderTrackingEnv
from glidertrack.policy import Policy
from glidertrack.rollout import run_episode
from glidertrack.video import Mission, render_mission

OUT = Path(__file__).resolve().parents[1] / "docs" / "videos"

VIDEOS = {
    "reference": dict(case="two_glider", n=2, rho=0.4, seed=77, file="reference_seed77.mp4",
                      title="Two gliders, 20 km apart",
                      subtitle="Learned SAC policy, coverage ratio 0.4 (8 km acoustic range), mission 77"),
    "loiter": dict(case="two_glider", n=2, rho=0.6, seed=77, file="coverage_rho060_seed77.mp4",
                   title="Two gliders, overlapping acoustic footprints",
                   subtitle="Same policy at coverage ratio 0.6 (12 km acoustic range): it loiters between the gliders"),
    "five_gliders": dict(case="multi_glider", n=5, rho=0.4, seed=900, file="multiglider_n5_seed900.mp4",
                         title="Five gliders over a 20 km span",
                         subtitle="Multi-glider policy, 8 km acoustic range, mission 900"),
}


def render(name):
    v = VIDEOS[name]
    env = GliderTrackingEnv(eval_config(v["case"], rho=v["rho"], n_gliders=v["n"]))
    ep = run_episode(env, Policy.load(v["case"]), v["seed"])
    path = render_mission(Mission.from_episode(ep, env), OUT / v["file"], v["title"], v["subtitle"])
    print(f"wrote {path}")


if __name__ == "__main__":
    for name in sys.argv[1:] or VIDEOS:
        render(name)
