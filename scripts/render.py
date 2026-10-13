"""Render a mission of a shipped policy as a GIF.

    python scripts/render.py --case two_glider --seed 77
    python scripts/render.py --case multi_glider --gliders 4 --seed 901
"""
from __future__ import annotations

import argparse
from pathlib import Path

from glidertrack.animate import render_gif
from glidertrack.configs import eval_config
from glidertrack.env import GliderTrackingEnv
from glidertrack.policy import Policy
from glidertrack.rollout import run_episode

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=["two_glider", "multi_glider"], required=True)
    ap.add_argument("--gliders", type=int, default=None, help="fleet size (multi_glider only)")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--rho", type=float, default=0.4)
    ap.add_argument("--every", type=int, default=24, help="env steps per frame (24 = 2 h)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    n = 2 if args.case == "two_glider" else (args.gliders or 4)
    ep = run_episode(GliderTrackingEnv(eval_config(args.case, rho=args.rho, n_gliders=n)), Policy.load(args.case),
                     args.seed)
    out = args.out or ROOT / "docs" / "media" / f"{args.case}.gif"
    title = "Two gliders, 20 km apart" if n == 2 else f"{n} gliders over a 20 km span"
    render_gif(ep, out, every=args.every, title=title)
    print(f"wrote {out}  (return {ep.rewards.sum():,.0f})")


if __name__ == "__main__":
    main()
