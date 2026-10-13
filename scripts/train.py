"""Train a policy with the recipe used for the shipped models.

    python scripts/train.py --case two_glider --out runs/two_glider
    python scripts/train.py --case multi_glider --out runs/multi_glider --steps 30000000

The best checkpoint lands in <out>/best (policy.zip + obs_norm.npz), the same layout as models/.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from glidertrack.training import RECIPES, train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=sorted(RECIPES), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None, help="total env steps (default 30M)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    args = ap.parse_args()

    cfg = replace(RECIPES[args.case], seed=args.seed)
    if args.steps:
        cfg = replace(cfg, total_steps=args.steps)
    if args.eval_every:
        cfg = replace(cfg, eval_every=args.eval_every)
    if args.eval_episodes:
        cfg = replace(cfg, eval_episodes=args.eval_episodes)
    train(cfg, args.out)


if __name__ == "__main__":
    main()
