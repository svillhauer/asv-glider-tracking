"""Evaluate a shipped policy on fixed seeds and write results/<case>.json.

    python scripts/evaluate.py --case two_glider
    python scripts/evaluate.py --case multi_glider --episodes 15
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from glidertrack.configs import eval_config
from glidertrack.env import GliderTrackingEnv
from glidertrack.metrics import episode_metrics, summarize
from glidertrack.policy import Policy
from glidertrack.rollout import run_episode

ROOT = Path(__file__).resolve().parents[1]
FLEETS = {"two_glider": [2], "multi_glider": [2, 3, 4, 5]}


def _one(args):
    case, n, rho, seed = args
    env = GliderTrackingEnv(eval_config(case, rho=rho, n_gliders=n))
    return n, episode_metrics(run_episode(env, Policy.load(case), seed))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=sorted(FLEETS), required=True)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--first-seed", type=int, default=900)
    ap.add_argument("--rho", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = [(args.case, n, args.rho, args.first_seed + k) for n in FLEETS[args.case] for k in range(args.episodes)]
    with ProcessPoolExecutor(args.workers) as pool:
        done = list(pool.map(_one, jobs))

    results = {"case": args.case, "rho": args.rho, "seeds": [args.first_seed, args.first_seed + args.episodes - 1],
               "fleets": {}}
    for n in FLEETS[args.case]:
        results["fleets"][str(n)] = summarize([m for k, m in done if k == n])

    out = ROOT / "results" / f"{args.case}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"{'N':>3} {'reward/step':>12} {'in range':>9} {'true err':>10} {'EKF std':>9} {'lost':>6}")
    for n, r in results["fleets"].items():
        print(f"{n:>3} {r['reward_per_step']:>12.2f} {100 * r['time_in_range']:>8.1f}% {r['true_error_m']:>8.0f} m"
              f" {r['ekf_std_m']:>7.0f} m {r['lost']:>3}/{r['episodes']}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
