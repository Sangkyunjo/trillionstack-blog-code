from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from algorithms import ALGORITHMS
from cliff_walking import evaluate_greedy, policy_grid


def summarize_training(result: dict[str, object], window: int = 200) -> dict[str, float]:
    returns = np.asarray(result["returns"], dtype=float)
    steps = np.asarray(result["steps"], dtype=float)
    falls = np.asarray(result["cliff_falls"], dtype=float)
    truncated = np.asarray(result["truncated"], dtype=float)
    tail = slice(max(0, returns.size - window), returns.size)
    return {
        "last_window_mean_return": float(np.mean(returns[tail])),
        "last_window_mean_steps": float(np.mean(steps[tail])),
        "last_window_mean_cliff_falls": float(np.mean(falls[tail])),
        "truncation_rate": float(np.mean(truncated)),
        "environment_steps": int(np.sum(steps)),
    }


def run(episodes: int, seeds: list[int], eval_episodes: int) -> dict[str, object]:
    output: dict[str, object] = {
        "config": {
            "episodes_per_seed": episodes,
            "seeds": seeds,
            "evaluation_episodes": eval_episodes,
            "max_steps_per_episode": 500,
        },
        "algorithms": {},
    }

    for name, algorithm in ALGORITHMS.items():
        runs = []
        for seed in seeds:
            result = algorithm(episodes=episodes, seed=seed)
            table = np.asarray(result["table"])
            runs.append(
                {
                    "seed": seed,
                    "training": summarize_training(result),
                    "evaluation": evaluate_greedy(
                        table,
                        episodes=eval_episodes,
                        seed=10_000 + seed,
                    ),
                    "policy_grid": policy_grid(table),
                }
            )

        aggregate = {}
        for group in ("training", "evaluation"):
            keys = runs[0][group].keys()
            aggregate[group] = {
                key: {
                    "mean": float(np.mean([item[group][key] for item in runs])),
                    "std": float(np.std([item[group][key] for item in runs], ddof=1))
                    if len(runs) > 1
                    else 0.0,
                }
                for key in keys
            }
        output["algorithms"][name] = {"aggregate": aggregate, "runs": runs}

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.episodes, list(range(args.seeds)), args.eval_episodes)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
