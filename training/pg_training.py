"""Standalone policy-gradient (REINFORCE) training script for RwandaHealthEnv."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.hyperparam_configs import REINFORCE_SEARCH_SPACE
from training.reinforce_agent import train_reinforce
from training.train_models import parse_args, to_serializable

DEFAULT_EPISODES = 600


def train_pg(args: argparse.Namespace, configs: Sequence[Dict[str, object]], device: torch.device) -> None:
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() / "pg"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_runs = args.max_combos or len(configs)
    selected_configs = list(configs)[:max_runs]
    results = []

    for index, base_config in enumerate(selected_configs):
        run_dir = output_dir / f"run_{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_seed = args.seed + index

        config = dict(base_config)
        if args.reinforce_episodes is not None:
            config["episodes"] = args.reinforce_episodes
        else:
            config.setdefault("episodes", DEFAULT_EPISODES)

        start_time = time.perf_counter()
        run_result = train_reinforce(config, run_dir, run_seed, args.num_eval_episodes, device)
        wall_time = time.perf_counter() - start_time

        metadata = {
            "algo": "reinforce",
            "run_index": index,
            "seed": run_seed,
            "hyperparameters": to_serializable(config),
            "training_walltime_sec": wall_time,
            "eval_mean_reward": float(run_result.mean_eval_reward),
            "eval_std_reward": float(run_result.std_eval_reward),
            "best_smoothed_reward": float(run_result.best_smoothed_reward),
            "episode_rewards_path": to_serializable(run_result.episode_rewards_path),
            "policy_path": to_serializable(run_result.policy_path),
        }

        with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        results.append(metadata)

    with (output_dir / "pg_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("Saved policy-gradient results to", output_dir)


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_pg(args, REINFORCE_SEARCH_SPACE, device)


if __name__ == "__main__":
    main()
