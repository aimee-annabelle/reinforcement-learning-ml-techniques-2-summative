"""Standalone PPO training script for the RwandaHealthEnv environment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import make_env
from training.hyperparam_configs import PPO_SEARCH_SPACE
from training.train_models import build_policy_kwargs, parse_args, to_serializable

DEFAULT_TIMESTEPS = 250_000


def train_ppo(args: argparse.Namespace, configs: Sequence[Dict[str, Any]]) -> None:
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() / "ppo"
    log_dir = (PROJECT_ROOT / args.log_dir).resolve() / "ppo"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    max_runs = args.max_combos or len(configs)
    selected_configs = list(configs)[:max_runs]
    results = []

    for index, base_config in enumerate(selected_configs):
        run_dir = output_dir / f"run_{index:02d}"
        best_model_dir = run_dir / "best_model"
        eval_log_dir = run_dir / "eval_logs"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        eval_log_dir.mkdir(parents=True, exist_ok=True)

        config = dict(base_config)
        policy_kwargs = build_policy_kwargs(config)
        total_timesteps = args.total_timesteps or int(config.pop("total_timesteps", DEFAULT_TIMESTEPS))
        eval_freq = args.eval_frequency or max(2_000, total_timesteps // 10)
        run_seed = args.seed + index

        train_env = DummyVecEnv([make_env(seed=run_seed, monitor=True)])
        eval_env = make_env(seed=run_seed + 1_000, monitor=True)()

        callback_on_new_best = None
        if not args.no_stop:
            callback_on_new_best = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=6,
                min_evals=6,
                verbose=1,
            )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(eval_log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=args.num_eval_episodes,
            deterministic=True,
            render=False,
            callback_on_new_best=callback_on_new_best,
        )

        start_time = time.perf_counter()
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(log_dir),
            device=args.device,
            seed=run_seed,
            **config,
        )

        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=args.progress_bar)
        wall_time = time.perf_counter() - start_time

        model.save(str(run_dir / "final_model"))
        train_env.close()
        eval_env.close()

        best_model_path = best_model_dir / "best_model.zip"
        if best_model_path.exists():
            eval_model = PPO.load(best_model_path, device=args.device)
        else:
            eval_model = model

        eval_env_final = make_env(seed=run_seed + 2_000, monitor=True)()
        mean_reward, std_reward = evaluate_policy(
            eval_model,
            eval_env_final,
            n_eval_episodes=args.num_eval_episodes,
            deterministic=True,
        )
        eval_env_final.close()

        logged_config = dict(config)
        logged_config["policy_kwargs"] = {
            "net_arch": policy_kwargs["net_arch"],
            "activation_fn": policy_kwargs["activation_fn"].__name__,
            "ortho_init": policy_kwargs["ortho_init"],
        }
        logged_config["total_timesteps"] = total_timesteps

        metadata = {
            "algo": "ppo",
            "run_index": index,
            "seed": run_seed,
            "hyperparameters": to_serializable(logged_config),
            "training_walltime_sec": wall_time,
            "eval_mean_reward": float(mean_reward),
            "eval_std_reward": float(std_reward),
            "callback_best_mean_reward": (
                float(eval_callback.best_mean_reward) if eval_callback.best_mean_reward is not None else None
            ),
            "best_model_path": to_serializable(best_model_path if best_model_path.exists() else run_dir / "final_model.zip"),
            "final_model_path": to_serializable(run_dir / "final_model.zip"),
            "tensorboard_log_dir": to_serializable(log_dir),
        }

        with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        results.append(metadata)

    with (output_dir / "ppo_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("Saved PPO results to", output_dir)


def main() -> None:
    args = parse_args()
    train_ppo(args, PPO_SEARCH_SPACE)


if __name__ == "__main__":
    main()
