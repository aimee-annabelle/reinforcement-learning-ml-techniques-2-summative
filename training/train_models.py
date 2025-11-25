"""Training entry-point for RwandaHealthEnv reinforcement learning baselines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Type

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import make_env
from training.hyperparam_configs import (
    A2C_SEARCH_SPACE,
    DQN_SEARCH_SPACE,
    PPO_SEARCH_SPACE,
    REINFORCE_SEARCH_SPACE,
)
from training.reinforce_agent import train_reinforce


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}

DEFAULT_TIMESTEPS = {
    "dqn": 150_000,
    "ppo": 250_000,
    "a2c": 180_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agents on RwandaHealthEnv")
    parser.add_argument("--algo", choices=["all", "dqn", "ppo", "a2c", "reinforce"], default="all")
    parser.add_argument("--output-dir", type=str, default="models", help="Where to store trained models")
    parser.add_argument("--log-dir", type=str, default="training_logs", help="TensorBoard and eval logs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device for stable-baselines algorithms")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override SB3 total timesteps")
    parser.add_argument("--max-combos", type=int, default=None, help="Limit number of hyperparameter configs per algorithm")
    parser.add_argument("--reinforce-episodes", type=int, default=None, help="Override episode count for REINFORCE runs")
    parser.add_argument("--num-eval-episodes", type=int, default=8, help="Evaluation episodes after each run")
    parser.add_argument("--eval-frequency", type=int, default=None, help="Override evaluation frequency for SB3 training")
    parser.add_argument("--no-stop", action="store_true", help="Disable early stopping on plateau during SB3 training")
    parser.add_argument("--progress-bar", action="store_true", help="Show stable-baselines3 progress bar during learning")
    return parser.parse_args()


def activation_from_name(name: str) -> Type[nn.Module]:
    activation_cls = ACTIVATION_MAP.get(name.lower())
    if activation_cls is None:
        raise ValueError(f"Unknown activation '{name}' in policy configuration")
    return activation_cls


def build_policy_kwargs(config: Dict[str, Any], *, supports_ortho: bool = True) -> Dict[str, Any]:
    net_arch = config.pop("policy_net_arch", [256, 256])
    activation_name = config.pop("policy_activation", "relu")
    ortho_init = bool(config.pop("ortho_init", True))
    activation_cls = activation_from_name(activation_name)
    policy_kwargs: Dict[str, Any] = {
        "net_arch": net_arch,
        "activation_fn": activation_cls,
    }
    if supports_ortho:
        policy_kwargs["ortho_init"] = ortho_init
    return policy_kwargs


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value)
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if callable(value):
        return value.__name__
    return value


def train_sb3_algorithm(
    algo_name: str,
    algo_class: Type[BaseAlgorithm],
    search_space: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    default_timesteps = DEFAULT_TIMESTEPS[algo_name]
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() / algo_name
    log_dir = (PROJECT_ROOT / args.log_dir).resolve() / algo_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    max_runs = args.max_combos or len(search_space)
    configs = list(search_space)[:max_runs]
    results: List[Dict[str, Any]] = []

    for index, base_config in enumerate(configs):
        run_dir = output_dir / f"run_{index:02d}"
        best_model_dir = run_dir / "best_model"
        eval_log_dir = run_dir / "eval_logs"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        eval_log_dir.mkdir(parents=True, exist_ok=True)

        config = dict(base_config)
        policy_kwargs = build_policy_kwargs(config, supports_ortho=algo_name != "dqn")
        total_timesteps = args.total_timesteps or int(config.pop("total_timesteps", default_timesteps))
        eval_freq = args.eval_frequency or max(1_000, total_timesteps // 10)
        run_seed = args.seed + index

        train_env = DummyVecEnv([make_env(seed=run_seed, monitor=True)])
        eval_env = make_env(seed=run_seed + 1_000, monitor=True)()

        callback_on_new_best = None
        if not args.no_stop:
            callback_on_new_best = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=5,
                min_evals=5,
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
        model = algo_class(
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
            eval_model = algo_class.load(best_model_path, device=args.device)
        else:
            eval_model = model

        eval_env_final = make_env(seed=run_seed + 2_000, monitor=True)()
        mean_reward, std_reward = sb3_evaluate_policy(
            eval_model,
            eval_env_final,
            n_eval_episodes=args.num_eval_episodes,
            deterministic=True,
        )
        eval_env_final.close()

        logged_config = dict(config)
        logged_policy_kwargs: Dict[str, Any] = {
            "net_arch": policy_kwargs["net_arch"],
            "activation_fn": policy_kwargs["activation_fn"].__name__,
        }
        if "ortho_init" in policy_kwargs:
            logged_policy_kwargs["ortho_init"] = policy_kwargs["ortho_init"]
        else:
            logged_policy_kwargs["ortho_init"] = "sb3-default"
        logged_config["policy_kwargs"] = logged_policy_kwargs
        logged_config["total_timesteps"] = total_timesteps

        metadata = {
            "algo": algo_name,
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

    with (output_dir / f"{algo_name}_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return results


def train_reinforce_sweep(configs: Sequence[Dict[str, Any]], args: argparse.Namespace, torch_device: torch.device) -> List[Dict[str, Any]]:
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() / "reinforce"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_runs = args.max_combos or len(configs)
    selected_configs = list(configs)[:max_runs]
    results: List[Dict[str, Any]] = []

    for index, base_config in enumerate(selected_configs):
        run_dir = output_dir / f"run_{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_seed = args.seed + index

        config = dict(base_config)
        if args.reinforce_episodes is not None:
            config["episodes"] = args.reinforce_episodes

        start_time = time.perf_counter()
        run_result = train_reinforce(config, run_dir, run_seed, args.num_eval_episodes, torch_device)
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

    with (output_dir / "reinforce_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return results


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    log_dir = (PROJECT_ROOT / args.log_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    summary: Dict[str, List[Dict[str, Any]]] = {}

    if args.algo in ("all", "dqn"):
        summary["dqn"] = train_sb3_algorithm("dqn", DQN, DQN_SEARCH_SPACE, args)
    if args.algo in ("all", "ppo"):
        summary["ppo"] = train_sb3_algorithm("ppo", PPO, PPO_SEARCH_SPACE, args)
    if args.algo in ("all", "a2c"):
        summary["a2c"] = train_sb3_algorithm("a2c", A2C, A2C_SEARCH_SPACE, args)
    if args.algo in ("all", "reinforce"):
        summary["reinforce"] = train_reinforce_sweep(REINFORCE_SEARCH_SPACE, args, torch_device)

    summary_path = output_dir / "experiment_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({k: to_serializable(v) for k, v in summary.items()}, handle, indent=2)

    print("Saved experiment summary to", summary_path)


if __name__ == "__main__":
    main()
