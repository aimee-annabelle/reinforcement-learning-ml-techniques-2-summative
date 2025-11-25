from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from environment.custom_env import ACTION_MEANINGS, RwandaHealthEnv

try:
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.base_class import BaseAlgorithm
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Stable Baselines3 is required to run the simulation entrypoint. "
        "Install the dependencies listed in requirements.txt before continuing."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "training" / "reports"
SUMMARY_PATH = REPORTS_DIR / "summary_metrics.csv"

SB3_CLASSES: Dict[str, type[BaseAlgorithm]] = {
    "A2C": A2C,
    "DQN": DQN,
    "PPO": PPO,
}


@dataclass
class RunSelection:
    algo: str
    checkpoint: Path
    metadata: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate the best-performing Rwanda Health RL agent with GUI and verbose terminal logs.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to simulate.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force greedy action selection during playback.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "ansi"],
        default="human",
        help="Rendering mode to request from the environment (human opens the pygame GUI).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override the automatically discovered checkpoint.",
    )
    parser.add_argument(
        "--algo",
        choices=["A2C", "DQN", "PPO", "REINFORCE"],
        default=None,
        help="Algorithm name when supplying a custom checkpoint path.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on the number of steps per episode during playback.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between rendered steps to control pacing in the recording.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Environment reset seed for reproducibility.",
    )
    return parser.parse_args()


def _normalise_path(path_str: str) -> Path:
    candidate = PROJECT_ROOT / Path(path_str)
    return candidate.resolve()


def select_best_run() -> RunSelection:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Could not locate training/reports/summary_metrics.csv. Run the analysis notebook to regenerate reports."
        )

    summary_df = pd.read_csv(SUMMARY_PATH)
    run_rows = summary_df[summary_df.get("metadata_source", "run") == "run"].copy()
    if run_rows.empty:
        raise ValueError("No per-run entries found in summary_metrics.csv.")

    best_row = run_rows.loc[run_rows["eval_mean_reward"].idxmax()]
    algo = str(best_row["algo"]).upper()
    path_str = best_row.get("best_model_path") or best_row.get("final_model_path") or best_row.get("policy_path")
    if not path_str:
        raise ValueError(f"No checkpoint path recorded for algorithm {algo}.")

    checkpoint = _normalise_path(path_str)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found on disk: {checkpoint}")

    metadata = json.loads(best_row.to_json())
    return RunSelection(algo=algo, checkpoint=checkpoint, metadata=metadata)


def load_model(algo: str, checkpoint: Path, env: RwandaHealthEnv):
    algo = algo.upper()
    if algo in SB3_CLASSES:
        model_class = SB3_CLASSES[algo]
        return model_class.load(str(checkpoint), env=env)
    raise NotImplementedError(
        f"Simulation for algorithm '{algo}' is not implemented. Provide a Stable Baselines3 checkpoint instead."
    )


def run_episode(
    model,
    env: RwandaHealthEnv,
    deterministic: bool,
    max_steps: Optional[int],
    step_delay: float,
    episode_index: int,
) -> float:
    obs, _ = env.reset()
    total_reward = 0.0
    step = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        env.render()

        action_name = ACTION_MEANINGS.get(int(action), str(int(action)))
        total_reward += float(reward)

        print(
            f"[Episode {episode_index + 1} | Step {step:03d}] "
            f"Action={action_name:<25} Reward={reward:+.3f} Total={total_reward:+.3f} "
            f"Queue={info.get('queue_length')} Kits={info.get('test_kits')} Meds={info.get('meds')}"
        )

        step += 1
        done = terminated or truncated or (max_steps is not None and step >= max_steps)

        if step_delay > 0.0:
            time.sleep(step_delay)

    return total_reward


def describe_run(selection: RunSelection) -> None:
    print("\n=== Selected Run Summary ===")
    print(f"Algorithm      : {selection.algo}")
    print(f"Checkpoint     : {selection.checkpoint}")
    mean_reward = selection.metadata.get("eval_mean_reward")
    std_reward = selection.metadata.get("eval_std_reward")
    walltime = selection.metadata.get("training_walltime_sec")
    if mean_reward is not None and std_reward is not None:
        print(f"Eval mean/std  : {mean_reward:.3f} ± {std_reward:.3f}")
    if walltime is not None:
        print(f"Train walltime : {walltime / 60:.1f} min")
    hyper = selection.metadata.get("hyperparameters")
    if hyper:
        print("Hyperparameters:")
        print(hyper)
    print("============================\n")


def main() -> None:
    args = parse_args()

    if args.model_path is None:
        selection = select_best_run()
    else:
        if args.algo is None:
            raise ValueError("--model-path requires --algo to specify the loader to use.")
        checkpoint = args.model_path.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Provided checkpoint not found: {checkpoint}")
        selection = RunSelection(algo=args.algo.upper(), checkpoint=checkpoint, metadata={})

    describe_run(selection)

    env = RwandaHealthEnv(render_mode=args.render_mode)
    if args.max_steps is not None:
        env.max_steps = args.max_steps
    if args.seed is not None:
        env.reset(seed=args.seed)

    model = load_model(selection.algo, selection.checkpoint, env)

    for episode in range(args.episodes):
        episode_return = run_episode(
            model=model,
            env=env,
            deterministic=args.deterministic,
            max_steps=args.max_steps,
            step_delay=args.step_delay,
            episode_index=episode,
        )
        print(f"Episode {episode + 1} return: {episode_return:+.3f}")

    env.close()


if __name__ == "__main__":
    main()
