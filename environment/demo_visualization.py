"""
Random rollout demo that drives the RwandaHealthEnv environment with pygame rendering.
"""

import sys
import os
import time

# Ensure the package root is discoverable when running as a standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import RwandaHealthEnv, ACTION_MEANINGS


def main() -> None:
    print("=" * 60)
    print("Rwanda Health Clinic - Environment Visualization Demo")
    print("=" * 60)
    print("\nSpawning environment...")

    env = RwandaHealthEnv(render_mode="human")

    try:
        obs, info = env.reset(seed=42)
        print("Initial observation shape:", obs.shape)
        print("Initial info:", info)

        max_steps = 50
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            print(
                f"Step {step + 1}/{max_steps} | Action={ACTION_MEANINGS[action]} | Reward={reward:+.2f} | "
                f"Queue={info['queue_length']} | Kits={info['test_kits']} | Meds={info['meds']}"
            )

            time.sleep(0.5)

            if terminated or truncated:
                print("Episode ended due to", "termination" if terminated else "truncation")
                obs, info = env.reset()
                print("\nEnvironment reset. Continue random rollout...\n")

        print("\n" + "=" * 60)
        print("Demo finished")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")

    finally:
        env.close()
        print("Renderer closed.")


if __name__ == "__main__":
    main()
