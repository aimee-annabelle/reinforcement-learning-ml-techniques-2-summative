"""
Example script demonstrating pygame visualization for Rwanda Health Clinic RL Environment
Run this script to see the visualization in action with a random agent
"""

import sys
import os
import time

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.rendering import ClinicRenderer


def create_sample_state(step: int, max_steps: int = 50) -> dict:
    """
    Create a sample environment state for demonstration

    Args:
        step: Current step number
        max_steps: Maximum steps in episode

    Returns:
        Dictionary containing environment state
    """
    import random

    return {
        'patient_age': random.randint(20, 70),
        'symptom_severity': random.uniform(0.3, 0.9),
        'chronic_risk': random.uniform(0.2, 0.8),
        'infection_risk': random.uniform(0.2, 0.8),
        'comorbidity_flag': random.choice([0, 1]),
        'ncd_tested': random.choice([True, False]),
        'infection_tested': random.choice([True, False]),
        'diagnosed': random.choice([True, False]),
        'treated': random.choice([True, False]),
        'test_kits': max(0, 10 - step // 5),
        'initial_test_kits': 10,
        'meds': max(0, 20 - step // 3),
        'initial_meds': 20,
        'queue_length': random.randint(0, 25),
        'max_queue': 30,
        'last_action': random.randint(0, 6),
        'last_reward': random.uniform(-5, 10),
        'step_count': step,
        'max_steps': max_steps,
    }


def main():
    """Main demonstration function"""
    print("=" * 60)
    print("Rwanda Health Clinic - Pygame Visualization Demo")
    print("=" * 60)
    print("\nInitializing renderer...")

    # Create renderer
    renderer = ClinicRenderer(screen_width=1000, screen_height=700, fps=2)

    print("Renderer initialized successfully!")
    print("\nRunning simulation with random actions...")
    print("Close the pygame window to stop the simulation.\n")

    try:
        max_steps = 30

        for step in range(max_steps):
            # Generate sample state
            state = create_sample_state(step, max_steps)

            # Render the state
            renderer.render(state)

            # Print step info to console
            print(f"Step {step + 1}/{max_steps} - "
                  f"Action: {renderer.action_meanings[state['last_action']]} - "
                  f"Reward: {state['last_reward']:+.2f}")

            # Slow down for viewing
            time.sleep(0.8)

        print("\n" + "=" * 60)
        print("Simulation complete!")
        print("=" * 60)

        # Keep window open for a moment
        time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")

    except Exception as e:
        print(f"\n\nError occurred: {e}")

    finally:
        print("Closing renderer...")
        renderer.close()
        print("Done!")


if __name__ == "__main__":
    main()
