"""
Quick test script for the rendering module
Run this to verify that pygame visualization is working correctly
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.rendering import ClinicRenderer


def test_renderer():
    """Quick test of the renderer"""
    print("Testing ClinicRenderer...")
    print("-" * 40)

    try:
        # Create renderer
        print("✓ Creating renderer...")
        renderer = ClinicRenderer()

        # Test state
        test_state = {
            'patient_age': 45,
            'symptom_severity': 0.75,
            'chronic_risk': 0.6,
            'infection_risk': 0.4,
            'comorbidity_flag': 1,
            'ncd_tested': True,
            'infection_tested': False,
            'diagnosed': True,
            'treated': False,
            'test_kits': 7,
            'initial_test_kits': 10,
            'meds': 15,
            'initial_meds': 20,
            'queue_length': 12,
            'max_queue': 30,
            'last_action': 2,  # Diagnose chronic
            'last_reward': 8.5,
            'step_count': 15,
            'max_steps': 50,
        }

        print("✓ Rendering test state...")
        print("\nA pygame window should appear showing the clinic visualization.")
        print("The window will stay open for 3 seconds.\n")

        # Render for 3 seconds
        import time
        for i in range(6):  # 3 seconds at 2 fps
            renderer.render(test_state)
            time.sleep(0.5)

        print("✓ Closing renderer...")
        renderer.close()

        print("\n" + "=" * 40)
        print("✓ All tests passed!")
        print("=" * 40)
        print("\nThe rendering module is working correctly.")
        print("You can now use it in your environment.")

        return True

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nMake sure pygame is installed:")
        print("  pip install pygame")
        return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nSomething went wrong. Check the error message above.")
        return False


if __name__ == "__main__":
    print("=" * 40)
    print("Rendering Module Test")
    print("=" * 40)
    print()

    success = test_renderer()

    if success:
        print("\nNext steps:")
        print("  1. Run demo_visualization.py for a full demo")
        print("  2. Integrate with your environment using integrate_rendering.py")
        print("  3. See RENDERING_README.md for detailed documentation")
    else:
        print("\nPlease fix the errors above and try again.")

    sys.exit(0 if success else 1)
