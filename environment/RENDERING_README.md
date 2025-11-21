# Rendering Module for Rwanda Health Clinic RL Environment

This module provides pygame-based visualization for the Rwanda Health Clinic reinforcement learning environment.

## Files

- **`rendering.py`**: Core rendering module with `ClinicRenderer` class
- **`custom_env.py`**: Gymnasium environment that drives the renderer during training and demos
- **`demo_visualization.py`**: Standalone rollout script that samples random actions from the custom environment and renders them

## Features

### Visual Components

1. **Patient Information Panel** (Left)

   - Age, symptom severity, risk factors
   - Comorbidity status
   - Workflow status indicators (tested, diagnosed, treated)

2. **Clinic Resources Panel** (Center-Top)

   - Test kits availability with progress bar
   - Medications availability with progress bar

3. **Patient Queue Panel** (Center-Bottom)

   - Current queue length
   - Color-coded warnings (green/yellow/red)
   - Maximum capacity indicator

4. **Last Action Panel** (Right-Top)

   - Action taken by the agent
   - Reward received (color-coded)

5. **Episode Progress Panel** (Right-Bottom)

   - Current step / total steps
   - Progress bar
   - Completion percentage

6. **Action Legend** (Bottom)
   - All available actions displayed

## Usage

### Option 1: Run the standalone demo

```bash
# From the project root directory
python environment/demo_visualization.py
```

The script creates `RwandaHealthEnv(render_mode="human")`, samples random actions, and renders the full pygame dashboard for approximately fifty steps (or until the episode ends).

### Option 2: Import the environment in notebooks or training scripts

```python
from environment.custom_env import RwandaHealthEnv

env = RwandaHealthEnv(render_mode="human")  # or None for headless training
obs, info = env.reset(seed=42)
```

The `training/rl_training.ipynb` notebook and any other scripts should import the environment from `environment.custom_env` so that training and evaluation share the same implementation.

## Requirements

```
pygame>=2.6.0
numpy>=1.21.0
```

## Color Coding

- **Green**: Positive status, good resource levels
- **Yellow**: Warning status, medium resource levels
- **Red**: Critical status, low resource levels, negative rewards
- **Blue**: Neutral information, test kits
- **Orange**: Action highlights

## Customization

You can customize the renderer by passing parameters:

```python
renderer = ClinicRenderer(
    screen_width=1200,   # Default: 1000
    screen_height=800,   # Default: 700
    fps=10              # Default: 4
)
```

## Integration with Training

When training RL agents, set `render_mode=None` to avoid opening a window. Switch to `render_mode="human"` for evaluation runs or video captures.

## Troubleshooting

**Issue**: Pygame window not responding

- **Solution**: Make sure to call `renderer.render()` in a loop and handle pygame events

**Issue**: Window closes immediately

- **Solution**: Add a delay or wait loop after rendering

**Issue**: Display is too fast/slow

- **Solution**: Adjust the `fps` parameter when creating the renderer

## Example Output

The visualization displays:

- Real-time patient information
- Resource depletion tracking
- Queue management status
- Agent decision-making process
- Reward feedback

Perfect for:

Suitable for debugging environment logic, recording training videos, presenting results, and understanding agent behavior.
