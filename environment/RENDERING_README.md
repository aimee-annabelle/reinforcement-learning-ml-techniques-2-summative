# Rendering Module for Rwanda Health Clinic RL Environment

This module provides pygame-based visualization for the Rwanda Health Clinic reinforcement learning environment.

## Files

- **`rendering.py`**: Core rendering module with `ClinicRenderer` class
- **`demo_visualization.py`**: Standalone demo script showing the visualization

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

### Option 1: Use with Your RL Environment

```python
from environment.rendering import ClinicRenderer

# In your environment class
class RwandaHealthEnv(gym.Env):
    def __init__(self, render_mode="human"):
        # ... your initialization code ...

        if render_mode == "human":
            self.renderer = ClinicRenderer()

    def render(self):
        if self.render_mode == "human":
            state = {
                'patient_age': self.patient_age,
                'symptom_severity': self.symptom_severity,
                'chronic_risk': self.chronic_risk,
                'infection_risk': self.infection_risk,
                'comorbidity_flag': self.comorbidity_flag,
                'ncd_tested': self.ncd_tested,
                'infection_tested': self.infection_tested,
                'diagnosed': self.diagnosed,
                'treated': self.treated,
                'test_kits': self.test_kits,
                'initial_test_kits': self.initial_test_kits,
                'meds': self.meds,
                'initial_meds': self.initial_meds,
                'queue_length': self.queue_length,
                'max_queue': self.max_queue,
                'last_action': self.last_action,
                'last_reward': self.last_reward,
                'step_count': self.step_count,
                'max_steps': self.max_steps,
            }
            self.renderer.render(state)

    def close(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()
```

### Option 2: Run Standalone Demo

```bash
# From the project root directory
python environment/demo_visualization.py
```

This will show a simulation with random states updating over 30 steps.

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

When training RL agents, you can toggle visualization:

```python
# Training mode (no visualization)
env = RwandaHealthEnv(render_mode=None)

# Evaluation mode (with visualization)
env = RwandaHealthEnv(render_mode="human")
```

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

- [x] Debugging environment logic
- [x] Recording training videos
- [x] Presenting results
- [x] Understanding agent behavior
