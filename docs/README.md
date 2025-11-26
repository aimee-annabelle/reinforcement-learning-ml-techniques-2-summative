# Documentation

This directory contains architectural diagrams and documentation for the Rwanda Health Clinic RL Agent project.

## Diagrams

### 1. Environment Architecture Diagram (`environment_diagram.png`)

This comprehensive diagram illustrates the complete architecture of the reinforcement learning environment, including:

- **RL Agent**: The neural network-based agent (DQN, PPO, A2C, or REINFORCE) that learns to make optimal decisions
- **State Space**: All 12 dimensions of the observation space that the agent receives
- **Action Space**: The 7 discrete actions available to the agent
- **Patient Conditions**: The four probabilistic patient states in the environment
- **Clinic Resources**: Initial resources and terminal conditions
- **Reward Signal**: The comprehensive reward structure that guides learning

**Key Components Shown:**

- Information flow between agent and environment
- State observation and action selection process
- Reward feedback loop for learning
- Patient state transitions
- Resource constraints and management

### 2. Workflow Diagram (`workflow_diagram.png`)

This diagram shows the cyclical agent-environment interaction pattern:

1. **Observe**: Agent receives state st from environment
2. **Decide**: Agent selects action at based on current policy
3. **Execute**: Action is executed in the environment
4. **Update**: Environment transitions to new state st+1
5. **Evaluate**: Agent receives reward rt for the action
6. **Learn**: Agent updates policy to maximize cumulative reward

**Mathematical Objective:**

```
Maximize Σ γᵗ · rt (cumulative discounted reward)
```

## Usage

These diagrams are referenced in the main README.md and can be used for:

- Understanding the environment design
- Explaining the RL framework to stakeholders
- Academic presentations and reports
- Documentation of the system architecture

## Regenerating Diagrams

To regenerate the diagrams after making changes:

```bash
python docs/generate_diagram.py
```

This will create/update:

- `environment_diagram.png` - Detailed architecture (16x10 inches, 300 DPI)
- `workflow_diagram.png` - Interaction workflow (14x8 inches, 300 DPI)

## Design Principles

The diagrams follow these principles:

- **Color coding**: Consistent colors for different components (agent=blue, environment=green, state=red, action=orange, reward=purple)
- **Clear flow**: Arrows show information and control flow
- **Comprehensive detail**: All key components are labeled and explained
- **Professional quality**: High DPI suitable for reports and presentations
- **Accessibility**: Clear fonts and high contrast for readability

## Requirements

The diagram generation script requires:

- matplotlib
- numpy

These are already included in the project's `requirements.txt`.
