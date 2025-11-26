"""
Generate environment architecture diagram for Rwanda Health Clinic RL Agent
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
agent_color = '#3498db'  # Blue
env_color = '#2ecc71'    # Green
state_color = '#e74c3c'  # Red
action_color = '#f39c12' # Orange
reward_color = '#9b59b6' # Purple
patient_color = '#1abc9c' # Teal
resource_color = '#e67e22' # Dark orange

# Title
ax.text(8, 9.5, 'Rwanda Health Clinic RL Environment Architecture',
        ha='center', va='top', fontsize=20, weight='bold')

# ============= ENVIRONMENT BOX =============
env_box = FancyBboxPatch((0.5, 1), 15, 7,
                         boxstyle="round,pad=0.1",
                         edgecolor=env_color, facecolor='#ecf9f2',
                         linewidth=3, zorder=0)
ax.add_patch(env_box)
ax.text(8, 7.7, 'Rwanda Health Clinic Environment',
        ha='center', fontsize=16, weight='bold', color=env_color)

# ============= AGENT BOX =============
agent_box = FancyBboxPatch((1.5, 4.5), 2.5, 2,
                          boxstyle="round,pad=0.1",
                          edgecolor=agent_color, facecolor='#d6eaf8',
                          linewidth=2.5)
ax.add_patch(agent_box)
ax.text(2.75, 6.1, 'RL Agent', ha='center', fontsize=13, weight='bold')
ax.text(2.75, 5.7, 'DQN/PPO', ha='center', fontsize=10)
ax.text(2.75, 5.4, 'A2C/REINFORCE', ha='center', fontsize=10)
ax.text(2.75, 5.0, 'Neural Network', ha='center', fontsize=9, style='italic')

# ============= STATE SPACE BOX =============
state_box = FancyBboxPatch((5.5, 5.5), 4.5, 1.8,
                          boxstyle="round,pad=0.05",
                          edgecolor=state_color, facecolor='#fadbd8',
                          linewidth=2)
ax.add_patch(state_box)
ax.text(7.75, 7.1, 'State Space (12D)', ha='center', fontsize=12, weight='bold', color=state_color)

# State features (two columns)
state_features_left = [
    '• Patient Age',
    '• Symptom Severity',
    '• Chronic Risk',
    '• Infection Risk',
    '• Comorbidity Flag',
    '• Test Kits Remaining'
]
state_features_right = [
    '• Medications Remaining',
    '• Queue Length',
    '• Time Step',
    '• Last Action',
    '• NCD Test Done',
    '• Infection Test Done'
]

y_start = 6.8
for i, feature in enumerate(state_features_left):
    ax.text(6.0, y_start - i*0.25, feature, fontsize=8, va='center')
for i, feature in enumerate(state_features_right):
    ax.text(8.0, y_start - i*0.25, feature, fontsize=8, va='center')

# ============= ACTION SPACE BOX =============
action_box = FancyBboxPatch((5.5, 2.2), 4.5, 2.8,
                           boxstyle="round,pad=0.05",
                           edgecolor=action_color, facecolor='#fef5e7',
                           linewidth=2)
ax.add_patch(action_box)
ax.text(7.75, 4.85, 'Action Space (7 discrete)', ha='center', fontsize=12, weight='bold', color=action_color)

# Actions
actions = [
    '0: Request NCD Test',
    '1: Request Infection Test',
    '2: Diagnose Chronic',
    '3: Diagnose Infection',
    '4: Allocate Medication',
    '5: Refer Patient',
    '6: Wait / No Action'
]
y_start = 4.5
for i, action in enumerate(actions):
    ax.text(6.0, y_start - i*0.3, action, fontsize=9, va='center')

# ============= PATIENT STATE BOX =============
patient_box = FancyBboxPatch((11, 5.5), 3.5, 1.8,
                            boxstyle="round,pad=0.05",
                            edgecolor=patient_color, facecolor='#d5f4e6',
                            linewidth=2)
ax.add_patch(patient_box)
ax.text(12.75, 7.1, 'Patient Conditions', ha='center', fontsize=12, weight='bold', color=patient_color)

conditions = [
    'Healthy/Mild (40%)',
    'Chronic Disease (25%)',
    'Infection (25%)',
    'Both Serious (10%)'
]
y_start = 6.7
for i, condition in enumerate(conditions):
    ax.text(11.5, y_start - i*0.35, condition, fontsize=9, va='center')

# ============= RESOURCES BOX =============
resource_box = FancyBboxPatch((11, 2.2), 3.5, 2.8,
                             boxstyle="round,pad=0.05",
                             edgecolor=resource_color, facecolor='#fef9e7',
                             linewidth=2)
ax.add_patch(resource_box)
ax.text(12.75, 4.85, 'Clinic Resources', ha='center', fontsize=12, weight='bold', color=resource_color)

resources = [
    'Test Kits: 10 initial',
    'Medications: 20 initial',
    'Max Queue: 30 patients',
    'Max Steps: 50',
    '',
    'Terminal if:',
    '• Test kits = 0',
    '• Medications = 0',
    '• Queue ≥ 30'
]
y_start = 4.5
for i, resource in enumerate(resources):
    fontsize = 9 if i < 4 else 8
    weight = 'normal' if i != 5 else 'bold'
    ax.text(11.5, y_start - i*0.28, resource, fontsize=fontsize, va='center', weight=weight)

# ============= REWARD BOX =============
reward_box = FancyBboxPatch((1.5, 1.5), 2.5, 2.5,
                           boxstyle="round,pad=0.05",
                           edgecolor=reward_color, facecolor='#ebdef0',
                           linewidth=2)
ax.add_patch(reward_box)
ax.text(2.75, 3.85, 'Reward Signal', ha='center', fontsize=12, weight='bold', color=reward_color)

rewards = [
    '+10: Correct Dx + test',
    '+6-8: Treatment',
    '+5: Complete care',
    '+3-8: Referral',
    '-6-10: Wrong Dx',
    '-10: Resources ∅',
    '-1×queue: Pressure'
]
y_start = 3.5
for i, reward in enumerate(rewards):
    ax.text(2.0, y_start - i*0.27, reward, fontsize=8, va='center')

# ============= ARROWS =============

# Agent to Action (Decision)
arrow1 = FancyArrowPatch((4, 5.5), (5.4, 3.5),
                        arrowstyle='->', mutation_scale=25, linewidth=2.5,
                        color=action_color, zorder=10)
ax.add_patch(arrow1)
ax.text(4.5, 4.3, 'Select\nAction', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# State to Agent (Observation)
arrow2 = FancyArrowPatch((7.75, 5.4), (4, 5.8),
                        arrowstyle='->', mutation_scale=25, linewidth=2.5,
                        color=state_color, zorder=10)
ax.add_patch(arrow2)
ax.text(5.8, 5.8, 'Observe\nState', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Action to Environment (Execution)
arrow3 = FancyArrowPatch((7.75, 2.1), (7.75, 1.3),
                        arrowstyle='->', mutation_scale=25, linewidth=2.5,
                        color='#34495e', zorder=10)
ax.add_patch(arrow3)
ax.text(8.5, 1.7, 'Execute', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Environment to Reward (Feedback)
arrow4 = FancyArrowPatch((5.4, 1.3), (4, 2.7),
                        arrowstyle='->', mutation_scale=25, linewidth=2.5,
                        color=reward_color, zorder=10)
ax.add_patch(arrow4)
ax.text(4.5, 1.8, 'Reward\nFeedback', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Reward to Agent (Learning)
arrow5 = FancyArrowPatch((2.75, 4.4), (2.75, 4.05),
                        arrowstyle='->', mutation_scale=25, linewidth=2.5,
                        color=reward_color, zorder=10)
ax.add_patch(arrow5)
ax.text(3.5, 4.2, 'Learn', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Patient to State (Information)
arrow6 = FancyArrowPatch((11, 6.5), (10.1, 6.5),
                        arrowstyle='->', mutation_scale=20, linewidth=2,
                        color=patient_color, linestyle='--', zorder=5)
ax.add_patch(arrow6)

# Resources to State (Status)
arrow7 = FancyArrowPatch((11, 3.5), (10.1, 5.7),
                        arrowstyle='->', mutation_scale=20, linewidth=2,
                        color=resource_color, linestyle='--', zorder=5)
ax.add_patch(arrow7)

# ============= LEGEND =============
ax.text(1, 0.8, 'RL Training Loop:', fontsize=11, weight='bold')
ax.text(1, 0.5, '1. Agent observes state  →  2. Selects action  →  3. Environment responds',
        fontsize=9)
ax.text(1, 0.2, '4. Receives reward  →  5. Updates policy  →  Repeat', fontsize=9)

# Performance note
ax.text(14.5, 0.6, 'Best Performance:', ha='right', fontsize=10, weight='bold')
ax.text(14.5, 0.3, 'DQN: 189.74 reward', ha='right', fontsize=9, color=agent_color)
ax.text(14.5, 0.05, 'PPO: 171.40 reward', ha='right', fontsize=9, color=agent_color)

# Add process flow indicators
circle1 = Circle((2.75, 5.5), 0.08, color=agent_color, zorder=15)
ax.add_patch(circle1)
ax.text(2.75, 5.5, '1', ha='center', va='center', fontsize=10,
        weight='bold', color='white', zorder=16)

circle2 = Circle((7.75, 6.5), 0.08, color=state_color, zorder=15)
ax.add_patch(circle2)
ax.text(7.75, 6.5, '2', ha='center', va='center', fontsize=10,
        weight='bold', color='white', zorder=16)

circle3 = Circle((7.75, 3.5), 0.08, color=action_color, zorder=15)
ax.add_patch(circle3)
ax.text(7.75, 3.5, '3', ha='center', va='center', fontsize=10,
        weight='bold', color='white', zorder=16)

circle4 = Circle((2.75, 2.7), 0.08, color=reward_color, zorder=15)
ax.add_patch(circle4)
ax.text(2.75, 2.7, '4', ha='center', va='center', fontsize=10,
        weight='bold', color='white', zorder=16)

plt.tight_layout()
plt.savefig('docs/environment_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Environment diagram saved to docs/environment_diagram.png")
plt.close()

# Create a simpler workflow diagram
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 8)
ax2.axis('off')

# Title
ax2.text(7, 7.5, 'Agent-Environment Interaction Workflow',
         ha='center', va='top', fontsize=18, weight='bold')

# Create circular workflow
center_x, center_y = 7, 4
radius = 2.5

# Agent
agent_circle = Circle((center_x, center_y + radius), 0.8,
                      color=agent_color, zorder=10)
ax2.add_patch(agent_circle)
ax2.text(center_x, center_y + radius, 'AGENT\n(Neural Net)',
         ha='center', va='center', fontsize=11, weight='bold', color='white')

# Environment
env_circle = Circle((center_x, center_y - radius), 0.8,
                    color=env_color, zorder=10)
ax2.add_patch(env_circle)
ax2.text(center_x, center_y - radius, 'ENVIRONMENT\n(Clinic)',
         ha='center', va='center', fontsize=11, weight='bold', color='white')

# State (right)
state_circle = Circle((center_x + radius, center_y), 0.6,
                      color=state_color, zorder=10)
ax2.add_patch(state_circle)
ax2.text(center_x + radius, center_y, 'STATE\nst',
         ha='center', va='center', fontsize=10, weight='bold', color='white')

# Action (left)
action_circle = Circle((center_x - radius, center_y), 0.6,
                       color=action_color, zorder=10)
ax2.add_patch(action_circle)
ax2.text(center_x - radius, center_y, 'ACTION\nat',
         ha='center', va='center', fontsize=10, weight='bold', color='white')

# Reward (bottom-left)
reward_circle = Circle((center_x - radius*0.7, center_y - radius*0.7), 0.5,
                       color=reward_color, zorder=10)
ax2.add_patch(reward_circle)
ax2.text(center_x - radius*0.7, center_y - radius*0.7, 'REWARD\nrt',
         ha='center', va='center', fontsize=9, weight='bold', color='white')

# Arrows showing flow
# State to Agent
arrow_s2a = FancyArrowPatch(
    (center_x + radius - 0.6, center_y + 0.3),
    (center_x + 0.3, center_y + radius - 0.8),
    arrowstyle='->', mutation_scale=30, linewidth=3,
    color=state_color, zorder=5
)
ax2.add_patch(arrow_s2a)

# Agent to Action
arrow_a2act = FancyArrowPatch(
    (center_x - 0.3, center_y + radius - 0.8),
    (center_x - radius + 0.6, center_y + 0.3),
    arrowstyle='->', mutation_scale=30, linewidth=3,
    color=action_color, zorder=5
)
ax2.add_patch(arrow_a2act)

# Action to Environment
arrow_act2e = FancyArrowPatch(
    (center_x - radius + 0.3, center_y - 0.6),
    (center_x - 0.3, center_y - radius + 0.8),
    arrowstyle='->', mutation_scale=30, linewidth=3,
    color=action_color, zorder=5
)
ax2.add_patch(arrow_act2e)

# Environment to State
arrow_e2s = FancyArrowPatch(
    (center_x + 0.3, center_y - radius + 0.8),
    (center_x + radius - 0.6, center_y - 0.3),
    arrowstyle='->', mutation_scale=30, linewidth=3,
    color=state_color, zorder=5
)
ax2.add_patch(arrow_e2s)

# Environment to Reward
arrow_e2r = FancyArrowPatch(
    (center_x - 0.5, center_y - radius + 0.5),
    (center_x - radius*0.7 + 0.4, center_y - radius*0.7 + 0.3),
    arrowstyle='->', mutation_scale=30, linewidth=3,
    color=reward_color, zorder=5
)
ax2.add_patch(arrow_e2r)

# Reward to Agent
arrow_r2a = FancyArrowPatch(
    (center_x - radius*0.7, center_y - radius*0.7 + 0.5),
    (center_x - 0.5, center_y + radius - 0.5),
    arrowstyle='->', mutation_scale=30, linewidth=3,
    color=reward_color, zorder=5, linestyle='--'
)
ax2.add_patch(arrow_r2a)

# Add labels for each transition
ax2.text(center_x + radius*0.8, center_y + radius*0.8, 'Observe',
         fontsize=10, weight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.text(center_x - radius*0.8, center_y + radius*0.8, 'Decide',
         fontsize=10, weight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.text(center_x - radius*0.8, center_y - radius*0.8, 'Execute',
         fontsize=10, weight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.text(center_x + radius*0.8, center_y - radius*0.8, 'Update',
         fontsize=10, weight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax2.text(center_x - radius*1.2, center_y, 'Evaluate',
         fontsize=10, weight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Add time step notation
ax2.text(7, 0.8, 'Time step t → t+1', ha='center', fontsize=12,
         style='italic', weight='bold')

# Add equation
ax2.text(7, 0.3, 'Objective: Maximize Σ γᵗ · rt  (cumulative discounted reward)',
         ha='center', fontsize=11, style='italic',
         bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))

plt.tight_layout()
plt.savefig('docs/workflow_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Workflow diagram saved to docs/workflow_diagram.png")
plt.close()

print("\n✓ All diagrams generated successfully!")
print("  - docs/environment_diagram.png (detailed architecture)")
print("  - docs/workflow_diagram.png (agent-environment interaction)")
