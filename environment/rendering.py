"""
Pygame Visualization Module for Rwanda Health Clinic RL Environment
Description: Provides visual rendering for the healthcare decision support system
"""

import pygame
import sys
from typing import Dict, Optional


class ClinicRenderer:
    """Handles pygame visualization for the Rwanda Health Clinic environment"""

    def __init__(self, screen_width: int = 1000, screen_height: int = 700, fps: int = 4):
        """
        Initialize the pygame renderer

        Args:
            screen_width: Width of the display window
            screen_height: Height of the display window
            fps: Frames per second for rendering
        """
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fps = fps

        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Rwanda Health Clinic - RL Environment")
        self.clock = pygame.time.Clock()

        # Initialize fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        # Define colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'green': (46, 204, 113),
            'red': (231, 76, 60),
            'blue': (52, 152, 219),
            'yellow': (241, 196, 15),
            'orange': (230, 126, 34),
            'gray': (189, 195, 199),
            'dark_gray': (52, 73, 94),
            'light_blue': (174, 214, 241),
            'light_green': (162, 217, 206),
            'light_red': (245, 183, 177),
            'light_yellow': (255, 250, 205),
        }

        # Action meanings
        self.action_meanings = {
            0: "Request NCD test",
            1: "Request infection test",
            2: "Diagnose chronic condition",
            3: "Diagnose infection",
            4: "Allocate medication",
            5: "Refer patient",
            6: "Wait / no action"
        }

    def render(self, env_state: Dict) -> None:
        """
        Render the current state of the environment

        Args:
            env_state: Dictionary containing environment state variables
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        # Clear screen
        self.screen.fill(self.colors['white'])

        # Render different panels
        self._render_title()
        self._render_patient_panel(env_state)
        self._render_resources_panel(env_state)
        self._render_queue_panel(env_state)
        self._render_action_panel(env_state)
        self._render_progress_panel(env_state)
        self._render_action_legend()

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _render_title(self) -> None:
        """Render the title bar"""
        title = self.font_large.render(
            "Rwanda Health Clinic - AI Decision Support",
            True,
            self.colors['dark_gray']
        )
        self.screen.blit(title, (20, 20))

        # Divider line
        pygame.draw.line(
            self.screen,
            self.colors['gray'],
            (20, 70),
            (self.screen_width - 20, 70),
            2
        )

    def _render_patient_panel(self, state: Dict) -> None:
        """Render patient information panel"""
        panel_x = 20
        panel_y = 90
        panel_width = 300
        panel_height = 280

        # Background box
        pygame.draw.rect(
            self.screen,
            self.colors['light_blue'],
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=10
        )
        pygame.draw.rect(
            self.screen,
            self.colors['blue'],
            (panel_x, panel_y, panel_width, panel_height),
            2,
            border_radius=10
        )

        # Header
        header = self.font_medium.render("Current Patient", True, self.colors['dark_gray'])
        self.screen.blit(header, (panel_x + 15, panel_y + 10))

        # Patient details
        y_offset = panel_y + 50
        details = [
            f"Age: {state.get('patient_age', 0)} years",
            f"Symptom Severity: {state.get('symptom_severity', 0):.2f}",
            f"Chronic Risk: {state.get('chronic_risk', 0):.2f}",
            f"Infection Risk: {state.get('infection_risk', 0):.2f}",
            f"Comorbidity: {'Yes' if state.get('comorbidity_flag', 0) else 'No'}",
        ]

        for detail in details:
            text = self.font_small.render(detail, True, self.colors['black'])
            self.screen.blit(text, (panel_x + 15, y_offset))
            y_offset += 30

        # Workflow status
        y_offset += 10
        workflow_header = self.font_small.render("Workflow Status:", True, self.colors['dark_gray'])
        self.screen.blit(workflow_header, (panel_x + 15, y_offset))
        y_offset += 25

        # Status indicators
        statuses = [
            ("NCD Tested", state.get('ncd_tested', False)),
            ("Infection Tested", state.get('infection_tested', False)),
            ("Diagnosed", state.get('diagnosed', False)),
            ("Treated", state.get('treated', False)),
        ]

        for status_name, status_val in statuses:
            color = self.colors['green'] if status_val else self.colors['red']
            pygame.draw.circle(self.screen, color, (panel_x + 25, y_offset + 8), 6)
            text = self.font_small.render(status_name, True, self.colors['black'])
            self.screen.blit(text, (panel_x + 40, y_offset))
            y_offset += 25

    def _render_resources_panel(self, state: Dict) -> None:
        """Render clinic resources panel"""
        panel_x = 340
        panel_y = 90
        panel_width = 300
        panel_height = 180

        # Background box
        pygame.draw.rect(
            self.screen,
            self.colors['light_green'],
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=10
        )
        pygame.draw.rect(
            self.screen,
            self.colors['green'],
            (panel_x, panel_y, panel_width, panel_height),
            2,
            border_radius=10
        )

        # Header
        header = self.font_medium.render("Clinic Resources", True, self.colors['dark_gray'])
        self.screen.blit(header, (panel_x + 15, panel_y + 10))

        # Test kits
        y_offset = panel_y + 50
        test_kits = state.get('test_kits', 0)
        initial_test_kits = state.get('initial_test_kits', 10)

        text = self.font_small.render(
            f"Test Kits: {test_kits}/{initial_test_kits}",
            True,
            self.colors['black']
        )
        self.screen.blit(text, (panel_x + 15, y_offset))

        # Progress bar for test kits
        bar_width = 250
        bar_height = 20
        bar_fill = int((test_kits / max(initial_test_kits, 1)) * bar_width)

        pygame.draw.rect(
            self.screen,
            self.colors['gray'],
            (panel_x + 15, y_offset + 25, bar_width, bar_height),
            border_radius=5
        )
        pygame.draw.rect(
            self.screen,
            self.colors['blue'],
            (panel_x + 15, y_offset + 25, bar_fill, bar_height),
            border_radius=5
        )

        # Medications
        y_offset += 60
        meds = state.get('meds', 0)
        initial_meds = state.get('initial_meds', 20)

        text = self.font_small.render(
            f"Medications: {meds}/{initial_meds}",
            True,
            self.colors['black']
        )
        self.screen.blit(text, (panel_x + 15, y_offset))

        # Progress bar for medications
        bar_fill = int((meds / max(initial_meds, 1)) * bar_width)

        pygame.draw.rect(
            self.screen,
            self.colors['gray'],
            (panel_x + 15, y_offset + 25, bar_width, bar_height),
            border_radius=5
        )
        pygame.draw.rect(
            self.screen,
            self.colors['green'],
            (panel_x + 15, y_offset + 25, bar_fill, bar_height),
            border_radius=5
        )

    def _render_queue_panel(self, state: Dict) -> None:
        """Render patient queue panel"""
        panel_x = 340
        panel_y = 290
        panel_width = 300
        panel_height = 170

        queue_length = state.get('queue_length', 0)
        max_queue = state.get('max_queue', 30)
        queue_ratio = queue_length / max(max_queue, 1)

        # Determine color based on queue status
        if queue_ratio > 0.7:
            queue_color = self.colors['red']
            bg_color = self.colors['light_red']
        elif queue_ratio > 0.4:
            queue_color = self.colors['yellow']
            bg_color = self.colors['light_yellow']
        else:
            queue_color = self.colors['green']
            bg_color = self.colors['light_green']

        # Background box
        pygame.draw.rect(
            self.screen,
            bg_color,
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=10
        )
        pygame.draw.rect(
            self.screen,
            queue_color,
            (panel_x, panel_y, panel_width, panel_height),
            2,
            border_radius=10
        )

        # Header
        header = self.font_medium.render("Patient Queue", True, self.colors['dark_gray'])
        self.screen.blit(header, (panel_x + 15, panel_y + 10))

        # Queue number (large)
        queue_text = self.font_large.render(str(queue_length), True, queue_color)
        text_rect = queue_text.get_rect(center=(panel_x + panel_width // 2, panel_y + 70))
        self.screen.blit(queue_text, text_rect)

        # Max queue
        max_text = self.font_small.render(f"/ {max_queue} max", True, self.colors['gray'])
        max_rect = max_text.get_rect(center=(panel_x + panel_width // 2, panel_y + 110))
        self.screen.blit(max_text, max_rect)

        # Warning if high load
        if queue_ratio > 0.7:
            warning = self.font_small.render("⚠ High Load!", True, self.colors['red'])
            warning_rect = warning.get_rect(center=(panel_x + panel_width // 2, panel_y + 140))
            self.screen.blit(warning, warning_rect)

    def _render_action_panel(self, state: Dict) -> None:
        """Render last action panel"""
        panel_x = 660
        panel_y = 90
        panel_width = 320
        panel_height = 180

        # Background box
        pygame.draw.rect(
            self.screen,
            self.colors['light_yellow'],
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=10
        )
        pygame.draw.rect(
            self.screen,
            self.colors['orange'],
            (panel_x, panel_y, panel_width, panel_height),
            2,
            border_radius=10
        )

        # Header
        header = self.font_medium.render("Last Action", True, self.colors['dark_gray'])
        self.screen.blit(header, (panel_x + 15, panel_y + 10))

        # Action text (with word wrapping)
        last_action = state.get('last_action', 6)
        action_text = self.action_meanings.get(last_action, "Unknown")

        # Simple word wrap
        words = action_text.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            if self.font_small.size(test_line)[0] > panel_width - 30:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        y_offset = panel_y + 55
        for line in lines:
            text = self.font_small.render(line, True, self.colors['black'])
            self.screen.blit(text, (panel_x + 15, y_offset))
            y_offset += 30

        # Reward display
        y_offset = panel_y + 130
        last_reward = state.get('last_reward', 0)

        reward_color = (self.colors['green'] if last_reward > 0
                       else self.colors['red'] if last_reward < 0
                       else self.colors['gray'])

        reward_text = self.font_medium.render(
            f"Reward: {last_reward:+.2f}",
            True,
            reward_color
        )
        self.screen.blit(reward_text, (panel_x + 15, y_offset))

    def _render_progress_panel(self, state: Dict) -> None:
        """Render episode progress panel"""
        panel_x = 660
        panel_y = 290
        panel_width = 320
        panel_height = 170

        # Background box
        pygame.draw.rect(
            self.screen,
            self.colors['light_blue'],
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=10
        )
        pygame.draw.rect(
            self.screen,
            self.colors['blue'],
            (panel_x, panel_y, panel_width, panel_height),
            2,
            border_radius=10
        )

        # Header
        header = self.font_medium.render("Episode Progress", True, self.colors['dark_gray'])
        self.screen.blit(header, (panel_x + 15, panel_y + 10))

        # Step counter
        y_offset = panel_y + 50
        step_count = state.get('step_count', 0)
        max_steps = state.get('max_steps', 50)

        step_text = self.font_small.render(
            f"Step: {step_count} / {max_steps}",
            True,
            self.colors['black']
        )
        self.screen.blit(step_text, (panel_x + 15, y_offset))

        # Progress bar
        progress_width = 280
        progress_fill = int((step_count / max(max_steps, 1)) * progress_width)

        pygame.draw.rect(
            self.screen,
            self.colors['gray'],
            (panel_x + 15, y_offset + 30, progress_width, 20),
            border_radius=5
        )
        pygame.draw.rect(
            self.screen,
            self.colors['blue'],
            (panel_x + 15, y_offset + 30, progress_fill, 20),
            border_radius=5
        )

        # Completion percentage
        completion = (step_count / max(max_steps, 1)) * 100
        completion_text = self.font_small.render(
            f"{completion:.1f}% Complete",
            True,
            self.colors['black']
        )
        self.screen.blit(completion_text, (panel_x + 15, y_offset + 65))

        # Additional info
        y_offset += 100
        info_text = self.font_small.render(
            f"Total Patients Seen: {step_count}",
            True,
            self.colors['black']
        )
        self.screen.blit(info_text, (panel_x + 15, y_offset))

    def _render_action_legend(self) -> None:
        """Render action legend at the bottom"""
        legend_y = 600
        legend_height = 80

        # Background
        pygame.draw.rect(
            self.screen,
            self.colors['gray'],
            (20, legend_y, self.screen_width - 40, legend_height),
            border_radius=10
        )

        # Header
        header = self.font_small.render("Available Actions:", True, self.colors['white'])
        self.screen.blit(header, (30, legend_y + 10))

        # Actions
        actions_display = [
            "0: NCD Test", "1: Infection Test", "2: Diagnose Chronic",
            "3: Diagnose Infection", "4: Allocate Med", "5: Refer Patient", "6: Wait"
        ]

        x_pos = 30
        y_pos = legend_y + 40

        for i, action in enumerate(actions_display):
            if i == 4:  # Move to second row after 4 items
                x_pos = 30
                y_pos += 25

            text = self.font_small.render(action, True, self.colors['white'])
            self.screen.blit(text, (x_pos, y_pos))
            x_pos += 160

    def close(self) -> None:
        """Close the pygame window and quit"""
        pygame.quit()


def create_renderer(screen_width: int = 1000, screen_height: int = 700, fps: int = 4) -> ClinicRenderer:
    """
    Factory function to create a ClinicRenderer instance

    Args:
        screen_width: Width of the display window
        screen_height: Height of the display window
        fps: Frames per second for rendering

    Returns:
        ClinicRenderer instance
    """
    return ClinicRenderer(screen_width, screen_height, fps)
