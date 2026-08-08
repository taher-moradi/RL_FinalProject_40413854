"""
gui/renderer.py
Pygame Grid Renderer for Visualizing Dynamic Maze Environment, Value Heatmaps, and Policy Overlays.
"""

import pygame
import numpy as np
from typing import Dict, Tuple, Optional, Any

from environments.generator import MazeGenerator
from environments.maze import DynamicMazeEnv


class MazeRenderer:
    """
    Pygame Graphical Renderer for Dynamic Maze.
    Handles rendering grid tiles, agent animations, status HUD, and policy/value overlays.
    """

    # Color Palette Definition (RGB)
    COLOR_BG = (30, 30, 35)
    COLOR_EMPTY = (245, 245, 245)
    COLOR_WALL = (40, 44, 52)
    COLOR_PENALTY = (235, 87, 87)
    COLOR_START = (46, 204, 113)
    COLOR_KEY = (241, 196, 15)
    COLOR_DOOR_LOCKED = (155, 89, 182)
    COLOR_DOOR_UNLOCKED = (210, 180, 222)
    COLOR_GOAL = (52, 152, 219)
    COLOR_AGENT = (230, 126, 34)
    COLOR_GRID_LINE = (200, 200, 200)
    COLOR_PANEL_BG = (20, 22, 26)
    COLOR_TEXT = (240, 240, 240)
    COLOR_ENERGY_BAR = (46, 204, 113)

    def __init__(self, env: DynamicMazeEnv, cell_size: int = 40):
        self.env = env
        self.cell_size = cell_size
        self.grid_size = env.rows

        self.maze_width = self.grid_size * cell_size
        self.maze_height = self.grid_size * cell_size
        self.panel_width = 320
        self.window_width = self.maze_width + self.panel_width
        self.window_height = max(self.maze_height, 680)

        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Dynamic Maze RL Agent - Pygame GUI")

        self.font_main = pygame.font.SysFont("Arial", 16)
        self.font_title = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 13)

    def render(
        self,
        show_policy: bool = False,
        policy_dict: Optional[Dict[Tuple[int, int], int]] = None,
        algorithm_name: str = "Q-Learning",
        current_episode: int = 1,
        cumulative_reward: float = 0.0,
        epsilon_lambda: float = 0.0,
        success_rate: float = 0.0
    ) -> None:
        """Main render loop executing grid, overlays, and panel UI drawing."""
        self.screen.fill(self.COLOR_BG)

        # 1. Draw Maze Grid Tiles
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                tile_type = self.env.grid[r, c]

                color = self.COLOR_EMPTY
                if tile_type == MazeGenerator.WALL:
                    color = self.COLOR_WALL
                elif tile_type == MazeGenerator.PENALTY:
                    color = self.COLOR_PENALTY
                elif tile_type == MazeGenerator.START:
                    color = self.COLOR_START
                elif tile_type == MazeGenerator.KEY:
                    color = self.COLOR_EMPTY if self.env.has_key else self.COLOR_KEY
                elif tile_type == MazeGenerator.DOOR:
                    color = self.COLOR_DOOR_UNLOCKED if self.env.has_key else self.COLOR_DOOR_LOCKED
                elif tile_type == MazeGenerator.GOAL:
                    color = self.COLOR_GOAL

                pygame.draw.rect(self.screen, color, cell_rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, cell_rect, 1)

                # Draw Policy Arrows if enabled
                if show_policy and policy_dict and (r, c) in policy_dict and tile_type not in [MazeGenerator.WALL, MazeGenerator.GOAL]:
                    action = policy_dict[(r, c)]
                    self._draw_action_arrow(r, c, action)

        # 2. Draw Agent Circle
        agent_r, agent_c = self.env.agent_pos
        agent_center = (
            int(agent_c * self.cell_size + self.cell_size / 2),
            int(agent_r * self.cell_size + self.cell_size / 2)
        )
        pygame.draw.circle(self.screen, self.COLOR_AGENT, agent_center, int(self.cell_size * 0.35))

        # 3. Draw Side Control Panel and HUD
        self._draw_hud_panel(
            algorithm_name=algorithm_name,
            current_episode=current_episode,
            cumulative_reward=cumulative_reward,
            epsilon_lambda=epsilon_lambda,
            success_rate=success_rate
        )

        pygame.display.flip()

    def _draw_action_arrow(self, r: int, c: int, action: int) -> None:
        """Draws directional arrow representing policy choice on cell."""
        center_x = c * self.cell_size + self.cell_size // 2
        center_y = r * self.cell_size + self.cell_size // 2
        offset = self.cell_size // 4

        arrows = {
            0: (center_x, center_y - offset),  # UP
            1: (center_x, center_y + offset),  # DOWN
            2: (center_x - offset, center_y),  # LEFT
            3: (center_x + offset, center_y)   # RIGHT
        }

        end_pos = arrows.get(action, (center_x, center_y))
        pygame.draw.line(self.screen, (20, 20, 20), (center_x, center_y), end_pos, 2)

    def _draw_hud_panel(
        self,
        algorithm_name: str,
        current_episode: int,
        cumulative_reward: float,
        epsilon_lambda: float,
        success_rate: float
    ) -> None:
        """Renders information panel displaying state, energy bar, and controls."""
        panel_rect = pygame.Rect(self.maze_width, 0, self.panel_width, self.window_height)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, panel_rect)

        x_start = self.maze_width + 20
        y = 20

        # Title
        lbl_title = self.font_title.render("RL Control Panel", True, self.COLOR_TEXT)
        self.screen.blit(lbl_title, (x_start, y))
        y += 40

        # Status Information
        info_lines = [
            f"Algorithm: {algorithm_name}",
            f"Episode: {current_episode}",
            f"Step Count: {self.env.step_count} / {self.env.max_steps}",
            f"Reward: {cumulative_reward:.2f}",
            f"Epsilon / Lambda: {epsilon_lambda:.4f}",
            f"Recent Success Rate: {success_rate * 100:.1f}%",
            f"Key Collected: {'YES' if self.env.has_key else 'NO'}",
            f"Last Event: {self.env.event_logs[-1] if self.env.event_logs else 'None'}"
        ]

        for line in info_lines:
            txt = self.font_main.render(line, True, self.COLOR_TEXT)
            self.screen.blit(txt, (x_start, y))
            y += 28

        y += 10
        # Energy Gauge
        lbl_energy = self.font_main.render(f"Energy: {self.env.energy} / {self.env.max_energy}", True, self.COLOR_TEXT)
        self.screen.blit(lbl_energy, (x_start, y))
        y += 25

        energy_pct = max(0.0, self.env.energy / float(self.env.max_energy))
        bar_width = 280
        bar_rect_bg = pygame.Rect(x_start, y, bar_width, 16)
        bar_rect_fg = pygame.Rect(x_start, y, int(bar_width * energy_pct), 16)

        pygame.draw.rect(self.screen, (70, 70, 70), bar_rect_bg)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, bar_rect_fg)
        y += 40

        # Controls Instructions
        lbl_ctrl = self.font_title.render("Key Controls:", True, self.COLOR_TEXT)
        self.screen.blit(lbl_ctrl, (x_start, y))
        y += 30

        controls = [
            "[SPACE] : Play / Pause",
            "[R]     : Reset Environment",
            "[P]     : Toggle Policy Arrows",
            "[1]     : Select Q-Learning",
            "[2]     : Select SARSA(lambda)",
            "[3]     : Select Value Iteration",
            "[UP/DN] : Adjust Speed"
        ]

        for ctrl in controls:
            txt = self.font_small.render(ctrl, True, (180, 180, 180))
            self.screen.blit(txt, (x_start, y))
            y += 22

    def close(self) -> None:
        """Closes Pygame display window cleanly."""
        pygame.quit()