"""
gui/app.py
Interactive Pygame GUI Application managing event loops, agent execution, and interactive user toggles.
"""

import sys
import pygame
import numpy as np

from environments.generator import MazeGenerator
from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import ValueIterationAgent
from gui.renderer import MazeRenderer


class MazeGUIApp:
    """
    Main Pygame Application loop managing real-time rendering, agent interactions, and user keyboard controls.
    """

    def __init__(self, student_id: str = "40413854"):
        self.student_id = student_id

        # Initialize Environment
        self.generator = MazeGenerator(student_id=student_id)
        self.grid, self.positions = self.generator.generate_valid_maze()
        self.env = DynamicMazeEnv(self.grid, self.positions, max_energy=50)

        # Initialize Agents
        self.q_agent = QLearningAgent(alpha=0.1, gamma=0.99)
        self.sarsa_agent = SarsaLambdaAgent(alpha=0.1, gamma=0.99, lam=0.7)
        self.vi_agent = ValueIterationAgent(self.env, gamma=0.99)
        self.vi_agent.solve()

        self.current_agent_name = "Q-Learning"
        self.active_agent = self.q_agent

        # Renderer Setup
        self.renderer = MazeRenderer(self.env, cell_size=38)

        # Execution States
        self.running = True
        self.paused = True
        self.show_policy = False
        self.fps = 10
        self.clock = pygame.time.Clock()

        # Metrics Tracking
        self.current_episode = 1
        self.cumulative_reward = 0.0
        self.recent_successes = []

    def run(self) -> None:
        """App execution main loop."""
        state = self.env.reset()

        while self.running:
            self._handle_user_events()

            if not self.paused:
                # Determine Action based on selected active agent
                if self.current_agent_name == "Value Iteration":
                    action = self.vi_agent.get_action(state)
                elif self.current_agent_name == "Q-Learning":
                    action = self.q_agent.get_action(state)
                elif self.current_agent_name == "SARSA(lambda)":
                    action = self.sarsa_agent.get_action(state)
                else:
                    action = 0

                next_state, reward, done, info = self.env.step(action)
                self.cumulative_reward += reward

                # Agent Q-Table Update during live GUI run
                if self.current_agent_name == "Q-Learning":
                    self.q_agent.update(state, action, reward, next_state, done)
                    self.q_agent.decay_epsilon(self.current_episode, 500)
                elif self.current_agent_name == "SARSA(lambda)":
                    next_action = self.sarsa_agent.get_action(next_state)
                    self.sarsa_agent.update(state, action, reward, next_state, next_action, done)
                    self.sarsa_agent.decay_epsilon()

                state = next_state

                if done:
                    success = 1 if self.env.event_logs[-1] == "GOAL_REACHED" else 0
                    self.recent_successes.append(success)
                    if len(self.recent_successes) > 20:
                        self.recent_successes.pop(0)

                    # Reset for next episode
                    self.current_episode += 1
                    self.cumulative_reward = 0.0
                    state = self.env.reset()
                    if self.current_agent_name == "SARSA(lambda)":
                        self.sarsa_agent.reset_traces()

            # Render Screen Update
            success_rate = float(np.mean(self.recent_successes)) if self.recent_successes else 0.0
            epsilon_val = getattr(self.active_agent, "epsilon", 0.0)

            self.renderer.render(
                show_policy=self.show_policy,
                policy_dict=self.vi_agent.policy,
                algorithm_name=self.current_agent_name,
                current_episode=self.current_episode,
                cumulative_reward=self.cumulative_reward,
                epsilon_lambda=epsilon_val,
                success_rate=success_rate
            )

            self.clock.tick(self.fps)

        self.renderer.close()

    def _handle_user_events(self) -> None:
        """Processes Pygame keyboard inputs and UI interactions."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self.env.reset()
                    self.cumulative_reward = 0.0

                elif event.key == pygame.K_p:
                    self.show_policy = not self.show_policy

                elif event.key == pygame.K_1:
                    self.current_agent_name = "Q-Learning"
                    self.active_agent = self.q_agent

                elif event.key == pygame.K_2:
                    self.current_agent_name = "SARSA(lambda)"
                    self.active_agent = self.sarsa_agent

                elif event.key == pygame.K_3:
                    self.current_agent_name = "Value Iteration"
                    self.active_agent = self.vi_agent

                elif event.key == pygame.K_UP:
                    self.fps = min(60, self.fps + 5)

                elif event.key == pygame.K_DOWN:
                    self.fps = max(1, self.fps - 5)


if __name__ == "__main__":
    app = MazeGUIApp(student_id="40413854")
    app.run()