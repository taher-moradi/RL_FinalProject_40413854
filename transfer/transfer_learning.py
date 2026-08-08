"""
transfer/transfer_learning.py
Implementation of Transfer Learning scenarios for Q-Learning in Dynamic Maze environments.
"""

import os
import copy
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional

from environments.generator import MazeGenerator
from environments.maze import DynamicMazeEnv, State
from agents.q_learning import QLearningAgent


class TargetEnvironmentBuilder:
    """
    Builds Similar and Different Target environments from a Source maze layout.
    Validated using BFS to guarantee path feasibility.
    """

    def __init__(self, source_grid: np.ndarray, source_positions: Dict[str, Tuple[int, int]], student_id: str = "40413854"):
        self.source_grid = source_grid.copy()
        self.source_positions = copy.deepcopy(source_positions)
        self.generator = MazeGenerator(student_id=student_id)
        self.grid_size = source_grid.shape[0]

    def build_similar_target(self, shift_ratio: float = 0.18, seed: int = 101) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        """
        Creates Similar Target Environment:
        - 15% - 20% of obstacles are relocated.
        - Start, Key, Door, and Goal locations remain FIXED.
        """
        attempt_seed = seed
        while True:
            rng = np.random.default_rng(attempt_seed)
            grid = self.source_grid.copy()
            
            # Identify wall positions excluding boundaries
            wall_indices = []
            for r in range(1, self.grid_size - 1):
                for c in range(1, self.grid_size - 1):
                    if grid[r, c] == MazeGenerator.WALL:
                        wall_indices.append((r, c))

            num_to_move = int(len(wall_indices) * shift_ratio)
            rng.shuffle(wall_indices)
            moved_walls = wall_indices[:num_to_move]

            for pos in moved_walls:
                grid[pos] = MazeGenerator.EMPTY

            # Find new empty locations for relocated walls
            empty_indices = [
                (r, c) for r in range(1, self.grid_size - 1) for c in range(1, self.grid_size - 1)
                if grid[r, c] == MazeGenerator.EMPTY and (r, c) not in self.source_positions.values()
            ]
            rng.shuffle(empty_indices)

            for i in range(num_to_move):
                if empty_indices:
                    grid[empty_indices.pop()] = MazeGenerator.WALL

            # Validate path using BFS
            start = self.source_positions["start"]
            key = self.source_positions["key"]
            goal = self.source_positions["goal"]

            path1 = self.generator._bfs_search(grid, start, key, allow_door=False)
            path2 = self.generator._bfs_search(grid, key, goal, allow_door=True)

            if path1 is not None and path2 is not None:
                return grid, copy.deepcopy(self.source_positions)

            attempt_seed += 1

    def build_different_target(self, shift_ratio: float = 0.40, seed: int = 202) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        """
        Creates Different Target Environment:
        - At least 35% - 40% of obstacles are changed.
        - Key or Goal location is shifted.
        - New penalty tiles are added.
        """
        attempt_seed = seed
        while True:
            rng = np.random.default_rng(attempt_seed)
            grid = self.source_grid.copy()
            positions = copy.deepcopy(self.source_positions)

            # Move Key position to a new location
            empty_for_key = [
                (r, c) for r in range(2, self.grid_size - 2) for c in range(2, self.grid_size - 2)
                if grid[r, c] == MazeGenerator.EMPTY and (r, c) not in positions.values()
            ]
            if empty_for_key:
                new_key = empty_for_key[rng.integers(0, len(empty_for_key))]
                grid[positions["key"]] = MazeGenerator.EMPTY
                positions["key"] = new_key
                grid[new_key] = MazeGenerator.KEY

            # Shift obstacles
            wall_indices = [
                (r, c) for r in range(1, self.grid_size - 1) for c in range(1, self.grid_size - 1)
                if grid[r, c] == MazeGenerator.WALL
            ]
            num_to_move = int(len(wall_indices) * shift_ratio)
            rng.shuffle(wall_indices)

            for pos in wall_indices[:num_to_move]:
                grid[pos] = MazeGenerator.EMPTY

            empty_indices = [
                (r, c) for r in range(1, self.grid_size - 1) for c in range(1, self.grid_size - 1)
                if grid[r, c] == MazeGenerator.EMPTY and (r, c) not in positions.values()
            ]
            rng.shuffle(empty_indices)

            for _ in range(num_to_move):
                if empty_indices:
                    grid[empty_indices.pop()] = MazeGenerator.WALL

            # Add 3 new penalty tiles
            for _ in range(3):
                if empty_indices:
                    grid[empty_indices.pop()] = MazeGenerator.PENALTY

            # BFS Validation
            start, key, goal = positions["start"], positions["key"], positions["goal"]
            path1 = self.generator._bfs_search(grid, start, key, allow_door=False)
            path2 = self.generator._bfs_search(grid, key, goal, allow_door=True)

            if path1 is not None and path2 is not None:
                return grid, positions

            attempt_seed += 1


class TransferLearningManager:
    """
    Executes the 4 Transfer Learning Scenarios using Q-Learning.
    """

    def __init__(self, source_agent: QLearningAgent, target_env: DynamicMazeEnv, source_grid: np.ndarray, target_grid: np.ndarray):
        self.source_q_table = source_agent.q_table
        self.target_env = target_env
        self.source_grid = source_grid
        self.target_grid = target_grid

    def _is_local_neighborhood_equal(self, r: int, c: int) -> bool:
        """Checks if 3x3 local neighborhood around cell (r, c) is identical in both grids."""
        rows, cols = self.source_grid.shape
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if self.source_grid[nr, nc] != self.target_grid[nr, nc]:
                        return False
        return True

    def run_scenario(
        self,
        scenario_id: int,
        beta: float = 0.5,
        num_episodes: int = 150,
        alpha: float = 0.1,
        gamma: float = 0.99
    ) -> Tuple[QLearningAgent, pd.DataFrame]:
        """
        Executes specified Transfer Learning Scenario:
        1: Train from scratch (Zero Q-table)
        2: Full Transfer
        3: Scaled Transfer (beta * Q_source)
        4: Selective Transfer (Local 3x3 matching)
        """
        target_agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon_start=0.5, epsilon_min=0.01)

        # Q-Table Initialization based on scenario
        if scenario_id == 1:
            pass  # Scratch baseline

        elif scenario_id == 2:
            target_agent.q_table = copy.deepcopy(self.source_q_table)

        elif scenario_id == 3:
            for state, q_vals in self.source_q_table.items():
                target_agent.q_table[state] = beta * q_vals.copy()

        elif scenario_id == 4:
            for state, q_vals in self.source_q_table.items():
                r, c, k, e = state
                if self._is_local_neighborhood_equal(r, c):
                    target_agent.q_table[state] = q_vals.copy()

        logs = []
        for ep in range(1, num_episodes + 1):
            state = self.target_env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = target_agent.get_action(state)
                next_state, reward, done, info = self.target_env.step(action)
                target_agent.update(state, action, reward, next_state, done)

                total_reward += reward
                steps += 1
                state = next_state

            target_agent.decay_epsilon(ep, num_episodes)
            success = 1 if self.target_env.event_logs[-1] == "GOAL_REACHED" else 0

            logs.append({
                "episode": ep,
                "scenario": f"Scenario_{scenario_id}",
                "reward": total_reward,
                "steps": steps,
                "success": success,
                "epsilon": target_agent.epsilon
            })

        return target_agent, pd.DataFrame(logs)