"""
environments/maze.py
Full Implementation of Dynamic Maze Environment as a Markov Decision Process (MDP).
"""

import numpy as np
from typing import Tuple, Dict, List, Any, Optional

# State type definition: (row, col, key_status, remaining_energy)
State = Tuple[int, int, int, int]


class DynamicMazeEnv:
    """
    Dynamic Maze Environment preserving Markov Property.
    State Space: (r, c, key_collected, remaining_energy)
    Action Space: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTIONS = [UP, DOWN, LEFT, RIGHT]
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    # Grid tile codes
    EMPTY = 0
    WALL = 1
    PENALTY = 2
    START = 3
    KEY = 4
    DOOR = 5
    GOAL = 6

    def __init__(
        self,
        grid: np.ndarray,
        positions: Dict[str, Tuple[int, int]],
        max_energy: int = 60,
        reward_mode: str = "sparse",
        seed: Optional[int] = 42
    ):
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape
        self.positions = positions

        self.start_pos = positions["start"]
        self.key_pos = positions["key"]
        self.door_pos = positions["door"]
        self.goal_pos = positions["goal"]

        self.max_energy = max_energy
        self.reward_mode = reward_mode
        self.rng = np.random.default_rng(seed)

        # Current state components
        self.agent_pos = self.start_pos
        self.has_key = 0
        self.energy = self.max_energy

        # Tracking logs and episode steps
        self.event_logs: List[str] = []
        self.step_count = 0
        self.max_steps = 3 * (self.rows * self.cols)  # Episode step cap based on spec

    def reset(self, seed: Optional[int] = None) -> State:
        """Resets the environment to initial configuration."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = self.start_pos
        self.has_key = 0
        self.energy = self.max_energy
        self.step_count = 0
        self.event_logs = ["RESET_EPISODE"]

        return self.get_state()

    def get_state(self) -> State:
        """Returns current Markovian state representation."""
        return (self.agent_pos[0], self.agent_pos[1], self.has_key, self.energy)

    def _get_perpendicular_actions(self, action: int) -> Tuple[int, int]:
        """Returns perpendicular directions to model stochastic transitions."""
        if action in [self.UP, self.DOWN]:
            return self.LEFT, self.RIGHT
        else:
            return self.UP, self.DOWN

    def _get_next_position(self, curr_pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Calculates next candidate position based purely on movement direction."""
        r, c = curr_pos
        if action == self.UP:
            nr, nc = r - 1, c
        elif action == self.DOWN:
            nr, nc = r + 1, c
        elif action == self.LEFT:
            nr, nc = r, c - 1
        elif action == self.RIGHT:
            nr, nc = r, c + 1
        else:
            nr, nc = r, c

        # Check boundary limits
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return (nr, nc)
        return curr_pos  # Wall collision at grid outer boundary

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Executes environment transition step with stochastic dynamics and reward computation.
        Transition probabilities: 0.8 chosen action, 0.1 left perpendicular, 0.1 right perpendicular.
        """
        self.step_count += 1

        # 1. Determine actual action execution based on transition probabilities
        perp1, perp2 = self._get_perpendicular_actions(action)
        actual_action = int(self.rng.choice([action, perp1, perp2], p=[0.8, 0.1, 0.1]))

        # 2. Decrement remaining energy
        self.energy -= 1

        # 3. Calculate candidate next position
        intended_next_pos = self._get_next_position(self.agent_pos, actual_action)
        cell_type = self.grid[intended_next_pos]

        # Check wall collision or locked door interaction
        is_wall_collision = (cell_type == self.WALL)
        is_door_locked = (cell_type == self.DOOR and self.has_key == 0)

        event = "NORMAL_MOVE"

        if is_wall_collision or intended_next_pos == self.agent_pos:
            next_pos = self.agent_pos
            event = "WALL_COLLISION"
        elif is_door_locked:
            next_pos = self.agent_pos
            event = "DOOR_LOCKED_ATTEMPT"
        else:
            next_pos = intended_next_pos
            if next_pos == self.key_pos and self.has_key == 0:
                self.has_key = 1
                event = "KEY_PICKUP"
            elif next_pos == self.door_pos and self.has_key == 1:
                event = "DOOR_PASSED"
            elif next_pos == self.goal_pos:
                event = "GOAL_REACHED"
            elif cell_type == self.PENALTY:
                event = "PENALTY_STEP"

        prev_pos = self.agent_pos
        self.agent_pos = next_pos

        # 4. Compute step reward
        reward = self._calculate_reward(prev_pos, next_pos, event)

        # 5. Evaluate termination conditions
        done = False
        if event == "GOAL_REACHED":
            done = True
        elif self.energy <= 0:
            done = True
            event = "ENERGY_DEPLETED"
        elif self.step_count >= self.max_steps:
            done = True
            event = "MAX_STEPS_REACHED"

        self.event_logs.append(event)

        info = {
            "actual_action": actual_action,
            "event": event,
            "step": self.step_count,
            "energy": self.energy
        }

        return self.get_state(), reward, done, info

    def _calculate_reward(self, prev_pos: Tuple[int, int], next_pos: Tuple[int, int], event: str) -> float:
        """Calculates step reward for Sparse or Shaped reward functions."""
        if self.reward_mode == "sparse":
            if event == "GOAL_REACHED":
                return 100.0
            elif event == "KEY_PICKUP":
                return 15.0
            elif event == "PENALTY_STEP":
                return -5.0
            elif event in ["WALL_COLLISION", "DOOR_LOCKED_ATTEMPT"]:
                return -1.0
            elif event == "ENERGY_DEPLETED":
                return -20.0
            else:
                return -0.1  # Step penalty

        elif self.reward_mode == "shaped":
            reward = -0.1  # Base step penalty

            if event == "GOAL_REACHED":
                return reward + 100.0
            elif event == "KEY_PICKUP":
                reward += 20.0
            elif event == "PENALTY_STEP":
                reward -= 8.0
            elif event in ["WALL_COLLISION", "DOOR_LOCKED_ATTEMPT"]:
                reward -= 2.0
            elif event == "ENERGY_DEPLETED":
                reward -= 25.0

            # Reward Shaping based on Manhattan Distance towards active milestone
            target = self.key_pos if self.has_key == 0 else self.goal_pos

            prev_dist = abs(prev_pos[0] - target[0]) + abs(prev_pos[1] - target[1])
            curr_dist = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])

            shaping_delta = (prev_dist - curr_dist) * 0.5
            reward += shaping_delta

            return reward

        else:
            raise ValueError(f"Invalid reward mode: {self.reward_mode}")

    def get_transition_prob(self, state: State, action: int) -> List[Tuple[float, State, float, bool]]:
        """
        Full transition model P(s' | s, a) and R(s, a, s') for Value Iteration.
        Returns list of tuples: (probability, next_state, reward, is_terminal)
        """
        r, c, key, energy = state

        if energy <= 0 or (r, c) == self.goal_pos:
            return [(1.0, state, 0.0, True)]

        transitions = []
        perp1, perp2 = self._get_perpendicular_actions(action)
        possible_actions = [(action, 0.8), (perp1, 0.1), (perp2, 0.1)]

        for act, prob in possible_actions:
            next_energy = energy - 1
            intended_pos = self._get_next_position((r, c), act)
            cell_type = self.grid[intended_pos]

            is_wall = (cell_type == self.WALL)
            is_locked = (cell_type == self.DOOR and key == 0)

            if is_wall or intended_pos == (r, c) or is_locked:
                next_r, next_c = r, c
                event = "WALL_COLLISION" if is_wall else ("DOOR_LOCKED_ATTEMPT" if is_locked else "STAY")
            else:
                next_r, next_c = intended_pos
                if intended_pos == self.key_pos and key == 0:
                    event = "KEY_PICKUP"
                elif intended_pos == self.goal_pos:
                    event = "GOAL_REACHED"
                elif cell_type == self.PENALTY:
                    event = "PENALTY_STEP"
                else:
                    event = "NORMAL_MOVE"

            next_key = 1 if (key == 1 or (next_r, next_c) == self.key_pos) else 0
            dummy_reward = self._calculate_reward((r, c), (next_r, next_c), event)

            done = False
            if (next_r, next_c) == self.goal_pos or next_energy <= 0:
                done = True

            next_state = (next_r, next_c, next_key, next_energy)
            transitions.append((prob, next_state, dummy_reward, done))

        return transitions

    def get_all_states(self) -> List[State]:
        """Generates all reachable states within the Markov state space."""
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == self.WALL:
                    continue
                for key in [0, 1]:
                    for e in range(self.max_energy + 1):
                        states.append((r, c, key, e))
        return states