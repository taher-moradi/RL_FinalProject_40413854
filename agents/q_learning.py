"""
agents/q_learning.py
Model-Free Off-Policy Q-Learning Algorithm implementation.
"""

import pickle
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from environments.maze import State


class QLearningAgent:
    """
    Model-Free Off-Policy Q-Learning Agent with tabular Q-function representation.
    """

    def __init__(
        self,
        actions: List[int] = [0, 1, 2, 3],
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        decay_type: str = "exponential"
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type

        # Q-Table representation: Dict[State, np.ndarray] where array shape is (num_actions,)
        self.q_table: Dict[State, np.ndarray] = {}
        self.rng = np.random.default_rng(42)

    def get_q_values(self, state: State) -> np.ndarray:
        """Retrieves Q-value array for a state, initializing to zero if unseen."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions), dtype=float)
        return self.q_table[state]

    def get_action(self, state: State, greedy: bool = False) -> int:
        """
        Selects action using epsilon-greedy exploration strategy.
        If greedy is True, selects purely greedy action.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(self.actions))

        q_vals = self.get_q_values(state)
        max_q = np.max(q_vals)
        # Random tie-breaking among best actions
        best_actions = np.where(q_vals == max_q)[0]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool
    ) -> float:
        """
        Performs Q-Learning Off-Policy Temporal Difference (TD) Update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        current_q = self.get_q_values(state)[action]
        next_max_q = 0.0 if done else np.max(self.get_q_values(next_state))

        td_target = reward + self.gamma * next_max_q
        td_error = td_target - current_q

        self.q_table[state][action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self, current_episode: int, total_episodes: int) -> float:
        """Decays epsilon value according to configured decay scheme."""
        if self.decay_type == "exponential":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif self.decay_type == "linear":
            decay_step = (self.epsilon_start - self.epsilon_min) / float(total_episodes)
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_step)
        return self.epsilon

    def save_q_table(self, filepath: str) -> None:
        """Saves learned Q-table dictionary to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filepath: str) -> None:
        """Loads Q-table dictionary from disk."""
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)


if __name__ == "__main__":
    from environments.generator import MazeGenerator
    from environments.maze import DynamicMazeEnv

    gen = MazeGenerator(student_id="40413854")
    grid, pos = gen.generate_valid_maze()
    env = DynamicMazeEnv(grid, pos, max_energy=50)

    agent = QLearningAgent(alpha=0.1, gamma=0.99)
    state = env.reset()

    for _ in range(10):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    print("Q-Learning Test Iteration Passed Successfully!")