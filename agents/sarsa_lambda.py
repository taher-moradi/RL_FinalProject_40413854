"""
agents/sarsa_lambda.py
Model-Free On-Policy SARSA(lambda) with Eligibility Traces implementation.
"""

import pickle
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from environments.maze import State


class SarsaLambdaAgent:
    """
    Model-Free On-Policy SARSA(lambda) Agent using Eligibility Traces.
    Supports 'replacing' and 'accumulating' trace update schemes.
    """

    def __init__(
        self,
        actions: List[int] = [0, 1, 2, 3],
        alpha: float = 0.1,
        gamma: float = 0.99,
        lam: float = 0.7,
        trace_type: str = "replacing",
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.trace_type = trace_type

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-Table: Dict[State, np.ndarray]
        self.q_table: Dict[State, np.ndarray] = {}
        # Eligibility Trace Table: Dict[State, np.ndarray]
        self.e_table: Dict[State, np.ndarray] = {}

        self.rng = np.random.default_rng(42)

    def reset_traces(self) -> None:
        """Clears eligibility traces at the beginning of each episode."""
        self.e_table.clear()

    def get_q_values(self, state: State) -> np.ndarray:
        """Retrieves Q-value array for state."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions), dtype=float)
        return self.q_table[state]

    def get_e_values(self, state: State) -> np.ndarray:
        """Retrieves Eligibility Trace array for state."""
        if state not in self.e_table:
            self.e_table[state] = np.zeros(len(self.actions), dtype=float)
        return self.e_table[state]

    def get_action(self, state: State, greedy: bool = False) -> int:
        """Selects action using epsilon-greedy behavior policy."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(self.actions))

        q_vals = self.get_q_values(state)
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int,
        done: bool
    ) -> Tuple[float, float]:
        """
        Performs SARSA(lambda) On-Policy update with Eligibility Traces:
        delta_t = r_{t+1} + gamma * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
        E(s, a) update according to trace_type (replacing or accumulating)
        Q(s, a) <- Q(s, a) + alpha * delta_t * E(s, a)
        E(s, a) <- gamma * lambda * E(s, a)
        """
        current_q = self.get_q_values(state)[action]
        next_q = 0.0 if done else self.get_q_values(next_state)[next_action]

        # 1. Compute TD Error
        td_error = reward + self.gamma * next_q - current_q

        # 2. Update Eligibility Trace for current state-action pair
        e_curr = self.get_e_values(state)
        if self.trace_type == "replacing":
            e_curr[action] = 1.0
        elif self.trace_type == "accumulating":
            e_curr[action] += 1.0
        else:
            raise ValueError(f"Unknown trace_type: {self.trace_type}")

        # 3. Update Q-values and decay eligibility traces across visited states
        visited_states = list(self.e_table.keys())
        current_trace_value = e_curr[action]

        for s in visited_states:
            e_arr = self.e_table[s]
            q_arr = self.get_q_values(s)

            # Update Q-value: Q(s, a) <- Q(s, a) + alpha * delta * E(s, a)
            q_arr += self.alpha * td_error * e_arr

            # Decay trace: E(s, a) <- gamma * lambda * E(s, a)
            e_arr *= (self.gamma * self.lam)

            # Prune negligible trace values to conserve memory
            e_arr[e_arr < 1e-5] = 0.0

        return float(td_error), float(current_trace_value)

    def decay_epsilon(self) -> float:
        """Decays epsilon exponentially."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
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

    agent = SarsaLambdaAgent(alpha=0.1, gamma=0.99, lam=0.7, trace_type="replacing")
    state = env.reset()
    agent.reset_traces()
    action = agent.get_action(state)

    for _ in range(10):
        next_state, reward, done, _ = env.step(action)
        next_action = agent.get_action(next_state)
        delta, trace = agent.update(state, action, reward, next_state, next_action, done)
        state, action = next_state, next_action
        if done:
            break

    print("SARSA(lambda) Test Iteration Passed Successfully!")