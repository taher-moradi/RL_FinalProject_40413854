"""
agents/value_iteration.py
Model-Based Value Iteration Algorithm implementation for Dynamic Maze MDP.
"""

import time
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from environments.maze import DynamicMazeEnv, State


class ValueIterationAgent:
    """
    Model-Based Value Iteration Agent.
    Computes optimal value function V*(s) and extracts optimal policy pi*(s)
    using complete environment transition dynamics P(s'|s,a) and R(s,a,s').
    """

    def __init__(
        self,
        env: DynamicMazeEnv,
        gamma: float = 0.99,
        theta: float = 1e-4
    ):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # State-Value function representation: Dict[State, float]
        self.V: Dict[State, float] = {}
        # Policy representation: Dict[State, int]
        self.policy: Dict[State, int] = {}

        self.all_states: List[State] = []
        self.iterations = 0
        self.execution_time = 0.0
        self.delta_history: List[float] = []

    def solve(self) -> Dict[str, Any]:
        """
        Runs Bellman Optimality Backup iterations until max delta < theta.
        """
        start_time = time.time()
        self.all_states = self.env.get_all_states()

        # Initialize Value Table
        for state in self.all_states:
            self.V[state] = 0.0

        self.iterations = 0
        self.delta_history = []

        while True:
            delta = 0.0
            new_V = self.V.copy()

            for state in self.all_states:
                r, c, key, energy = state
                
                # Terminal condition state check
                if energy <= 0 or (r, c) == self.env.goal_pos:
                    new_V[state] = 0.0
                    continue

                q_values = []
                for action in self.env.ACTIONS:
                    transitions = self.env.get_transition_prob(state, action)
                    q_val = 0.0
                    for prob, next_state, reward, done in transitions:
                        next_v = 0.0 if done else self.V.get(next_state, 0.0)
                        q_val += prob * (reward + self.gamma * next_v)
                    q_values.append(q_val)

                best_value = max(q_values)
                delta = max(delta, abs(best_value - self.V[state]))
                new_V[state] = best_value

            self.V = new_V
            self.iterations += 1
            self.delta_history.append(delta)

            # Convergence Check
            if delta < self.theta:
                break

        self.execution_time = time.time() - start_time
        self.extract_policy()

        return {
            "iterations": self.iterations,
            "execution_time": self.execution_time,
            "final_delta": self.delta_history[-1] if self.delta_history else 0.0,
            "gamma": self.gamma,
            "theta": self.theta
        }

    def extract_policy(self) -> Dict[State, int]:
        """
        Extracts greedy policy pi*(s) from computed optimal state values V*(s).
        """
        for state in self.all_states:
            r, c, key, energy = state
            if energy <= 0 or (r, c) == self.env.goal_pos:
                self.policy[state] = 0
                continue

            best_action = 0
            best_q_val = float("-inf")

            for action in self.env.ACTIONS:
                transitions = self.env.get_transition_prob(state, action)
                q_val = 0.0
                for prob, next_state, reward, done in transitions:
                    next_v = 0.0 if done else self.V.get(next_state, 0.0)
                    q_val += prob * (reward + self.gamma * next_v)

                if q_val > best_q_val:
                    best_q_val = q_val
                    best_action = action

            self.policy[state] = best_action

        return self.policy

    def get_action(self, state: State) -> int:
        """Returns optimal action for given state according to extracted policy."""
        return self.policy.get(state, 0)

    def get_value(self, state: State) -> float:
        """Returns optimal state value V*(s)."""
        return self.V.get(state, 0.0)


if __name__ == "__main__":
    from environments.generator import MazeGenerator

    gen = MazeGenerator(student_id="40413854")
    grid, pos = gen.generate_valid_maze()
    env = DynamicMazeEnv(grid, pos, max_energy=50)

    agent = ValueIterationAgent(env, gamma=0.99, theta=1e-4)
    results = agent.solve()

    print("Value Iteration Completed Successfully!")
    print(f"Iterations to Converge: {results['iterations']}")
    print(f"Execution Time: {results['execution_time']:.4f} seconds")