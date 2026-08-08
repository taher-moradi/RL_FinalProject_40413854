"""
experiments/run_experiments.py
Master Experiment Runner automating all training, benchmark comparisons, and transfer learning tests.
"""

import os
import json
import time
import pandas as pd
import numpy as np

from environments.generator import MazeGenerator
from environments.maze import DynamicMazeEnv
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from transfer.transfer_learning import TargetEnvironmentBuilder, TransferLearningManager


def ensure_directories():
    """Creates required output directories."""
    dirs = [
        "results/raw_data",
        "results/models",
        "results/figures",
        "results/configs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def run_value_iteration_experiments(env: DynamicMazeEnv) -> pd.DataFrame:
    """Executes Value Iteration sensitivity sweep across gamma values."""
    print("\n--- Running Value Iteration Experiments ---")
    results = []
    gammas = [0.90, 0.95, 0.99]

    for g in gammas:
        agent = ValueIterationAgent(env, gamma=g, theta=1e-4)
        metrics = agent.solve()
        results.append({
            "gamma": g,
            "iterations": metrics["iterations"],
            "execution_time": metrics["execution_time"],
            "final_delta": metrics["final_delta"]
        })
        print(f"Gamma: {g} | Iterations: {metrics['iterations']} | Time: {metrics['execution_time']:.4f}s")

    df = pd.DataFrame(results)
    df.to_csv("results/raw_data/value_iteration_results.csv", index=False)
    return df


def run_q_learning_experiments(env: DynamicMazeEnv, episodes: int = 300) -> Tuple[QLearningAgent, pd.DataFrame]:
    """Executes Q-Learning training with decay schemes and saves results."""
    print("\n--- Running Q-Learning Experiments ---")
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01)

    logs = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        wall_collisions = 0
        penalty_entries = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            steps += 1
            if info["event"] == "WALL_COLLISION":
                wall_collisions += 1
            elif info["event"] == "PENALTY_STEP":
                penalty_entries += 1

            state = next_state

        agent.decay_epsilon(ep, episodes)
        success = 1 if env.event_logs[-1] == "GOAL_REACHED" else 0

        logs.append({
            "episode": ep,
            "reward": total_reward,
            "steps": steps,
            "success": success,
            "wall_collisions": wall_collisions,
            "penalty_entries": penalty_entries,
            "epsilon": agent.epsilon
        })

    df = pd.DataFrame(logs)
    df.to_csv("results/raw_data/q_learning_results.csv", index=False)
    agent.save_q_table("results/models/q_learning_source.pkl")
    return agent, df


def run_sarsa_lambda_experiments(env: DynamicMazeEnv, episodes: int = 300) -> pd.DataFrame:
    """Executes SARSA(lambda) sweep over lambda values [0.0, 0.3, 0.7, 0.9]."""
    print("\n--- Running SARSA(lambda) Experiments ---")
    lambdas = [0.0, 0.3, 0.7, 0.9]
    all_logs = []

    for lam in lambdas:
        agent = SarsaLambdaAgent(alpha=0.1, gamma=0.99, lam=lam, trace_type="replacing")
        
        for ep in range(1, episodes + 1):
            state = env.reset()
            agent.reset_traces()
            action = agent.get_action(state)

            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                next_state, reward, done, info = env.step(action)
                next_action = agent.get_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)

                total_reward += reward
                steps += 1
                state, action = next_state, next_action

            agent.decay_epsilon()
            success = 1 if env.event_logs[-1] == "GOAL_REACHED" else 0

            all_logs.append({
                "lambda": lam,
                "episode": ep,
                "reward": total_reward,
                "steps": steps,
                "success": success
            })

    df = pd.DataFrame(all_logs)
    df.to_csv("results/raw_data/sarsa_lambda_results.csv", index=False)
    return df


def run_transfer_experiments(source_grid: np.ndarray, source_pos: dict, source_agent: QLearningAgent):
    """Executes Transfer Learning across Similar and Different Target Maps."""
    print("\n--- Running Transfer Learning Experiments ---")
    builder = TargetEnvironmentBuilder(source_grid, source_pos, student_id="40413854")

    # Build Target Maps
    sim_grid, sim_pos = builder.build_similar_target()
    diff_grid, diff_pos = builder.build_different_target()

    sim_env = DynamicMazeEnv(sim_grid, sim_pos, max_energy=50)
    diff_env = DynamicMazeEnv(diff_grid, diff_pos, max_energy=50)

    # Manager for Similar Map
    mgr_sim = TransferLearningManager(source_agent, sim_env, source_grid, sim_grid)
    sim_dfs = []
    for sc in [1, 2, 3, 4]:
        _, df = mgr_sim.run_scenario(scenario_id=sc, num_episodes=100)
        df["target_type"] = "Similar"
        sim_dfs.append(df)

    # Manager for Different Map
    mgr_diff = TransferLearningManager(source_agent, diff_env, source_grid, diff_grid)
    diff_dfs = []
    for sc in [1, 2, 3, 4]:
        _, df = mgr_diff.run_scenario(scenario_id=sc, num_episodes=100)
        df["target_type"] = "Different"
        diff_dfs.append(df)

    final_df = pd.concat(sim_dfs + diff_dfs, ignore_index=True)
    final_df.to_csv("results/raw_data/transfer_learning_results.csv", index=False)
    print("Transfer Learning Experiments Completed Successfully!")


def main():
    ensure_directories()
    print("Initializing Maze Generator...")
    gen = MazeGenerator(student_id="40413854")
    grid, pos = gen.generate_valid_maze(save_path="environments/maps/maze_40413854.json")
    env = DynamicMazeEnv(grid, pos, max_energy=50, reward_mode="sparse")

    # Save Configuration
    config_data = {
        "student_id": "40413854",
        "base_seed": 5,
        "grid_size": 16,
        "max_energy": 50,
        "reward_mode": "sparse"
    }
    with open("results/configs/experiment_config.json", "w") as f:
        json.dump(config_data, f, indent=4)

    run_value_iteration_experiments(env)
    q_agent, _ = run_q_learning_experiments(env, episodes=250)
    run_sarsa_lambda_experiments(env, episodes=250)
    run_transfer_experiments(grid, pos, q_agent)


if __name__ == "__main__":
    main()