"""
experiments/analysis.py
Data Analysis and Visual Analytics generation script.
Generates learning curves, heatmaps, policy grids, and transfer metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

from environments.generator import MazeGenerator
from environments.maze import DynamicMazeEnv
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent


def setup_plot_style():
    """Sets clean Matplotlib plotting layout."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 1.0


def generate_learning_curves():
    """Plots and saves Q-Learning and SARSA(lambda) learning curves."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Q-Learning Rewards
    if os.path.exists("results/raw_data/q_learning_results.csv"):
        df_q = pd.read_csv("results/raw_data/q_learning_results.csv")
        axes[0].plot(df_q["episode"], df_q["reward"], color="#1f77b4", alpha=0.4, label="Raw Episode Reward")
        axes[0].plot(df_q["episode"], df_q["reward"].rolling(15).mean(), color="#0055ff", linewidth=2.0, label="15-Ep Moving Avg")
        axes[0].set_title("Q-Learning Reward Curve", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total Reward")
        axes[0].legend()

    # SARSA(lambda) Comparison
    if os.path.exists("results/raw_data/sarsa_lambda_results.csv"):
        df_sarsa = pd.read_csv("results/raw_data/sarsa_lambda_results.csv")
        for lam, group in df_sarsa.groupby("lambda"):
            rolling_reward = group["reward"].rolling(15).mean()
            axes[1].plot(group["episode"], rolling_reward, label=f"λ = {lam}", linewidth=1.8)
        axes[1].set_title("SARSA(λ) Learning Speed across λ", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("15-Ep Moving Avg Reward")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/figures/learning_curves.png", dpi=300)
    plt.close()
    print("Saved: results/figures/learning_curves.png")


def generate_value_heatmap(env: DynamicMazeEnv, agent: ValueIterationAgent):
    """Plots and saves Value Function Heatmap for Value Iteration."""
    setup_plot_style()
    grid_size = env.rows
    value_grid = np.zeros((grid_size, grid_size))

    for r in range(grid_size):
        for c in range(grid_size):
            if env.grid[r, c] == MazeGenerator.WALL:
                value_grid[r, c] = np.nan
            else:
                # Max value across energy levels for key=0
                vals = [agent.get_value((r, c, 0, e)) for e in range(env.max_energy + 1)]
                value_grid[r, c] = max(vals) if vals else 0.0

    plt.figure(figsize=(8, 7))
    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="#222222")  # Dark grey walls

    im = plt.imshow(value_grid, cmap=cmap, origin="upper")
    plt.colorbar(im, label="State Value V*(s)")
    plt.title("Value Iteration - State-Value Heatmap V*(s)", fontsize=13, fontweight="bold")
    plt.xlabel("Grid Column")
    plt.ylabel("Grid Row")

    plt.tight_layout()
    plt.savefig("results/figures/value_heatmap.png", dpi=300)
    plt.close()
    print("Saved: results/figures/value_heatmap.png")


def generate_transfer_comparison_plot():
    """Plots learning curves for Transfer Learning Scenarios."""
    setup_plot_style()
    if not os.path.exists("results/raw_data/transfer_learning_results.csv"):
        return

    df = pd.read_csv("results/raw_data/transfer_learning_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for idx, target_type in enumerate(["Similar", "Different"]):
        ax = axes[idx]
        sub_df = df[df["target_type"] == target_type]

        for sc, group in sub_df.groupby("scenario"):
            rolling = group["reward"].rolling(10).mean()
            ax.plot(group["episode"], rolling, label=sc.replace("_", " "), linewidth=1.8)

        ax.set_title(f"Transfer Scenarios on {target_type} Target Map", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("10-Ep Moving Avg Reward")
        ax.legend()

    plt.tight_layout()
    plt.savefig("results/figures/transfer_learning_comparison.png", dpi=300)
    plt.close()
    print("Saved: results/figures/transfer_learning_comparison.png")


def main():
    print("\n--- Generating Visual Analytics and Figures ---")
    gen = MazeGenerator(student_id="40123454")
    grid, pos = gen.generate_valid_maze()
    env = DynamicMazeEnv(grid, pos, max_energy=50)

    vi_agent = ValueIterationAgent(env, gamma=0.99)
    vi_agent.solve()

    generate_learning_curves()
    generate_value_heatmap(env, vi_agent)
    generate_transfer_comparison_plot()
    print("All Analytics Generated Successfully!")


if __name__ == "__main__":
    main()