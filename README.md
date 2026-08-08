# Dynamic Maze Reinforcement Learning Final Project

**Student ID:** 40413854  
**Base Seed:** 5  
**Grid Size:** 16x16  
**Dynamic Feature:** Limited Energy State Space `(row, col, key_collected, remaining_energy)`

---

## Project Overview

This repository contains a full Python implementation of a **Dynamic Maze Reinforcement Learning Agent** operating under a stochastic Markov Decision Process (MDP).

### Key Features
1. **Custom MDP Environment (`environments/maze.py`)**:
   - Probabilistic state transitions ($0.8$ intended action, $0.1$ perpendicular left, $0.1$ perpendicular right).
   - Dynamic Energy Constraint (Limited Energy incorporated into Markov state representation).
   - Sparse and Shaped reward options.
2. **Path Validation (`environments/generator.py`)**:
   - Automatic 16x16 maze generation validated using **BFS** for feasible paths (`Start -> Key -> Door -> Goal`).
3. **Core RL Algorithms**:
   - **Model-Based**: Value Iteration (`agents/value_iteration.py`).
   - **Model-Free Off-Policy**: Q-Learning with linear/exponential $\epsilon$-decay (`agents/q_learning.py`).
   - **Model-Free On-Policy**: SARSA($\lambda$) with Replacing Eligibility Traces (`agents/sarsa_lambda.py`).
4. **Transfer Learning Framework (`transfer/transfer_learning.py`)**:
   - Evaluates knowledge transfer across **Similar** (15-20% shift) and **Different** (35-40% shift) target maps across 4 scenarios.
5. **Interactive Pygame GUI (`gui/`)**:
   - Real-time step animation, HUD status, energy bar, policy arrows overlay, and algorithm toggles.

---

## Installation & Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/taher-moradi/RL_FinalProject_40413854.git
   cd RL_FinalProject_40413854