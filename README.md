# Dynamic Maze Reinforcement Learning Project

**Student ID:** 40413854  
**Base Seed:** 5  
**Grid Dimensions:** 16x16 (`15 + (5 % 4) = 16`)  
**Dynamic Feature:** Limited Energy (`s = (row, col, key_status, remaining_energy)`)  
**GUI Framework:** Pygame  
**Repository Access:** Public  

---

## 1. Project Overview & Problem Formulation

This repository contains a complete, ground-up implementation of a **Dynamic Maze Reinforcement Learning Agent** operating within a stochastic Markov Decision Process (MDP).

### Key Technical Highlights:
- **Markov Property Preservation**: State $s = (r, c, k, e)$ incorporates agent spatial coordinates $(r, c)$, key acquisition status $k \in \{0, 1\}$, and remaining energy $e \in [0, E_{max}]$.
- **Stochastic Transition Model**: Executed action occurs with $0.8$ probability; perpendicular drift occurs with $0.1$ probability for left and right each. Boundary/Wall collisions penalize the agent while keeping it stationary.
- **Pure NumPy Implementation**: All core algorithms (Value Iteration, Q-Learning, SARSA($\lambda$)) are built from scratch without external RL libraries (such as Stable-Baselines3 or RLlib).

---

## 2. Installation & Prerequisites

### System Requirements
- Python `3.10` or higher
- `pip` package manager

### Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/taher-moradi/RL_FinalProject_40413854.git
   cd RL_FinalProject_40413854
   ```

2. **Create and Activate Virtual Environment (Recommended):**

   Linux/macOS:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   Windows (PowerShell):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. How to Run the Project (Execution Guide)

The main entry point is `main.py`, providing simple Command Line Arguments (CLI) to launch the Pygame GUI or run large-scale experiment benchmarks.

### A. Launch Interactive Pygame GUI

To launch the interactive graphical application with live step animations and control panels:
```bash
python main.py --mode gui
```

Or directly run the GUI application script:
```bash
python gui/app.py
```

**GUI Keyboard Controls:**

| Key | Action |
|---|---|
| `[SPACE]` | Play / Pause Agent execution |
| `[R]` | Reset Environment to initial state |
| `[P]` | Toggle Optimal Policy Arrows overlay |
| `[1]` | Select Q-Learning Agent |
| `[2]` | Select SARSA(λ) Agent |
| `[3]` | Select Value Iteration Agent |
| `[UP ARROW]` | Increase Animation Speed (FPS) |
| `[DOWN ARROW]` | Decrease Animation Speed (FPS) |

### B. Run Full Benchmark Experiments & Reproduce Results

To run the full suite of experiments (Value Iteration sweep, Q-Learning training, SARSA(λ) λ-sweeps, and Transfer Learning scenarios):
```bash
python main.py --mode experiment
```

Or run individual experiment scripts directly:
```bash
python experiments/run_experiments.py
```

This script will automatically:
1. Generate and validate the 16×16 maze layout using BFS.
2. Execute Value Iteration across γ ∈ [0.90, 0.95, 0.99].
3. Train Q-Learning agent with ε-decay and record step-by-step logs.
4. Train SARSA(λ) across λ ∈ [0.0, 0.3, 0.7, 0.9].
5. Build Similar and Different Target Maps and execute the 4 Transfer Learning Scenarios.
6. Save raw CSV metrics under `results/raw_data/` and trained models under `results/models/`.

### C. Generate Analysis Figures & Visual Analytics

To process raw data CSVs and generate publication-quality figures:
```bash
python experiments/analysis.py
```

Generated plots will be saved directly into `results/figures/`:
- `learning_curves.png`: Training performance curves.
- `value_heatmap.png`: V*(s) State-Value Heatmap.
- `transfer_learning_comparison.png`: Evaluation of Transfer Scenarios across target maps.

### D. Execute Unit Tests

To verify environment dynamics, BFS path validity, and energy depletion rules:
```bash
pytest tests/test_env.py -v
```

---

## 4. Directory Structure

```
RL_FinalProject_40413854/
├── environments/
│   ├── maze.py              # MDP Environment implementation
│   ├── generator.py         # Maze generation and BFS path validation
│   └── maps/                # JSON maze map storage
├── agents/
│   ├── value_iteration.py   # Model-Based Value Iteration
│   ├── q_learning.py        # Model-Free Off-Policy Q-Learning
│   └── sarsa_lambda.py      # Model-Free On-Policy SARSA(lambda)
├── transfer/
│   └── transfer_learning.py # Target environments and 4 Transfer scenarios
├── gui/
│   ├── app.py               # Pygame main application loop
│   └── renderer.py          # Pygame screen renderer and panel HUD
├── experiments/
│   ├── run_experiments.py   # Automated benchmark launcher
│   ├── analysis.py          # Plotting and visual analytics generator
│   └── configs/             # Experiment configuration JSONs
├── results/
│   ├── raw_data/            # Raw metric CSV output files
│   ├── models/              # Saved Q-table models (.pkl)
│   ├── figures/             # Output PNG figures and plots
│   └── videos/              # Optional episode recordings
├── tests/
│   └── test_env.py          # Pytest unit tests
├── requirements.txt         # Package dependencies
├── README.md                # Project documentation and reproduction guide
└── main.py                  # Master entry point CLI
```

---

## 5. Experiment Hyperparameters Summary

| Parameter | Value | Description |
|---|---|---|
| Grid Size | 16×16 | Determined via base seed 5 |
| Max Energy | 50 | Energy step limit before episode termination |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Alpha (α) | 0.10 | Learning rate for TD updates |
| Epsilon (ε) | 1.0 → 0.01 | Exploration rate with exponential decay |
| Lambda (λ) | [0.0, 0.3, 0.7, 0.9] | Eligibility trace decay parameter |
| Trace Type | Replacing | Replacing eligibility trace update rule |

---

## 6. AI Tools Citation & Assistance Transparency Table

As required by project instructions (Page 10), the table below details AI assistant usage during code architecture setup and debugging:

| Use Case / Feature | AI Initial Proposal | Human Modifications & Fixes | Reason for Correction |
|---|---|---|---|
| State Representation | Suggested simple 2D coordinates (x, y). | Expanded state to (row, col, key_status, energy). | Simple (x, y) violates Markov Property due to key requirement and limited energy constraints. |
| SARSA(λ) Traces | Accumulating trace update E(s, a) += 1. | Replaced with Replacing Trace E(s, a) = 1 for visited state-action. | Accumulating traces caused trace inflation and Q-value explosion in loopy grid states. |
| Value Iteration Backup | In-place update using single value array. | Modified to double-buffered V and new_V sync updates. | In-place updates introduce asynchronous Bellman update bias during iteration convergence. |

---

## 7. License & Academic Integrity

This project is submitted for the Final Reinforcement Learning Course Project. All environment dynamics, algorithm implementations, and analytical code were developed independently in accordance with course academic integrity standards.
