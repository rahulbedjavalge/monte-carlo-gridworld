# Monte Carlo RL Agent for 4x4 GridWorld

## Overview
This project implements a First-Visit Monte Carlo Reinforcement Learning agent to solve a 4x4 GridWorld environment. The agent learns state values using a random policy and visualizes the learning process.

## Features
- **GridWorld Environment:** 4x4 grid, start at (0,0), goal at (3,3), -1 reward per step, 0 at goal.
- **Agent:** First-Visit Monte Carlo, random policy, 10,000 episodes, saves V(s) and rewards every 1,000 episodes.
- **Visualization:**
  - Heatmaps of V(s) at intervals
  - Animated GIF of value function evolution
  - Animated GIF of agent moving from start to goal
  - Reward curve over episodes

## Directory Structure
- `env/gridworld.py`: GridWorld environment
- `agent/monte_carlo.py`: Monte Carlo agent
- `agent/agent_path_demo.py`: Script to generate agent path video
- `visualize.py`: Visualization utilities
- `main.py`: Main training and visualization script
- `results/`: Output images and videos

## How to Run
1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Train and visualize:
   ```powershell
   python main.py
   ```
3. Generate agent path video:
   ```powershell
   python agent/agent_path_demo.py
   ```

## Outputs
- `results/V_*.png`: Value function heatmaps
- `results/V_evolution.gif`: Value function learning animation
- `results/agent_path.gif`: Agent movement animation
- `results/rewards.png`: Reward curve

## Notes
- All code runs on CPU; no GPU required.
- For faster runs, reduce the number of episodes in `main.py`.