# Tic-Tac-Toe Reinforcement Learning Project

**Course:** AAI 551 - Engineering Programming: Python
**Semester:** Fall 2025
**Team Members:**
- Jinwen Wu (jwu74@stevens.edu)
- Samantha Ramcharran (sramchar@stevens.edu)

---

## Problem Description

We implemented and compared 5 different reinforcement learning algorithms to play Tic-Tac-Toe:

1. **Q-Learning** - Table-based learning
2. **SARSA** - On-policy learning
3. **DQN** - Neural network Q-learning
4. **REINFORCE** - Policy gradient method
5. **MCTS** - Monte Carlo Tree Search

The goal is to train AI agents to play Tic-Tac-Toe well and compare which method works best.

---

## Algorithm Overview

### 1. Q-Learning (Off-Policy Temporal Difference)
Q-Learning is a value-based method that learns optimal Q-values Q(s,a) using a table. It will update using the maximum Q-value of the next state, making it an off-policy algorithm.

**Key Features:**
- Stores Q-values in a dictionary (state-action table)
- Updates every step using: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
- Simple and effective for small state spaces

### 2. SARSA (On-Policy Temporal Difference)
SARSA is similar to Q-Learning but learns the policy it's actually following. It uses the Q-value of the action actually taken in the next state.

**Key Features:**
- On-policy learning (learns current policy, not optimal)
- Updates using: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
- More conservative than Q-Learning

### 3. DQN (Deep Q-Network)
DQN uses neural networks to approximate Q-values instead of tables, enabling it to handle large state spaces. It includes experience replay and target networks for stable training.

**Key Features:**
- Neural network replaces Q-table
- Experience replay buffer breaks temporal correlations
- Target network stabilizes training
- Best for large or continuous state spaces

### 4. REINFORCE (Policy Gradient)
REINFORCE directly learns a policy network that outputs action probabilities. It optimizes the policy using the policy gradient theorem.

**Key Features:**
- Policy-based method (outputs probabilities, not values)
- Updates at episode end using complete returns
- Policy gradient: ∇J(θ) = E[∇log π(a|s) · G]
- Natural exploration through stochastic policy

### 5. MCTS (Monte Carlo Tree Search)
MCTS is a search algorithm, not a learning algorithm. It builds a search tree through simulations using four steps: Selection, Expansion, Simulation, and Backpropagation.

**Key Features:**
- No training required (pure search)
- Uses UCB1 formula to balance exploration/exploitation
- Performs random rollouts to evaluate positions
- Effective for deterministic games with perfect information

---

## How to Run

### Setup

```bash
# Install all required packages
pip install -r requirements.txt
```

### Main Program (Jupyter Notebook)
```bash
# Open the main notebook
jupyter notebook AAI551_TicTacToe_RL_Project.ipynb
```

Run all cells in order to see the results.

### Alternative: Python Scripts
```bash
# Train agents
python train_all.py

# Test agents
python test_agents.py

# Run tests
pytest tests/ -v
```

---

## Project Structure

```
AAI551_TicTacToe_RL/
├── AAI551_TicTacToe_RL_Project.ipynb  # Main program
├── agents/                             # 5 RL agent implementations
├── networks/                           # Neural networks for DQN/REINFORCE
├── utils/                              # Helper functions
├── tests/                              # Unit tests
├── config.py                           # Configuration
├── train_all.py                        # Training script
└── test_agents.py                      # Testing script
```

---

## Team Contributions

**Jinwen Wu:**
- Implemented Q-Learning and SARSA agents
- Built training and testing framework
- Created environment wrapper
- Wrote unit tests
- Project setup and configuration

**Samantha Ramcharran:**
- Implemented DQN and REINFORCE agents
- Built neural networks (Q-Network, Policy Network)
- Implemented MCTS algorithm
- Created Jupyter notebook demo
- Documentation and README

Both members worked together through pair programming and code reviews.

---

## Results

After training (all agents vs. random opponent):

| Algorithm | Win Rate | Training Episodes | Type |
|-----------|----------|-------------------|------|
| **REINFORCE** | **~95%** | 100,000 | Policy-based |
| **Q-Learning** | ~80% | 100,000 | Value-based |
| **SARSA** | ~80% | 100,000 | Value-based |
| **MCTS** | ~60% | N/A (search) | Search-based |
| **DQN** | ~44% | 100,000 | Value-based (neural) |

### Key Observations:

**REINFORCE performed best** because:
- Direct policy optimization suits this problem
- Stochastic policy provides natural exploration
- Short episodes reduce variance issues

**Q-Learning and SARSA performed similarly** because:
- Tic-Tac-Toe is deterministic (no randomness)
- Both converge to near-optimal strategies
- Small state space fits tabular methods well

**DQN underperformed** because:
- Neural network is overkill for small state space
- Needs more training data for neural network convergence
- Q-table methods are more efficient for this problem

**MCTS moderate performance** because:
- Limited simulations per move (1000)
- No learning between games
- More effective against strategic opponents

---

## Requirements Satisfied

### Part 1
-  Classes with inheritance (RLAgent base class → 5 agent classes)
-  Functions (play_episode, state_to_key, etc.)
-  Libraries (PyTorch, NumPy, Matplotlib, Pandas)
-  Exception handling + Pytest (33 tests)
-  Data I/O (save/load models with pickle and torch)
-  Loops and conditionals
-  Docstrings and comments
-  README file

### Part 2
-  Built-in modules (pickle, random, os)
-  Mutable/Immutable objects (dict, list / int, str, tuple)
-  `__str__`, `__len__`, `__repr__` methods
-  `if __name__ == "__main__"` blocks
-  List comprehension
-  Lambda functions (in tests and utilities)
-  enumerate, zip, filter (in various files)

---

## Dependencies

- Python 3.10+
- torch
- numpy
- matplotlib
- pandas
- pettingzoo
- pytest
- tqdm

See `requirements.txt` for complete list.

---



## References

1. Sutton & Barto - Reinforcement Learning: An Introduction
2. Mnih et al. - Deep Q-Learning (Nature 2015)
3. PettingZoo documentation
4. PyTorch documentation



