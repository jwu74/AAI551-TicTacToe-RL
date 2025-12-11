"""
Unified Testing Script for All Agents

Tests all trained agents and compares their performance.
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from agents import QLearningAgent, SARSAAgent, DQNAgent, REINFORCEAgent, MCTSAgent
from utils.environment import TicTacToeEnv, RandomAgent, play_episode
from config import Config


def test_agent(agent, agent_name, num_games=1000, verbose=True):
    """
    Test a single agent.

    Args:
        agent: Agent to test
        agent_name: Name of agent
        num_games: Number of test games
        verbose: Whether to print progress
    """
    env = TicTacToeEnv()
    opponent = RandomAgent()

    # Set epsilon to 0 for pure exploitation (except MCTS)
    if hasattr(agent, 'epsilon') and agent_name != 'MCTS':
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0

    wins, draws, losses = 0, 0, 0

    # Test loop
    if verbose:
        iterator = tqdm(range(num_games), desc=f"Testing {agent_name}")
    else:
        iterator = range(num_games)

    for _ in iterator:
        reward, _, info = play_episode(env, agent, opponent, verbose=False)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    env.close()

    # Restore epsilon
    if hasattr(agent, 'epsilon') and agent_name != 'MCTS':
        agent.epsilon = original_epsilon

    # Calculate the rates
    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = losses / num_games

    if verbose:
        print(f"\n{agent_name} Results:")
        print(f"  Wins:  {wins:4d} ({win_rate*100:5.2f}%)")
        print(f"  Draws: {draws:4d} ({draw_rate*100:5.2f}%)")
        print(f"  Losses: {losses:4d} ({loss_rate*100:5.2f}%)")

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate
    }


def plot_results(results, save_path='results/comparison.png'):
    """Plot comparison of all agents."""
    agent_names = list(results.keys())
    win_rates = [results[name]['win_rate'] * 100 for name in agent_names]
    draw_rates = [results[name]['draw_rate'] * 100 for name in agent_names]
    loss_rates = [results[name]['loss_rate'] * 100 for name in agent_names]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(agent_names))
    width = 0.25

    bars1 = ax.bar(x - width, win_rates, width, label='Wins', color='green', alpha=0.8)
    bars2 = ax.bar(x, draw_rates, width, label='Draws', color='gray', alpha=0.8)
    bars3 = ax.bar(x + width, loss_rates, width, label='Losses', color='red', alpha=0.8)

    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Agent Performance Comparison vs Random Opponent', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")

    plt.show()


def main():
    """Main testing function."""
    print("="*60)
    print("AAI 551 Tic-Tac-Toe RL Project")
    print("Testing All Agents")
    print("="*60)

    # Set random seed
    np.random.seed(Config.RANDOM_SEED)

    # Create agents and load models
    agents = {}

    # Q-Learning
    try:
        agent = QLearningAgent()
        agent.load(Config.MODEL_PATHS['q_learning'])
        agents['Q-Learning'] = agent
    except Exception as e:
        print(f"Warning: Could not load Q-Learning agent: {e}")

    # SARSA
    try:
        agent = SARSAAgent()
        agent.load(Config.MODEL_PATHS['sarsa'])
        agents['SARSA'] = agent
    except Exception as e:
        print(f"Warning: Could not load SARSA agent: {e}")

    # DQN
    try:
        agent = DQNAgent()
        agent.load(Config.MODEL_PATHS['dqn'])
        agents['DQN'] = agent
    except Exception as e:
        print(f"Warning: Could not load DQN agent: {e}")

    # REINFORCE
    try:
        agent = REINFORCEAgent()
        agent.load(Config.MODEL_PATHS['reinforce'])
        agents['REINFORCE'] = agent
    except Exception as e:
        print(f"Warning: Could not load REINFORCE agent: {e}")

    # MCTS (no loading needed)
    agents['MCTS'] = MCTSAgent(**Config.MCTS_CONFIG)

    if not agents:
        print("\nNo agents found! Please train agents first:")
        print("  python train_all.py")
        return

    print(f"\nFound {len(agents)} agent(s) to test\n")

    # Test each agent
    results = {}
    for name, agent in agents.items():
        results[name] = test_agent(agent, name, num_games=Config.EVAL_GAMES, verbose=True)

    # Print summary
    print("\n" + "="*60)
    print("Testing Summary")
    print("="*60)
    print(f"{'Agent':<15s} {'Win Rate':>10s} {'Draw Rate':>10s} {'Loss Rate':>10s}")
    print("-"*60)
    for name, result in results.items():
        print(f"{name:<15s} {result['win_rate']*100:>9.2f}% "
              f"{result['draw_rate']*100:>9.2f}% "
              f"{result['loss_rate']*100:>9.2f}%")
    print("="*60)

    # Plot results
    try:
        plot_results(results)
    except Exception as e:
        print(f"\nWarning: Could not create plot: {e}")

    print("\nTesting completed!")


if __name__ == "__main__":
    main()

