"""
Unified Training Script for All Agents

Trains all 5 RL agents and saves them to models/ directory.
"""

import os
import numpy as np
from tqdm import tqdm

from agents import QLearningAgent, SARSAAgent, DQNAgent, REINFORCEAgent, MCTSAgent
from utils.environment import TicTacToeEnv, RandomAgent, play_episode
from config import Config


def train_agent(agent, agent_name, num_episodes=1000, verbose=True):
    """
    Train a single agent.

    Args:
        agent: Agent to train
        agent_name: Name of agent
        num_episodes: Number of training episodes
        verbose: Whether to print progress
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {agent_name}")
        print(f"{'='*60}")

    env = TicTacToeEnv()
    opponent = RandomAgent()

    wins, draws, losses = 0, 0, 0

    # Training loop
    for episode in tqdm(range(num_episodes), desc=f"Training {agent_name}"):
        reward, _, info = play_episode(env, agent, opponent, verbose=False)

        # Track stats
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

        # Update agent
        agent.log_episode(reward)
        agent.reset_episode()

    env.close()

    # Print results
    win_rate = wins / num_episodes
    draw_rate = draws / num_episodes
    loss_rate = losses / num_episodes

    if verbose:
        print(f"\nTraining completed!")
        print(f"  Win Rate:  {win_rate*100:.2f}%")
        print(f"  Draw Rate: {draw_rate*100:.2f}%")
        print(f"  Loss Rate: {loss_rate*100:.2f}%")

    return {'wins': wins, 'draws': draws, 'losses': losses,
            'win_rate': win_rate, 'draw_rate': draw_rate, 'loss_rate': loss_rate}


def main(agent_name=None, episodes=None):
    """Main training function."""
    print("="*60)
    print("AAI 551 Tic-Tac-Toe RL Project")
    if agent_name:
        print(f"Training {agent_name}")
    else:
        print("Training All Agents")
    print("="*60)

    # Set random seed
    np.random.seed(Config.RANDOM_SEED)

    # Create models directory
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # Define agents
    agents = {
        'q-learning': (QLearningAgent(**Config.Q_LEARNING_CONFIG),
                      Config.TRAINING_EPISODES['q_learning']),
        'sarsa': (SARSAAgent(**Config.SARSA_CONFIG),
                 Config.TRAINING_EPISODES['sarsa']),
        'dqn': (DQNAgent(**Config.DQN_CONFIG),
               Config.TRAINING_EPISODES['dqn']),
        'reinforce': (REINFORCEAgent(**Config.REINFORCE_CONFIG),
                     Config.TRAINING_EPISODES['reinforce']),
        'mcts': (MCTSAgent(**Config.MCTS_CONFIG), 0)
    }

    # Filter agents if specific agent requested
    if agent_name:
        agent_name_lower = agent_name.lower()
        if agent_name_lower not in agents:
            print(f"Error: Unknown agent '{agent_name}'")
            print(f"Available agents: {', '.join(agents.keys())}")
            return
        agents = {agent_name_lower: agents[agent_name_lower]}

    results = {}

    # Train each agent
    for name, (agent, num_episodes) in agents.items():
        # Override episodes if specified
        if episodes is not None:
            num_episodes = episodes

        if num_episodes > 0:
            display_name = name.replace('_', '-').title()
            results[display_name] = train_agent(agent, display_name, num_episodes, verbose=True)

            # Save agent
            save_path = Config.MODEL_PATHS.get(name)
            if save_path:
                agent.save(save_path)
        else:
            print(f"\n{name}: No training needed (search-based)")
            results[name] = {'win_rate': 'N/A'}

    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for name, result in results.items():
        win_rate = result.get('win_rate')
        if isinstance(win_rate, float):
            print(f"{name:12s}: {win_rate*100:5.2f}% win rate")
        else:
            print(f"{name:12s}: {win_rate}")

    print("="*60)
    print("\nTraining completed!")
    print(f"Models saved in: {Config.MODEL_DIR}")
    print("\nTo test agents, run: python test_agents.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train RL agents for Tic-Tac-Toe')
    parser.add_argument('--agent', type=str, default=None,
                        help='Train specific agent: q-learning, sarsa, dqn, reinforce, mcts')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes (overrides config)')

    args = parser.parse_args()

    main(agent_name=args.agent, episodes=args.episodes)
