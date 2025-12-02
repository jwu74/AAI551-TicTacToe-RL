"""Tic-Tac-Toe environment wrapper."""

import numpy as np
from pettingzoo.classic import tictactoe_v3
from typing import Tuple, List, Optional


class TicTacToeEnv:
    """Wrapper for PettingZoo Tic-Tac-Toe environment."""

    def __init__(self, render_mode: Optional[str] = None):
        self.env = tictactoe_v3.env(render_mode=render_mode)
        self.render_mode = render_mode

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, str]:
        self.env.reset(seed=seed)

        # Get first agent
        agent = self.env.agent_selection

        # Get observation
        observation, _, _, _, _ = self.env.last()
        state = self._process_observation(observation, agent)

        return state, agent

    def _process_observation(self, observation: dict, agent: str) -> np.ndarray:
        board = observation['observation']

        # Get player's pieces (channel 0) and opponent's pieces (channel 1)
        player_pieces = board[:, :, 0]
        opponent_pieces = board[:, :, 1]

        # Create combined representation
        # Player = 1, Opponent = -1, Empty = 0
        state = player_pieces.astype(float) - opponent_pieces.astype(float)

        # Flatten to 1D array
        return state.flatten()

    def get_valid_actions(self, observation: dict) -> List[int]:
        action_mask = observation['action_mask']
        return [i for i in range(9) if action_mask[i] == 1]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, str, dict]:
        """
        Take a step in the environment.

        Args:
            action (int): Action to take (0-8)

        Returns:
            Tuple containing:
                - state (np.ndarray): Next state
                - reward (float): Reward received
                - terminated (bool): Whether episode ended naturally
                - truncated (bool): Whether episode was truncated
                - next_agent (str): Next agent to act
                - info (dict): Additional information
        """
        # Take action
        self.env.step(action)

        # Get next agent
        next_agent = self.env.agent_selection

        # Get observation and reward for next agent
        observation, reward, terminated, truncated, info = self.env.last()

        # Process state
        state = self._process_observation(observation, next_agent)

        return state, reward, terminated, truncated, next_agent, info

    def render(self):
        """Render the environment."""
        if self.render_mode is not None:
            self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    @staticmethod
    def print_board(state: np.ndarray):
        board = state.reshape(3, 3)
        symbols = {1: 'X', -1: 'O', 0: '.'}

        # Use lambda for symbol mapping
        get_symbol = lambda val: symbols[val]

        print("\n  0 1 2")
        for i in range(3):
            print(f"{i}", end=" ")
            for j in range(3):
                print(get_symbol(board[i, j]), end=" ")
            print()
        print()


class RandomAgent:
    """Random agent for testing."""

    def __init__(self, name: str = "Random"):
        self.name = name

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        return np.random.choice(valid_actions)

    def update(self, *args, **kwargs):
        pass

    def reset_episode(self):
        pass

    def __str__(self):
        return f"{self.name} Agent"


def play_episode(env: TicTacToeEnv,
                agent1,
                agent2,
                verbose: bool = False) -> Tuple[float, float, dict]:
    """
    Play one episode between two agents.

    Args:
        env (TicTacToeEnv): Game environment
        agent1: First agent (plays as X, goes first)
        agent2: Second agent (plays as O)
        verbose (bool): Whether to print game progress

    Returns:
        Tuple containing:
            - reward1 (float): Reward for agent1
            - reward2 (float): Reward for agent2
            - info (dict): Episode information
    """
    # Reset environment
    state, current_agent_name = env.reset()

    agents = {'player_1': agent1, 'player_2': agent2}

    # Track history for learning
    history = {'player_1': [], 'player_2': []}

    episode_terminated = False
    episode_truncated = False

    if verbose:
        print(f"\n{'='*40}")
        print("Starting new episode")
        print(f"{'='*40}")
        env.print_board(state)

    step_count = 0

    while not (episode_terminated or episode_truncated):
        step_count += 1

        # Get current agent
        current_agent = agents[current_agent_name]

        # Get observation to find valid actions
        observation, _, _, _, _ = env.env.last()
        valid_actions = env.get_valid_actions(observation)

        if verbose:
            print(f"\nStep {step_count}: {current_agent_name} ({current_agent.name})")
            print(f"Valid actions: {valid_actions}")

        # Select action
        action = current_agent.select_action(state, valid_actions)

        if verbose:
            print(f"Selected action: {action}")

        # Store transition
        transition = {
            'state': state.copy(),
            'action': action,
            'agent': current_agent_name
        }

        # Take step
        next_state, reward, terminated, truncated, next_agent_name, info = env.step(action)

        if verbose:
            env.print_board(next_state)

        # Get the ACTUAL reward for the current agent from env.rewards
        # PettingZoo's env.last() returns the next agent's reward, not the current one!
        actual_reward = env.env.rewards.get(current_agent_name, 0.0)

        # Update transition with correct reward
        transition['reward'] = actual_reward
        transition['next_state'] = next_state.copy()
        transition['done'] = terminated or truncated

        history[current_agent_name].append(transition)

        # Update current agent (immediate learning)
        current_agent.update(
            state=transition['state'],
            action=transition['action'],
            reward=transition['reward'],
            next_state=transition['next_state'],
            done=transition['done']
        )

        # Move to next state
        state = next_state
        current_agent_name = next_agent_name
        episode_terminated = terminated
        episode_truncated = truncated

    # Get final rewards from the environment's reward dictionary
    reward1 = env.env.rewards.get('player_1', 0.0)
    reward2 = env.env.rewards.get('player_2', 0.0)

    if verbose:
        print(f"\n{'='*40}")
        print(f"Episode finished after {step_count} steps")
        print(f"Agent 1 ({agent1.name}) reward: {reward1}")
        print(f"Agent 2 ({agent2.name}) reward: {reward2}")
        print(f"{'='*40}\n")

    info = {
        'steps': step_count,
        'reward1': reward1,
        'reward2': reward2,
        'winner': 'player_1' if reward1 > 0 else ('player_2' if reward2 > 0 else 'draw')
    }

    return reward1, reward2, info


