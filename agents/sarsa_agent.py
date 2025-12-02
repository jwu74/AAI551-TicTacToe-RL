"""SARSA agent implementation."""

import numpy as np
import pickle
import random
from typing import List, Dict
from .base_agent import RLAgent, state_to_key


class SARSAAgent(RLAgent):
    """SARSA agent with on-policy learning."""

    def __init__(self,
                 learning_rate: float = 0.2,
                 gamma: float = 0.99,
                 epsilon: float = 0.05,
                 epsilon_decay: float = 0.995):
        super().__init__(
            name="SARSA",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon
        )

        self.q_table = {}
        self.epsilon_decay = epsilon_decay
        self.next_action = None  # Store next action for SARSA update

    def _get_state_key(self, state: np.ndarray) -> str:
        """Convert state to string key."""
        return state_to_key(state)

    def _initialize_state(self, state_key: str, valid_actions: List[int]) -> None:
        """Initialize Q-values for new state."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {str(action): 0.0 for action in valid_actions}

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for state-action pair."""
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            return 0.0
        return self.q_table[state_key].get(str(action), 0.0)

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Select action using epsilon-greedy policy."""
        state_key = self._get_state_key(state)
        self._initialize_state(state_key, valid_actions)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            return self._greedy_action(state_key, valid_actions)

    def _greedy_action(self, state_key: str, valid_actions: List[int]) -> int:
        """Select greedy action."""
        if state_key not in self.q_table:
            return random.choice(valid_actions)

        q_values = {action: self.q_table[state_key].get(str(action), 0.0)
                   for action in valid_actions}
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> None:
        state_key = self._get_state_key(state)
        action_key = str(action)

        # Get current Q-value
        current_q = self.get_q_value(state, action)

        # SARSA: use actual next action's Q-value
        if done:
            next_q = 0.0
        else:
            # Get valid actions for next state
            next_state_flat = next_state.flatten()
            next_valid_actions = [i for i in range(9) if next_state_flat[i] == 0]

            if next_valid_actions:
                # Select next action using current policy (epsilon-greedy)
                next_action = self.select_action(next_state, next_valid_actions)
                next_q = self.get_q_value(next_state, next_action)
            else:
                next_q = 0.0

        # SARSA update
        target = reward + self.gamma * next_q
        new_q = current_q + self.learning_rate * (target - current_q)

        # Update Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action_key] = new_q

    def reset_episode(self) -> None:
        """Reset episode and decay epsilon."""
        self.decay_epsilon(self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save Q-table to file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon,
                    'episode_count': self.episode_count
                }, f)
            print(f"SARSA agent saved to {filepath}")
        except Exception as e:
            raise IOError(f"Failed to save agent: {str(e)}")

    def load(self, filepath: str) -> None:
        """Load Q-table from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data.get('epsilon', self.epsilon)
                self.episode_count = data.get('episode_count', 0)
            print(f"SARSA agent loaded from {filepath}")
            print(f"Loaded {len(self.q_table)} states")
        except Exception as e:
            raise IOError(f"Failed to load agent: {str(e)}")

    def __len__(self) -> int:
        """Return number of states in Q-table."""
        return len(self.q_table)
