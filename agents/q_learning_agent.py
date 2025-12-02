"""Q-Learning agent implementation."""

import numpy as np
import pickle
import random
from typing import List, Dict, Tuple
from .base_agent import RLAgent, state_to_key


class QLearningAgent(RLAgent):
    """Q-Learning agent with tabular Q-values."""

    def __init__(self,
                 learning_rate: float = 0.2,
                 gamma: float = 0.99,
                 epsilon: float = 0.05,
                 epsilon_decay: float = 0.995):
        super().__init__(
            name="Q-Learning",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon
        )

        self.q_table = {}
        self.epsilon_decay = epsilon_decay

    def _get_state_key(self, state: np.ndarray) -> str:
        return state_to_key(state)

    def _initialize_state(self, state_key: str, valid_actions: List[int]) -> None:
        if state_key not in self.q_table:
            self.q_table[state_key] = {str(action): 0.0 for action in valid_actions}

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            return 0.0
        return self.q_table[state_key].get(str(action), 0.0)

    def get_max_q_value(self, state: np.ndarray) -> float:
        state_key = self._get_state_key(state)
        if state_key not in self.q_table or not self.q_table[state_key]:
            return 0.0

        q_values = list(self.q_table[state_key].values())
        return max(q_values) if q_values else 0.0

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        state_key = self._get_state_key(state)

        # Initialize state if not seen before
        self._initialize_state(state_key, valid_actions)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            return self._greedy_action(state_key, valid_actions)

    def _greedy_action(self, state_key: str, valid_actions: List[int]) -> int:
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

        current_q = self.get_q_value(state, action)

        if done:
            max_next_q = 0.0
        else:
            max_next_q = self.get_max_q_value(next_state)

        target = reward + self.gamma * max_next_q
        new_q = current_q + self.learning_rate * (target - current_q)

        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action_key] = new_q

    def reset_episode(self) -> None:
        self.decay_epsilon(self.epsilon_decay)

    def save(self, filepath: str) -> None:
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon,
                    'episode_count': self.episode_count,
                    'training_stats': self.training_stats
                }, f)
            print(f"Q-Learning agent saved to {filepath}")
        except Exception as e:
            raise IOError(f"Failed to save agent: {str(e)}")

    def load(self, filepath: str) -> None:
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data.get('epsilon', self.epsilon)
                self.episode_count = data.get('episode_count', 0)
                self.training_stats = data.get('training_stats', [])
            print(f"Q-Learning agent loaded from {filepath}")
            print(f"Loaded {len(self.q_table)} states")
        except Exception as e:
            raise IOError(f"Failed to load agent: {str(e)}")

    def get_policy(self, state: np.ndarray, valid_actions: List[int]) -> Dict[int, float]:
        state_key = self._get_state_key(state)
        self._initialize_state(state_key, valid_actions)

        q_values = {action: self.q_table[state_key].get(str(action), 0.0)
                   for action in valid_actions}

        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]

        n_actions = len(valid_actions)
        n_best = len(best_actions)

        policy = {}
        for action in valid_actions:
            if action in best_actions:
                policy[action] = (1 - self.epsilon) / n_best + self.epsilon / n_actions
            else:
                policy[action] = self.epsilon / n_actions

        return policy

    def __len__(self) -> int:
        return len(self.q_table)

    def print_q_table(self, max_states: int = 10) -> None:
        print(f"\n{'='*60}")
        print(f"Q-Table for {self.name}")
        print(f"Total states: {len(self.q_table)}")
        print(f"{'='*60}\n")

        for i, (state_key, actions) in enumerate(self.q_table.items()):
            if i >= max_states:
                print(f"... ({len(self.q_table) - max_states} more states)")
                break

            print(f"State: {state_key}")
            for action, q_value in actions.items():
                print(f"  Action {action}: Q = {q_value:.4f}")
            print()


