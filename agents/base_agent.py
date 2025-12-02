"""Base RL agent class."""

from abc import ABC, abstractmethod
import json
from typing import Dict, List, Optional
import numpy as np


class RLAgent(ABC):
    """Base class for all RL agents."""

    def __init__(self,
                 name: str,
                 learning_rate: float = 0.2,
                 gamma: float = 0.99,
                 epsilon: float = 0.05):
        # Validate parameters
        if not 0 < learning_rate <= 1:
            raise ValueError(f"Learning rate must be in (0, 1], got {learning_rate}")
        if not 0 <= gamma <= 1:
            raise ValueError(f"Gamma must be in [0, 1], got {gamma}")
        if not 0 <= epsilon <= 1:
            raise ValueError(f"Epsilon must be in [0, 1], got {epsilon}")

        self.name = name
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Training tracking
        self.training_stats = []
        self.episode_count = 0
        self.total_rewards = 0.0

    @abstractmethod
    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        pass

    @abstractmethod
    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> None:
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass

    def reset_episode(self) -> None:
        pass

    def log_episode(self, reward: float, info: Optional[Dict] = None) -> None:
        self.episode_count += 1
        self.total_rewards += reward

        stats = {
            'episode': self.episode_count,
            'reward': reward,
            'avg_reward': self.total_rewards / self.episode_count
        }

        if info:
            stats.update(info)

        self.training_stats.append(stats)

    def get_stats(self) -> List[Dict]:
        return self.training_stats

    def get_avg_reward(self, last_n: int = 100) -> float:
        if not self.training_stats:
            return 0.0

        recent_stats = self.training_stats[-last_n:]
        # Use filter to get only positive rewards
        rewards = list(filter(lambda s: 'reward' in s, recent_stats))
        if not rewards:
            return 0.0
        return np.mean([s['reward'] for s in rewards])

    def set_epsilon(self, epsilon: float) -> None:
        if not 0 <= epsilon <= 1:
            raise ValueError(f"Epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = epsilon

    def decay_epsilon(self, decay_rate: float = 0.995) -> None:
        self.epsilon = max(0.01, self.epsilon * decay_rate)

    def __str__(self) -> str:
        return (f"{self.name} Agent\n"
                f"  Learning Rate: {self.learning_rate}\n"
                f"  Gamma: {self.gamma}\n"
                f"  Epsilon: {self.epsilon:.4f}\n"
                f"  Episodes Trained: {self.episode_count}")

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"lr={self.learning_rate}, gamma={self.gamma}, "
                f"epsilon={self.epsilon})")

    def __len__(self) -> int:
        return self.episode_count

    def save_stats(self, filepath: str) -> None:
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'agent_name': self.name,
                    'episode_count': self.episode_count,
                    'total_rewards': self.total_rewards,
                    'stats': self.training_stats
                }, f, indent=2)
            print(f"Stats saved to {filepath}")
        except Exception as e:
            raise IOError(f"Failed to save stats to {filepath}: {str(e)}")

    def load_stats(self, filepath: str) -> None:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.episode_count = data['episode_count']
                self.total_rewards = data['total_rewards']
                self.training_stats = data['stats']
            print(f"Stats loaded from {filepath}")
        except Exception as e:
            raise IOError(f"Failed to load stats from {filepath}: {str(e)}")


def state_to_key(state: np.ndarray) -> str:
    return str(state.flatten().tolist())

def key_to_state(key: str) -> np.ndarray:
    return np.array(eval(key)).reshape(3, 3)
