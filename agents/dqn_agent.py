"""DQN agent with neural network Q-learning."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List
from .base_agent import RLAgent
from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer


class DQNAgent(RLAgent):
    """DQN agent with experience replay and target network."""

    def __init__(self,
                 state_dim: int = 9,
                 action_dim: int = 9,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 0.9,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05,
                 batch_size: int = 64,
                 buffer_size: int = 10000,
                 target_update: int = 100):
        super().__init__(
            name="DQN",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.update_counter = 0

        # Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size, batch_size=batch_size)

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Explore
            return random.choice(valid_actions)
        else:
            # Exploit
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.q_network(state_tensor)

                # Mask invalid actions
                mask = torch.full_like(q_values, float('-inf'))
                mask[valid_actions] = 0
                q_values = q_values + mask

                return q_values.argmax().item()

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> None:
        """Store transition and train network."""
        # Add to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Train if enough samples
        if self.replay_buffer.is_ready():
            self._train_step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _train_step(self) -> None:
        """Perform one training step."""
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset_episode(self) -> None:
        """Decay epsilon after episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save model to file."""
        try:
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'episode_count': self.episode_count
            }, filepath)
            print(f"DQN agent saved to {filepath}")
        except Exception as e:
            raise IOError(f"Failed to save agent: {str(e)}")

    def load(self, filepath: str) -> None:
        """Load model from file."""
        try:
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.episode_count = checkpoint.get('episode_count', 0)
            print(f"DQN agent loaded from {filepath}")
        except Exception as e:
            raise IOError(f"Failed to load agent: {str(e)}")

    def __len__(self) -> int:
        """Return replay buffer size."""
        return len(self.replay_buffer)
