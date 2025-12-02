"""REINFORCE policy gradient agent."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List
from .base_agent import RLAgent
from networks.policy_network import PolicyNetwork


class REINFORCEAgent(RLAgent):
    """REINFORCE policy gradient agent."""

    def __init__(self,
                 state_dim: int = 9,
                 action_dim: int = 9,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 0.1):
        super().__init__(
            name="REINFORCE",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network
        self.policy_network = PolicyNetwork(state_dim, action_dim)

        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state)

        # Get action probabilities
        probs = self.policy_network(state_tensor)

        # Mask invalid actions and renormalize
        valid_probs = probs[valid_actions]
        valid_probs = valid_probs / valid_probs.sum()

        # Sample from valid actions
        dist = torch.distributions.Categorical(valid_probs)
        action_idx = dist.sample()

        # Map back to actual action
        action = valid_actions[action_idx.item()]

        # Store log probability for training
        log_prob = dist.log_prob(action_idx)
        self.episode_log_probs.append(log_prob)

        return action

    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> None:
        """Store transition and update policy at episode end."""
        # Store transition
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

        # Update policy when episode ends
        if done:
            self._update_policy()

    def _update_policy(self) -> None:
        """Update policy using REINFORCE algorithm."""
        if len(self.episode_rewards) == 0:
            return

        # Compute discounted returns
        returns = self._compute_returns()

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * G)

        # Optimize
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []

    def _compute_returns(self) -> torch.Tensor:
        """Compute discounted returns for episode."""
        returns = []
        G = 0

        # Compute returns backwards
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32)

    def reset_episode(self) -> None:
        """Reset episode memory."""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []

    def save(self, filepath: str) -> None:
        """Save model to file."""
        try:
            torch.save({
                'policy_network': self.policy_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'episode_count': self.episode_count
            }, filepath)
            print(f"REINFORCE agent saved to {filepath}")
        except Exception as e:
            raise IOError(f"Failed to save agent: {str(e)}")

    def load(self, filepath: str) -> None:
        """Load model from file."""
        try:
            checkpoint = torch.load(filepath)
            self.policy_network.load_state_dict(checkpoint['policy_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.episode_count = checkpoint.get('episode_count', 0)
            print(f"REINFORCE agent loaded from {filepath}")
        except Exception as e:
            raise IOError(f"Failed to load agent: {str(e)}")

    def __len__(self) -> int:
        """Return number of episodes trained."""
        return self.episode_count
