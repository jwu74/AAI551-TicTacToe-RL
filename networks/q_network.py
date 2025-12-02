"""
Q-Network for DQN Agent.
Estimates Q-values for state-action pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Simple Q-network with 2 hidden layers.
    Input: state (9 values) -> Hidden layers -> Output: Q-values for each action (9 values)
    """

    def __init__(self, input_dim=9, output_dim=9, hidden_dims=[128, 64]):
        """
        Initialize Q-network.

        Args:
            input_dim: State dimension (default 9 for tic-tac-toe)
            output_dim: Number of actions (default 9)
            hidden_dims: Hidden layer sizes
        """
        super(QNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (Q-values for each action)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass to get Q-values.

        Args:
            state: Input state tensor

        Returns:
            Q-values for each action
        """
        return self.network(state)


if __name__ == "__main__":
    # Simple test
    print("Testing QNetwork...")

    net = QNetwork()
    print(f"Network: {net}")

    # Test forward pass
    state = torch.zeros(1, 9)
    q_values = net(state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")

    print("✅ Test passed!")
