"""
Policy Network for REINFORCE Agent.
Outputs action probabilities given a state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Simple policy network with 2 hidden layers.
    Input: state (9 values) -> Hidden layers -> Output: action probs (9 values)
    """

    def __init__(self, input_dim=9, output_dim=9, hidden_dims=[128, 64]):
        """
        Initialize policy network.

        Args:
            input_dim: State dimension (default 9 for tic-tac-toe)
            output_dim: Number of actions (default 9)
            hidden_dims: Hidden layer sizes
        """
        super(PolicyNetwork, self).__init__()

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

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass through network.

        Args:
            state: Input state tensor

        Returns:
            Action probabilities (softmax output)
        """
        x = self.network(state)
        return F.softmax(x, dim=-1)


if __name__ == "__main__":
    # Simple test
    print("Testing PolicyNetwork...")

    net = PolicyNetwork()
    print(f"Network: {net}")

    # Test forward pass
    state = torch.zeros(1, 9)
    output = net(state)
    print(f"Output shape: {output.shape}")
    print(f"Output sum: {output.sum().item():.4f} (should be 1.0)")

    print("✅ Test passed!")
