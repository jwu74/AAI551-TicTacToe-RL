"""
Unit tests for RL agents.

Tests the functionality of all five RL agents including:
- Agent initialization
- Action selection
- Q-value updates
- Model saving/loading
- Exception handling
"""

import pytest
import numpy as np
import os
import tempfile
from agents import QLearningAgent, SARSAAgent, DQNAgent, REINFORCEAgent, MCTSAgent


class TestQLearningAgent:
    """Test suite for Q-Learning agent."""

    def test_initialization(self):
        """Test Q-Learning agent initialization."""
        agent = QLearningAgent(learning_rate=0.1, gamma=0.9, epsilon=0.1)

        assert agent.learning_rate == 0.1
        assert agent.gamma == 0.9
        assert agent.epsilon == 0.1
        assert len(agent.q_table) == 0
        assert agent.name == "Q-Learning"

    def test_action_selection(self):
        """Test action selection mechanism."""
        agent = QLearningAgent(epsilon=0.0)  # No exploration
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        valid_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Should return a valid action
        action = agent.select_action(state, valid_actions)
        assert action in valid_actions

    def test_q_value_update(self):
        """Test Q-value update mechanism."""
        agent = QLearningAgent(learning_rate=0.2, gamma=0.99)

        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        action = 4
        reward = 1.0
        next_state = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0])
        done = True

        # Q-value should be 0 before update
        assert agent.get_q_value(state, action) == 0.0

        # Update Q-value
        agent.update(state, action, reward, next_state, done)

        # Q-value should be updated (approximately learning_rate * reward)
        assert agent.get_q_value(state, action) == pytest.approx(0.2, rel=0.01)

    def test_save_load(self):
        """Test model saving and loading."""
        agent = QLearningAgent()

        # Train a bit
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        agent.update(state, 4, 1.0, state, True)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            agent.save(temp_path)

            # Load into new agent
            new_agent = QLearningAgent()
            new_agent.load(temp_path)

            # Check Q-table matches
            assert len(new_agent.q_table) == len(agent.q_table)
            assert new_agent.get_q_value(state, 4) == agent.get_q_value(state, 4)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_invalid_learning_rate(self):
        """Test exception handling for invalid learning rate."""
        with pytest.raises(ValueError, match="Learning rate must be in"):
            QLearningAgent(learning_rate=0.0)

        with pytest.raises(ValueError, match="Learning rate must be in"):
            QLearningAgent(learning_rate=1.5)

    def test_string_representation(self):
        """Test __str__ method."""
        agent = QLearningAgent(learning_rate=0.2)
        str_repr = str(agent)

        assert "Q-Learning" in str_repr
        assert "lr=0.2" in str_repr or "0.2" in str_repr

    def test_length(self):
        """Test __len__ method."""
        agent = QLearningAgent()
        assert len(agent) == 0

        # Add some states
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        agent.update(state, 0, 1.0, state, True)
        assert len(agent) > 0


class TestSARSAAgent:
    """Test suite for SARSA agent."""

    def test_initialization(self):
        """Test SARSA agent initialization."""
        agent = SARSAAgent(learning_rate=0.15, gamma=0.95, epsilon=0.2)

        assert agent.learning_rate == 0.15
        assert agent.gamma == 0.95
        assert agent.epsilon == 0.2
        assert agent.name == "SARSA"

    def test_on_policy_update(self):
        """Test SARSA on-policy update mechanism."""
        agent = SARSAAgent(learning_rate=0.2)

        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        action = 4
        reward = 1.0
        next_state = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0])
        done = True

        # Perform update
        agent.update(state, action, reward, next_state, done)

        # Check Q-value updated
        assert agent.get_q_value(state, action) > 0


class TestDQNAgent:
    """Test suite for DQN agent."""

    def test_initialization(self):
        """Test DQN agent initialization."""
        agent = DQNAgent(learning_rate=0.001, gamma=0.99, epsilon=0.9)

        assert agent.learning_rate == 0.001
        assert agent.gamma == 0.99
        assert agent.epsilon == 0.9
        assert agent.name == "DQN"

    def test_action_selection(self):
        """Test DQN action selection."""
        agent = DQNAgent(epsilon=0.0)
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        valid_actions = [0, 1, 2, 3, 4]

        action = agent.select_action(state, valid_actions)
        assert action in valid_actions

    def test_experience_replay(self):
        """Test experience replay buffer."""
        agent = DQNAgent(buffer_size=100)

        # Add experiences
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        next_state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

        for i in range(10):
            agent.update(state, i % 9, 1.0, next_state, False)

        # Buffer should have experiences
        assert len(agent.replay_buffer) > 0


class TestREINFORCEAgent:
    """Test suite for REINFORCE agent."""

    def test_initialization(self):
        """Test REINFORCE agent initialization."""
        agent = REINFORCEAgent(learning_rate=0.001, gamma=0.99)

        assert agent.learning_rate == 0.001
        assert agent.gamma == 0.99
        assert agent.name == "REINFORCE"

    def test_policy_action_selection(self):
        """Test policy-based action selection."""
        agent = REINFORCEAgent()
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        valid_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Should select a valid action
        action = agent.select_action(state, valid_actions)
        assert action in valid_actions


class TestMCTSAgent:
    """Test suite for MCTS agent."""

    def test_initialization(self):
        """Test MCTS agent initialization."""
        agent = MCTSAgent(n_simulations=100, exploration_constant=1.41)

        assert agent.n_simulations == 100
        assert agent.exploration_constant == 1.41
        assert agent.name == "MCTS"

    def test_action_selection(self):
        """Test MCTS action selection."""
        agent = MCTSAgent(n_simulations=10)
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        valid_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        action = agent.select_action(state, valid_actions)
        assert action in valid_actions


class TestAgentExceptions:
    """Test exception handling across agents."""

    def test_invalid_gamma(self):
        """Test exception for invalid gamma values."""
        with pytest.raises(ValueError, match="Gamma must be in"):
            QLearningAgent(gamma=1.5)

        with pytest.raises(ValueError, match="Gamma must be in"):
            SARSAAgent(gamma=-0.1)

    def test_invalid_epsilon(self):
        """Test exception for invalid epsilon values."""
        with pytest.raises(ValueError, match="Epsilon must be in"):
            QLearningAgent(epsilon=-0.1)

        with pytest.raises(ValueError, match="Epsilon must be in"):
            DQNAgent(epsilon=1.5)

    def test_load_nonexistent_file(self):
        """Test exception when loading from nonexistent file."""
        agent = QLearningAgent()

        with pytest.raises(IOError, match="Failed to load agent"):
            agent.load("nonexistent_file_12345.pkl")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
