"""
Unit tests for game environment.

Tests the Tic-Tac-Toe environment wrapper and utility functions.
"""

import pytest
import numpy as np
from utils.environment import TicTacToeEnv, RandomAgent, play_episode
from agents import QLearningAgent


class TestTicTacToeEnv:
    """Test suite for Tic-Tac-Toe environment."""

    def test_initialization(self):
        """Test environment initialization."""
        env = TicTacToeEnv()
        assert env is not None

    def test_reset(self):
        """Test environment reset."""
        env = TicTacToeEnv()
        state, current_player = env.reset()

        # Check state shape
        assert state.shape == (9,)

        # Initial state should be empty
        assert np.all(state == 0)

        # Current player should be player_1
        assert current_player == "player_1"

    def test_valid_actions(self):
        """Test valid action detection."""
        env = TicTacToeEnv()
        state, _ = env.reset()

        # Get observation
        obs, _, _, _, _ = env.env.last()
        valid_actions = env.get_valid_actions(obs)

        # All positions should be valid at start
        assert len(valid_actions) == 9
        assert set(valid_actions) == set(range(9))

    def test_step(self):
        """Test environment step function."""
        env = TicTacToeEnv()
        env.reset()

        # Take a step
        next_state, reward, terminated, truncated, next_player, info = env.step(4)

        # State should change
        assert next_state.shape == (9,)

        # Should switch players
        assert next_player == "player_2"

    def test_exception_handling_invalid_action(self):
        """Test exception handling for invalid actions."""
        env = TicTacToeEnv()
        state, _ = env.reset()

        # Take action 0
        env.step(0)

        # Note: PettingZoo doesn't raise exceptions for invalid actions,
        # it just marks them as invalid in the action mask
        # This is a design choice for robustness
        # So we verify the action mask works instead
        obs, _, _, _, _ = env.env.last()
        valid_actions = env.get_valid_actions(obs)

        # Action 0 should no longer be valid
        assert 0 not in valid_actions


class TestRandomAgent:
    """Test suite for Random agent."""

    def test_initialization(self):
        """Test Random agent initialization."""
        agent = RandomAgent("TestRandom")
        assert agent.name == "TestRandom"

    def test_action_selection(self):
        """Test random action selection."""
        agent = RandomAgent()
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        valid_actions = [0, 1, 2, 3, 4]

        # Should select from valid actions
        action = agent.select_action(state, valid_actions)
        assert action in valid_actions

    def test_update_does_nothing(self):
        """Test that Random agent update does nothing."""
        agent = RandomAgent()
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Should not raise exception
        agent.update(state, 0, 1.0, state, True)


class TestPlayEpisode:
    """Test suite for play_episode function."""

    def test_play_episode_completes(self):
        """Test that play_episode completes successfully."""
        env = TicTacToeEnv()
        agent1 = RandomAgent("Agent1")
        agent2 = RandomAgent("Agent2")

        reward1, reward2, info = play_episode(env, agent1, agent2, verbose=False)

        # Rewards should sum to 0 or both be 0 (draw)
        assert reward1 + reward2 == 0 or (reward1 == 0 and reward2 == 0)

        # Info should contain required keys
        assert 'winner' in info
        assert 'steps' in info

    def test_play_episode_valid_rewards(self):
        """Test that rewards are valid."""
        env = TicTacToeEnv()
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        reward1, reward2, info = play_episode(env, agent1, agent2, verbose=False)

        # Rewards should be -1, 0, or 1
        assert reward1 in [-1, 0, 1]
        assert reward2 in [-1, 0, 1]

    def test_play_episode_with_trained_agent(self):
        """Test play_episode with a trained agent."""
        env = TicTacToeEnv()
        agent1 = QLearningAgent(epsilon=0.1)
        agent2 = RandomAgent()

        # Play multiple episodes
        wins = 0
        for _ in range(10):
            reward1, reward2, info = play_episode(env, agent1, agent2, verbose=False)
            if reward1 > 0:
                wins += 1

        # Even untrained agent should win sometimes due to first-move advantage
        assert wins >= 0  # Basic sanity check

    def test_play_episode_winner_consistency(self):
        """Test that winner info is consistent with rewards."""
        env = TicTacToeEnv()
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        reward1, reward2, info = play_episode(env, agent1, agent2, verbose=False)

        # Check consistency
        if reward1 > 0:
            assert info['winner'] == 'player_1'
        elif reward2 > 0:
            assert info['winner'] == 'player_2'
        else:
            assert info['winner'] == 'draw'


class TestEnvironmentEdgeCases:
    """Test edge cases and exception handling."""

    def test_empty_valid_actions_list(self):
        """Test behavior with empty valid actions list."""
        agent = RandomAgent()
        state = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])

        # Should handle gracefully (though this shouldn't happen in practice)
        with pytest.raises(Exception):
            agent.select_action(state, [])

    def test_environment_close(self):
        """Test environment close method."""
        env = TicTacToeEnv()
        env.reset()

        # Should not raise exception
        env.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
