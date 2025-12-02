"""MCTS agent using tree search."""

import numpy as np
import random
from typing import List
from .base_agent import RLAgent
from utils.mcts_node import MCTSNode


class MCTSAgent(RLAgent):
    """MCTS agent using tree search and simulations."""

    def __init__(self,
                 n_simulations: int = 1000,
                 exploration_constant: float = 1.414):
        super().__init__(
            name="MCTS",
            learning_rate=0.001,  # Dummy value (not used)
            gamma=1.0,
            epsilon=0.0
        )

        self.n_simulations = n_simulations
        self.exploration_constant = exploration_constant

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Select action using MCTS."""
        # Create root node
        root = MCTSNode(state.reshape(3, 3), valid_actions=valid_actions)

        # Run simulations
        for _ in range(self.n_simulations):
            self._simulate(root)

        # Select best action
        return root.best_action()

    def _simulate(self, root: MCTSNode) -> None:
        """Run one MCTS simulation."""
        node = root
        state = node.state.copy()

        # 1. Selection - traverse tree using UCB1
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_constant)
            if node.action is not None:
                # Apply action to state
                state = self._apply_action(state, node.action, player=1)

        # 2. Expansion - add new child if not terminal
        if not node.is_terminal() and not node.is_fully_expanded():
            action = random.choice(node.untried_actions)
            next_state = self._apply_action(state, action, player=1)
            node = node.expand(action, next_state)
            state = next_state

        # 3. Simulation - random playout
        reward = self._rollout(state.copy())

        # 4. Backpropagation - update statistics
        node.backpropagate(reward)

    def _apply_action(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """Apply action to state."""
        new_state = state.copy()
        flat_state = new_state.flatten()
        flat_state[action] = player
        return flat_state.reshape(3, 3)

    def _rollout(self, state: np.ndarray) -> float:
        """Random playout from state."""
        current_state = state.copy()
        current_player = -1  # Opponent's turn

        for _ in range(9):  # Max 9 moves
            # Check if game is over
            winner = self._check_winner(current_state)
            if winner != 0:
                return 1.0 if winner == 1 else -1.0

            # Get valid actions
            flat_state = current_state.flatten()
            valid_actions = [i for i in range(9) if flat_state[i] == 0]

            if not valid_actions:
                return 0.0  # Draw

            # Random action
            action = random.choice(valid_actions)
            current_state = self._apply_action(current_state, action, current_player)
            current_player *= -1  # Switch player

        return 0.0  # Draw if no winner

    def _check_winner(self, state: np.ndarray) -> int:
        """Check if there's a winner. Returns 1, -1, or 0."""
        # Check rows
        for i in range(3):
            if abs(state[i].sum()) == 3:
                return int(state[i, 0])

        # Check columns
        for j in range(3):
            if abs(state[:, j].sum()) == 3:
                return int(state[0, j])

        # Check diagonals
        if abs(state.diagonal().sum()) == 3:
            return int(state[0, 0])
        if abs(np.fliplr(state).diagonal().sum()) == 3:
            return int(state[0, 2])

        return 0  # No winner

    def update(self, *args, **kwargs) -> None:
        """MCTS doesn't need updates."""
        pass

    def reset_episode(self) -> None:
        """MCTS has no episode state."""
        pass

    def save(self, filepath: str) -> None:
        """MCTS has no parameters to save."""
        print(f"MCTS agent doesn't need saving (no learned parameters)")

    def load(self, filepath: str) -> None:
        """MCTS has no parameters to load."""
        print(f"MCTS agent doesn't need loading (no learned parameters)")

    def __len__(self) -> int:
        """Return simulation count."""
        return self.n_simulations
