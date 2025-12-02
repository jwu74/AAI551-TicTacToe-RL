"""MCTS tree node implementation."""

import numpy as np
import math
from typing import List, Optional, Tuple


class MCTSNode:
    """Node in MCTS tree."""

    def __init__(self,
                 state: np.ndarray,
                 parent: Optional['MCTSNode'] = None,
                 action: Optional[int] = None,
                 valid_actions: Optional[List[int]] = None):
        self.state = state.copy() if isinstance(state, np.ndarray) else np.array(state)
        self.parent = parent
        self.action = action
        self.children = []

        # Statistics
        self.visits = 0
        self.value = 0.0

        # Available actions
        if valid_actions is None:
            # Find valid actions (empty positions)
            self.untried_actions = self._get_valid_actions()
        else:
            self.untried_actions = valid_actions.copy()

    def _get_valid_actions(self) -> List[int]:
        """
        Get list of valid actions from current state.

        Returns:
            List[int]: Indices of empty positions (valid moves)
        """
        flat_state = self.state.flatten()
        return [i for i in range(9) if flat_state[i] == 0]

    def is_fully_expanded(self) -> bool:
        """
        Check if all actions from this node have been tried.

        Returns:
            bool: True if fully expanded, False otherwise
        """
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """
        Check if this node represents a terminal state.

        Returns:
            bool: True if game is over, False otherwise
        """
        # Check if no valid moves left
        return len(self._get_valid_actions()) == 0

    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """
        Select best child using UCB1 (Upper Confidence Bound) formula.

        UCB1 = (value / visits) + c * sqrt(ln(parent_visits) / visits)

        Args:
            exploration_constant (float): Exploration parameter (default: sqrt(2))

        Returns:
            MCTSNode: Child with highest UCB1 value

        Raises:
            ValueError: If node has no children
        """
        if not self.children:
            raise ValueError("Cannot select best child from node with no children")

        # Calculate UCB1 for each child
        ucb_values = []
        for child in self.children:
            if child.visits == 0:
                ucb = float('inf')  # Unvisited children have infinite value
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * math.sqrt(
                    math.log(self.visits) / child.visits
                )
                ucb = exploitation + exploration
            ucb_values.append(ucb)

        # Return child with maximum UCB value
        best_idx = np.argmax(ucb_values)
        return self.children[best_idx]

    def expand(self, action: int, next_state: np.ndarray) -> 'MCTSNode':
        """
        Expand tree by creating a new child node.

        Args:
            action (int): Action to take
            next_state (np.ndarray): Resulting state after action

        Returns:
            MCTSNode: Newly created child node

        Raises:
            ValueError: If action is not in untried_actions
        """
        if action not in self.untried_actions:
            raise ValueError(f"Action {action} not in untried actions: {self.untried_actions}")

        # Remove action from untried list
        self.untried_actions.remove(action)

        # Create child node
        child = MCTSNode(
            state=next_state,
            parent=self,
            action=action
        )

        # Add to children
        self.children.append(child)

        return child

    def update(self, reward: float) -> None:
        """
        Update node statistics with simulation result.

        Args:
            reward (float): Reward from simulation
        """
        self.visits += 1
        self.value += reward

    def backpropagate(self, reward: float) -> None:
        """
        Backpropagate reward up the tree to root.

        Args:
            reward (float): Reward to propagate
        """
        self.update(reward)
        if self.parent is not None:
            # Alternate reward sign for opponent's perspective
            self.parent.backpropagate(-reward)

    def best_action(self) -> int:
        """
        Get action leading to most visited child (exploitation).

        Returns:
            int: Best action based on visit counts

        Raises:
            ValueError: If node has no children
        """
        if not self.children:
            raise ValueError("Cannot determine best action from node with no children")

        # Select child with most visits
        visits = [child.visits for child in self.children]
        best_idx = np.argmax(visits)
        return self.children[best_idx].action

    def get_action_probs(self, temperature: float = 1.0) -> List[Tuple[int, float]]:
        """
        Get probability distribution over actions based on visit counts.

        Args:
            temperature (float): Temperature parameter for softmax (default: 1.0)
                               Higher = more exploration, Lower = more exploitation

        Returns:
            List[Tuple[int, float]]: List of (action, probability) pairs
        """
        if not self.children:
            return []

        # Get visit counts
        visits = np.array([child.visits for child in self.children])
        actions = [child.action for child in self.children]

        # Apply temperature
        if temperature == 0:
            # Greedy selection
            probs = np.zeros_like(visits, dtype=float)
            probs[np.argmax(visits)] = 1.0
        else:
            # Softmax with temperature
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)

        return list(zip(actions, probs))

    def __str__(self) -> str:
        """
        String representation of node.

        Returns:
            str: Node information
        """
        avg_value = self.value / self.visits if self.visits > 0 else 0
        return (f"MCTSNode(action={self.action}, visits={self.visits}, "
                f"avg_value={avg_value:.3f}, children={len(self.children)})")

    def __repr__(self) -> str:
        """
        Official string representation.

        Returns:
            str: Node representation
        """
        return self.__str__()

    def __len__(self) -> int:
        """
        Get number of child nodes.

        Returns:
            int: Number of children
        """
        return len(self.children)

    def print_tree(self, depth: int = 0, max_depth: int = 3) -> None:
        """
        Print tree structure (for debugging).

        Args:
            depth (int): Current depth in tree
            max_depth (int): Maximum depth to print
        """
        if depth > max_depth:
            return

        indent = "  " * depth
        avg_value = self.value / self.visits if self.visits > 0 else 0
        print(f"{indent}Action: {self.action}, Visits: {self.visits}, "
              f"Avg Value: {avg_value:.3f}")

        for child in self.children:
            child.print_tree(depth + 1, max_depth)


