"""Experience replay buffer for DQN."""

import random
from collections import deque
from typing import List, Tuple
import numpy as np


class ReplayBuffer:
    """Replay buffer for storing experiences."""

    def __init__(self, capacity=10000, max_size=None, batch_size=64):
        # Support both 'capacity' and 'max_size' for compatibility
        size = max_size if max_size is not None else capacity
        self.buffer = deque(maxlen=size)
        self.capacity = size
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def add(self, state, action, reward, next_state, done):
        """Alias for push() for compatibility."""
        self.push(state, action, reward, next_state, done)

    def sample(self, batch_size=None):
        size = batch_size if batch_size is not None else self.batch_size
        batch = random.sample(self.buffer, size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size=None):
        """Check if buffer has enough samples."""
        size = batch_size if batch_size is not None else self.batch_size
        return len(self.buffer) >= size


if __name__ == "__main__":
    print("Testing ReplayBuffer...")
    buffer = ReplayBuffer(capacity=100)
    print(f"Buffer size: {len(buffer)}")
    print("✅ Import successful")
