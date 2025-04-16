import random
from collections import deque
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """
    Fixed-capacity buffer that stores tuples of (state, action, reward, next_state, done).
    """
    def __init__(self, capacity: int) -> None:
        # Use deque as it facilitates fixed-capacity appends and pops
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)


    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Save a transition tuple to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def can_sample(self, batch_size: int) -> bool:
        """
        Check if the buffer has enough transitions to sample a batch.
        """
        return len(self.buffer) >= batch_size


    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.
        Returns separate lists (or arrays) for each component.
        """

        if not self.can_sample(batch_size):
            raise ValueError("Not enough transitions in the buffer to sample a batch.")

        # Randomly sample batch_size transitions from the buffer
        batch = random.sample(self.buffer, batch_size)

        # batch is a list of tuples (state, action, reward, next_state, done)
        # we want to return separate lists for each component. 
        # *batch unpacks the list into separate arguments. zip groups matching elements from each tuple into lists.
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        return states, actions, rewards, next_states, dones


    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    A simple MLP that outputs Q-values for each discrete action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        """
        Define the NN architecture.
        """
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:

            # Add dense layer; layers are initialised automatically for nn.Linear with PyTorch defaults:
            # - Weights: Kaiming uniform (suitable for ReLU)
            # - Biases: Uniform in ±1/sqrt(fan_in)
            layers.append(nn.Linear(input_dim, hidden_dim))

            # Add ReLU activation function
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        # Define the network
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-network.

        Args:
            x (torch.Tensor): A batch of input states with shape (batch_size, state_dim).
                            Even if using a single state, it should be shaped as (1, state_dim).

        Returns:
            torch.Tensor: Q-values for each action, one row per input state.
                        Shape: (batch_size, action_dim)

        Notes:
            - This method is called automatically when you do `q_values = model(state)`.
            - The shape convention (batch_size, state_dim) is standard in PyTorch, even for single inputs.
            - Use `model.eval()` and `with torch.no_grad()` during inference to disable gradients.
        """
        return self.net(x)


class DQNController:
    """
    A DQN controller that uses a Q-network to select actions.
    """
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            hidden_dims: Tuple[int, ...] = (64, 64),
            lr: float = 1e-3, 
            gamma: float = 0.99, 
            epsilon: float = 1.0, 
            epsilon_min: float = 0.01,
            epsilon_decay: float = 1e-4,
            buffer_capacity: int = 100000,
            batch_size: int = 64,
            target_update_interval: int = 1000
        ) -> None:
        """
        Initialize the DQN controller.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Tuple of hidden layer dimensions for the Q-network
            lr: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            buffer_capacity: Maximum size of the replay buffer
            batch_size: Number of transitions to sample for training
            target_update_interval: Number of steps between target network updates
        """
        super().__init__()

