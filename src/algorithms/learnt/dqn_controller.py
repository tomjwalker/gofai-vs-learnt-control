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
        Returns separate arrays for each component.
        """

        if not self.can_sample(batch_size):
            raise ValueError("Not enough transitions in the buffer to sample a batch.")

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays with consistent dtypes
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)  # PyTorch expects LongTensor for indices
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        
        return states, actions, rewards, next_states, dones


    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    A simple MLP that outputs Q-values for each discrete action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            # nn.Linear initialises with PyTorch defaults:
            # - Weights: Kaiming uniform (suitable for ReLU)
            # - Biases: Uniform in ±1/sqrt(fan_in)
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (batch_size, state_dim) → (batch_size, action_dim)."""
        return self.net(x)


class DQNController:
    """A Deep‑Q controller using epsilon‑greedy exploration and a target network."""

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

        # Online Q‑network
        self.q_net = QNetwork(state_dim, action_dim, hidden_dims)
        # Target network starts as an exact copy
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer & hyper‑parameters
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval

        # Misc.
        self.action_dim = action_dim
        self.learn_step_counter = 0

    # ------------------------------------------------
    #  Interaction
    # ------------------------------------------------
    def select_action(self, state, eval_mode=False):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state: The current state observation.
            eval_mode (bool): If True, always select the greedy action (epsilon=0).

        Returns:
            int: The selected action.
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------
    #  Learning step
    # ------------------------------------------------
    def update(self):
        """
        Updates the Q-network using a batch sampled from the replay buffer.
        Decreases epsilon after each update.
        Updates the target network periodically.
        """
        if not self.buffer.can_sample(self.batch_size):
            return # Not enough samples in buffer yet

        self.learn_step_counter += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions).unsqueeze(1)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.BoolTensor(dones).unsqueeze(1)
        not_done      = (~dones_t).float()  # 1‑for‑not‑terminal

        # Current Q(s,a)
        q_sa = self.q_net(states_t).gather(1, actions_t)

        # Compute target: r + γ * max_a' Q_target(s',a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]  # (B,1)
            td_target  = rewards_t + self.gamma * not_done * max_next_q

        # MSE loss
        loss = F.mse_loss(q_sa, td_target)

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network
        if self.learn_step_counter % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon (only if not in pure evaluation mode)
        if self.epsilon > self.epsilon_min:
           self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))
