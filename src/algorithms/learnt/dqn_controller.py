from __future__ import annotations

import random
from collections import deque
from typing import TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

StateArray: TypeAlias = np.ndarray
Transition = tuple[StateArray, np.ndarray, float, StateArray, bool]


class ReplayBuffer:
    """Fixed-capacity replay buffer for `(state, action, reward, next_state, done)` tuples."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: StateArray,
        action: np.ndarray,
        reward: float,
        next_state: StateArray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def sample(
        self,
        batch_size: int,
    ) -> tuple[StateArray, np.ndarray, np.ndarray, StateArray, np.ndarray]:
        if not self.can_sample(batch_size):
            raise ValueError("Not enough transitions in the buffer to sample a batch.")

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Simple MLP that outputs one Q-value per discrete action."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNController:
    """Deep Q-learning controller with epsilon-greedy exploration and a target network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1e-4,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_interval: int = 1000,
    ) -> None:
        self.q_net = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.action_dim = action_dim
        self.learn_step_counter = 0

    def select_action(self, state: StateArray, eval_mode: bool = False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: StateArray,
        action: int | np.ndarray,
        reward: float,
        next_state: StateArray,
        done: bool,
    ) -> None:
        action_array = np.array([action], dtype=np.int64)
        self.buffer.push(state, action_array, reward, next_state, done)

    def update(self) -> None:
        if not self.buffer.can_sample(self.batch_size):
            return

        self.learn_step_counter += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.as_tensor(dones, dtype=torch.bool).unsqueeze(1)
        not_done = (~dones_tensor).float()

        q_sa = self.q_net(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_tensor).max(dim=1, keepdim=True)[0]
            td_target = rewards_tensor + self.gamma * not_done * max_next_q

        loss = F.mse_loss(q_sa, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step_counter % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))
