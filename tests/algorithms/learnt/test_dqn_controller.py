import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

import pytest
import numpy as np
from src.algorithms.learnt.dqn_controller import ReplayBuffer

def test_replay_buffer_initialization():
    """Test that the replay buffer initializes correctly."""
    buffer = ReplayBuffer(capacity=3)
    assert len(buffer) == 0
    assert buffer.capacity == 3

def test_replay_buffer_single_push():
    """Test that a single transition can be pushed to the buffer."""
    buffer = ReplayBuffer(capacity=3)
    state = np.array([1.0, 2.0, 3.0, 4.0])
    action = np.array([0.5])
    reward = 1.0
    next_state = np.array([1.1, 2.1, 3.1, 4.1])
    done = False
    
    buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 1

def test_replay_buffer_can_sample():
    """Test the can_sample method."""
    buffer = ReplayBuffer(capacity=3)
    state = np.array([1.0, 2.0, 3.0, 4.0])
    action = np.array([0.5])
    reward = 1.0
    next_state = np.array([1.1, 2.1, 3.1, 4.1])
    done = False
    
    buffer.push(state, action, reward, next_state, done)
    assert not buffer.can_sample(2)
    assert buffer.can_sample(1)

def test_replay_buffer_sample():
    """Test that transitions can be sampled from the buffer."""
    buffer = ReplayBuffer(capacity=3)
    state = np.array([1.0, 2.0, 3.0, 4.0])
    action = np.array([0.5])
    reward = 1.0
    next_state = np.array([1.1, 2.1, 3.1, 4.1])
    done = False
    
    buffer.push(state, action, reward, next_state, done)
    sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = buffer.sample(1)
    
    assert np.array_equal(sampled_states[0], state)
    assert np.array_equal(sampled_actions[0], action)
    assert sampled_rewards[0] == reward
    assert np.array_equal(sampled_next_states[0], next_state)
    assert sampled_dones[0] == done

def test_replay_buffer_capacity_limit():
    """Test that the buffer maintains its capacity limit."""
    buffer = ReplayBuffer(capacity=3)
    state = np.array([1.0, 2.0, 3.0, 4.0])
    action = np.array([0.5])
    reward = 1.0
    next_state = np.array([1.1, 2.1, 3.1, 4.1])
    done = False
    
    # Fill the buffer
    buffer.push(state, action, reward, next_state, done)
    buffer.push(state * 2, action * 2, reward * 2, next_state * 2, not done)
    buffer.push(state * 3, action * 3, reward * 3, next_state * 3, done)
    assert len(buffer) == 3
    
    # Add one more, should discard the oldest
    buffer.push(state * 4, action * 4, reward * 4, next_state * 4, not done)
    assert len(buffer) == 3
    
    # Verify the oldest was discarded
    all_states, _, _, _, _ = buffer.sample(3)
    assert not np.array_equal(all_states[0], state)

def test_replay_buffer_empty_sample():
    """Test that sampling from an empty buffer raises an error."""
    buffer = ReplayBuffer(capacity=3)
    with pytest.raises(ValueError):
        buffer.sample(1)

def test_replay_buffer_invalid_sample_size():
    """Test that sampling more than buffer size raises an error."""
    buffer = ReplayBuffer(capacity=3)
    state = np.array([1.0, 2.0, 3.0, 4.0])
    action = np.array([0.5])
    reward = 1.0
    next_state = np.array([1.1, 2.1, 3.1, 4.1])
    done = False
    
    buffer.push(state, action, reward, next_state, done)
    with pytest.raises(ValueError):
        buffer.sample(2) 