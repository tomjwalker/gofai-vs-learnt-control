import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

import pytest
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.algorithms.learnt.dqn_controller import ReplayBuffer, QNetwork, DQNController

# ===== ReplayBuffer Tests =====
class TestReplayBuffer:
    """Tests for the ReplayBuffer class."""
    
    def test_replay_buffer_initialization(self):
        """Test that the replay buffer initializes correctly."""
        buffer = ReplayBuffer(capacity=3)
        assert len(buffer) == 0
        assert buffer.capacity == 3

    def test_replay_buffer_single_push(self):
        """Test that a single transition can be pushed to the buffer."""
        buffer = ReplayBuffer(capacity=3)
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        action = np.array([0], dtype=np.int64)  # Use integer action
        reward = 1.0
        next_state = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        assert len(buffer) == 1

    def test_replay_buffer_can_sample(self):
        """Test the can_sample method."""
        buffer = ReplayBuffer(capacity=3)
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        action = np.array([0], dtype=np.int64)  # Use integer action
        reward = 1.0
        next_state = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        assert not buffer.can_sample(2)
        assert buffer.can_sample(1)

    def test_replay_buffer_sample(self):
        """Test that transitions can be sampled from the buffer."""
        buffer = ReplayBuffer(capacity=3)
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        action = np.array([0], dtype=np.int64)  # Use integer action
        reward = 1.0
        next_state = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = buffer.sample(1)
        
        assert np.array_equal(sampled_states[0], state)
        assert np.array_equal(sampled_actions[0], action)
        assert sampled_rewards[0] == reward
        assert np.array_equal(sampled_next_states[0], next_state)
        assert sampled_dones[0] == done

    def test_replay_buffer_capacity_limit(self):
        """Test that the buffer maintains its capacity limit."""
        buffer = ReplayBuffer(capacity=3)
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        action = np.array([0], dtype=np.int64)  # Use integer action
        reward = 1.0
        next_state = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)
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

    def test_replay_buffer_empty_sample(self):
        """Test that sampling from an empty buffer raises an error."""
        buffer = ReplayBuffer(capacity=3)
        with pytest.raises(ValueError):
            buffer.sample(1)

    def test_replay_buffer_invalid_sample_size(self):
        """Test that sampling more than buffer size raises an error."""
        buffer = ReplayBuffer(capacity=3)
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        action = np.array([0], dtype=np.int64)  # Use integer action
        reward = 1.0
        next_state = np.array([1.1, 2.1, 3.1, 4.1], dtype=np.float32)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        with pytest.raises(ValueError):
            buffer.sample(2)


# ===== QNetwork Tests =====
class TestQNetwork:
    """Tests for the QNetwork class."""
    
    def test_qnetwork_initialization(self):
        """Test that the Q-net initializes correctly with different architectures."""
        # Test with default hidden dimensions
        network = QNetwork(state_dim=4, action_dim=2)
        assert isinstance(network, nn.Module)
        assert len(list(network.parameters())) > 0
        
        # Test with custom hidden dimensions
        network = QNetwork(state_dim=4, action_dim=2, hidden_dims=(32, 64, 32))
        assert isinstance(network, nn.Module)
        assert len(list(network.parameters())) > 0

    def test_qnetwork_forward_pass(self):
        """Test the forward pass of the Q-net."""
        network = QNetwork(state_dim=4, action_dim=2)
        
        # Test single state input
        state = torch.randn(1, 4)  # (batch_size=1, state_dim=4)
        q_values = network(state)
        assert q_values.shape == (1, 2)  # (batch_size=1, action_dim=2)
        
        # Test batch input
        states = torch.randn(32, 4)  # (batch_size=32, state_dim=4)
        q_values = network(states)
        assert q_values.shape == (32, 2)  # (batch_size=32, action_dim=2)

    def test_qnetwork_forward_pass_edge_cases(self):
        """Test edge cases in the forward pass of the Q-net."""
        network = QNetwork(state_dim=4, action_dim=2)
        
        # Test with zero input
        state = torch.zeros(1, 4)
        q_values = network(state)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()
        
        # Test with large input
        state = torch.ones(1, 4) * 1000
        q_values = network(state)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_qnetwork_gradient_flow(self):
        """Test that gradients can flow through the Q-net."""
        network = QNetwork(state_dim=4, action_dim=2)
        state = torch.randn(1, 4, requires_grad=True)
        q_values = network(state)
        
        # Compute a dummy loss and backpropagate
        loss = q_values.sum()
        loss.backward()
        
        # Check that gradients were computed
        for param in network.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_qnetwork_device_handling(self):
        """Test that the Q-net can handle different devices."""
        network = QNetwork(state_dim=4, action_dim=2)
        
        # Test on CPU
        state = torch.randn(1, 4)
        q_values_cpu = network(state)
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            network = network.cuda()
            state = state.cuda()
            q_values_gpu = network(state)
            assert torch.allclose(q_values_cpu, q_values_gpu.cpu())

    def test_qnetwork_parameter_initialization(self):
        """Test that Q-net parameters are initialized correctly."""
        network = QNetwork(state_dim=4, action_dim=2)
        
        # Check that weights are initialized (not zero)
        for name, param in network.named_parameters():
            if 'weight' in name:
                assert not torch.allclose(param, torch.zeros_like(param))
            
            # Check that biases are initialized (not zero)
            if 'bias' in name:
                assert not torch.allclose(param, torch.zeros_like(param))

    def test_qnetwork_eval_mode(self):
        """Test that the Q-net behaves correctly in eval mode."""
        network = QNetwork(state_dim=4, action_dim=2)
        network.eval()
        
        # Test that gradients are not computed in eval mode
        with torch.no_grad():
            state = torch.randn(1, 4)
            q_values = network(state)
            assert not q_values.requires_grad 

# ===== DQNController Tests =====
class TestDQNController:
    """Tests for the DQNController class."""

    @pytest.fixture
    def dqn_config(self):
        """Provides a basic DQN configuration for testing."""
        return {
            "state_dim": 4,
            "action_dim": 2,
            "hidden_dims": (16, 16),
            "lr": 1e-3,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 1e-4,
            "buffer_capacity": 1000,
            "batch_size": 32,
            "target_update_interval": 100
        }

    def test_dqn_initialization(self, dqn_config):
        """Test DQNController initialization."""
        controller = DQNController(**dqn_config)

        # Check q_net and target_net
        assert isinstance(controller.q_net, QNetwork)
        assert isinstance(controller.target_net, QNetwork)
        assert controller.q_net is not controller.target_net # Ensure they are separate instances
        
        # Check if target net parameters are initially identical to q_net
        q_params = list(controller.q_net.parameters())
        target_params = list(controller.target_net.parameters())
        for q_p, t_p in zip(q_params, target_params):
            assert torch.equal(q_p, t_p)
            
        # Check if gradients are enabled/disabled appropriately
        assert all(p.requires_grad for p in controller.q_net.parameters())
        # Target network should be in eval mode (for testing)
        assert controller.target_net.training == False

        # Check optimizer
        assert isinstance(controller.optimizer, optim.Adam)
        assert controller.optimizer.defaults['lr'] == dqn_config['lr']

        # Check buffer and hyperparameters
        assert isinstance(controller.buffer, ReplayBuffer)
        assert controller.buffer.capacity == dqn_config['buffer_capacity']
        assert controller.batch_size == dqn_config['batch_size']
        assert controller.gamma == dqn_config['gamma']
        assert controller.epsilon == dqn_config['epsilon']
        assert controller.epsilon_min == dqn_config['epsilon_min']
        assert controller.epsilon_decay == dqn_config['epsilon_decay']
        assert controller.target_update_interval == dqn_config['target_update_interval']
        assert controller.action_dim == dqn_config['action_dim']
        assert controller.learn_step_counter == 0

    def test_select_action_epsilon_greedy(self, dqn_config):
        """Test epsilon-greedy action selection."""
        # Test with epsilon=1.0 (pure exploration)
        dqn_config['epsilon'] = 1.0
        controller = DQNController(**dqn_config)
        state = np.random.rand(dqn_config['state_dim'])
        
        # With epsilon=1.0, should always choose random action
        actions = [controller.select_action(state) for _ in range(100)]
        # Verify actions are within valid range
        assert all(0 <= a < dqn_config['action_dim'] for a in actions)
        # With enough samples, should see different actions (randomness)
        assert len(set(actions)) > 1, "Random actions should have variety"

        # Test with epsilon=0.0 (pure exploitation)
        dqn_config['epsilon'] = 0.0
        controller = DQNController(**dqn_config)
        state = np.random.rand(dqn_config['state_dim'])
        
        # With epsilon=0.0, should always choose greedy action
        action = controller.select_action(state)
        # Verify greedy action
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            expected_action = controller.q_net(state_tensor).argmax(dim=1).item()
        assert action == expected_action

    def test_store_transition(self, dqn_config):
        """Test storing transitions in the replay buffer."""
        controller = DQNController(**dqn_config)
        assert len(controller.buffer) == 0
        
        state = np.random.rand(dqn_config['state_dim'])
        action = 0  # fixed action for testing
        reward = 1.0
        next_state = np.random.rand(dqn_config['state_dim'])
        done = False
        
        controller.store_transition(state, action, reward, next_state, done)
        assert len(controller.buffer) == 1

    def test_update_no_op_when_buffer_insufficient(self, dqn_config):
        """Test that update does nothing if the buffer is not full enough."""
        dqn_config['batch_size'] = 5
        controller = DQNController(**dqn_config)
        
        # Add fewer transitions than batch_size
        for _ in range(dqn_config['batch_size'] - 1):
            state = np.random.rand(dqn_config['state_dim'])
            action = 0  # fixed action for testing
            reward = 1.0
            next_state = np.random.rand(dqn_config['state_dim'])
            done = False
            controller.store_transition(state, action, reward, next_state, done)
        
        # Save initial state for comparison
        initial_q_params = [p.clone().detach() for p in controller.q_net.parameters()]
        initial_epsilon = controller.epsilon
        initial_learn_steps = controller.learn_step_counter

        # Call update
        controller.update()

        # Verify no parameters changed
        for p_initial, p_updated in zip(initial_q_params, controller.q_net.parameters()):
            assert torch.allclose(p_initial, p_updated)
        # Verify epsilon didn't decay
        assert controller.epsilon == initial_epsilon
        # Verify learn step counter didn't increment
        assert controller.learn_step_counter == initial_learn_steps

    def test_update_logic(self, dqn_config):
        """Test the core update logic (loss calculation, optimizer step)."""
        dqn_config['batch_size'] = 2
        dqn_config['buffer_capacity'] = 10 # Small buffer for faster filling
        dqn_config['epsilon'] = 0.1 # Use non-zero epsilon to test decay
        controller = DQNController(**dqn_config)

        # Fill buffer sufficiently
        for _ in range(dqn_config['batch_size'] * 2):
            state = np.random.rand(dqn_config['state_dim'])
            action = 0 # fixed action for testing
            reward = np.random.rand()
            next_state = np.random.rand(dqn_config['state_dim'])
            done = bool(np.random.choice([True, False]))
            controller.store_transition(state, action, reward, next_state, done)
        
        assert len(controller.buffer) >= dqn_config['batch_size']

        # Capture initial state
        initial_q_params = [p.clone().detach() for p in controller.q_net.parameters()]
        initial_epsilon = controller.epsilon
        initial_learn_steps = controller.learn_step_counter

        # Perform update
        controller.update()

        # Check that parameters were updated (loss was non-zero and optimizer stepped)
        updated = False
        for p_initial, p_updated in zip(initial_q_params, controller.q_net.parameters()):
            if not torch.allclose(p_initial, p_updated):
                updated = True
                break
        assert updated, "Q-network parameters should have been updated after a learning step."

        # Check epsilon decay - only if initial epsilon > min
        if initial_epsilon > controller.epsilon_min:
            assert controller.epsilon < initial_epsilon
            assert controller.epsilon >= dqn_config['epsilon_min']
            expected_epsilon = max(initial_epsilon * (1 - dqn_config['epsilon_decay']), dqn_config['epsilon_min'])
            assert abs(controller.epsilon - expected_epsilon) < 1e-9

        # Check learn step counter increment
        assert controller.learn_step_counter == initial_learn_steps + 1

    def test_target_network_update(self, dqn_config):
        """Test that the target network updates periodically."""
        dqn_config['batch_size'] = 2
        dqn_config['buffer_capacity'] = 50 # Needs enough capacity for multiple updates
        dqn_config['target_update_interval'] = 5
        controller = DQNController(**dqn_config)

        # Fill buffer
        for _ in range(dqn_config['batch_size'] * 2):
            state = np.random.rand(dqn_config['state_dim'])
            action = 0 # fixed action for testing
            reward = 1.0
            next_state = np.random.rand(dqn_config['state_dim'])
            done = False
            controller.store_transition(state, action, reward, next_state, done)

        # Run updates up to (but not including) the target update interval
        for _ in range(dqn_config['target_update_interval'] - 1):
            controller.update()

        # Target network should NOT have updated yet - parameters should differ
        # Because q_net has been updated but target_net hasn't
        q_params = list(controller.q_net.parameters())
        target_params = list(controller.target_net.parameters())
        
        # Check at least one parameter differs
        params_differ = False
        for q_p, t_p in zip(q_params, target_params):
            if not torch.allclose(q_p.data, t_p.data):
                params_differ = True
                break
                
        assert params_differ, "Q-network should have changed, but target should not have synced yet."

        # Run one more update to trigger target network sync
        controller.update()
        assert controller.learn_step_counter == dqn_config['target_update_interval']

        # Verify that target network is now updated
        # The target_net should match q_net after sync
        q_net_dict = controller.q_net.state_dict()
        target_net_dict = controller.target_net.state_dict()
        
        # Check each key in the state dict
        for key in q_net_dict:
            assert torch.allclose(q_net_dict[key], target_net_dict[key]), f"Target network sync failed for {key}"

    def test_epsilon_decay(self, dqn_config):
        """Test epsilon decay logic including minimum value."""
        dqn_config['epsilon'] = 0.5
        dqn_config['epsilon_min'] = 0.1
        dqn_config['epsilon_decay'] = 0.1 # Faster decay for testing
        dqn_config['batch_size'] = 2
        dqn_config['buffer_capacity'] = 10
        controller = DQNController(**dqn_config)

        # Fill buffer
        for _ in range(dqn_config['batch_size'] * 2):
            state = np.random.rand(dqn_config['state_dim'])
            action = 0 # fixed action for testing
            reward = 1.0
            next_state = np.random.rand(dqn_config['state_dim'])
            done = False
            controller.store_transition(state, action, reward, next_state, done)

        # Track epsilon decay
        expected_epsilon = dqn_config['epsilon']
        for _ in range(10): # Simulate multiple updates
            controller.update()
            expected_epsilon = max(expected_epsilon * (1 - dqn_config['epsilon_decay']), dqn_config['epsilon_min'])
            assert abs(controller.epsilon - expected_epsilon) < 1e-9 # Use tolerance for float comparison

        # Ensure epsilon doesn't go below min
        assert controller.epsilon >= dqn_config['epsilon_min']
        
        # Run many more updates to ensure it stops at min
        for _ in range(100):
            controller.update()
        assert abs(controller.epsilon - dqn_config['epsilon_min']) < 1e-9 