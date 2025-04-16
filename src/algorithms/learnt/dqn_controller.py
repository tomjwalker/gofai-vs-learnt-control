import random
from collections import deque
import numpy as np

class ReplayBuffer:
    """
    Fixed-capacity buffer that stores tuples of (state, action, reward, next_state, done).
    """
    def __init__(self, capacity: int):
        # Use deque as it facilitates fixed-capacity appends and pops
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)


    def push(self, state, action, reward, next_state, done):
        """
        Save a transition tuple to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def can_sample(self, batch_size: int):
        """
        Check if the buffer has enough transitions to sample a batch.
        """
        return len(self.buffer) >= batch_size


    def sample(self, batch_size: int):
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


    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # ===== Test the ReplayBuffer class =====
    
    # Instantiate the replay buffer
    buffer = ReplayBuffer(capacity=3)
    print("Test 1: Initialization")
    assert len(buffer) == 0, "Buffer should be empty initially"
    assert buffer.capacity == 3, "Buffer capacity should be 3"
    print("✓ Passed initialization test")

    # Push one transition to the buffer
    state = np.array([1.0, 2.0, 3.0, 4.0])
    action = np.array([0.5])
    reward = 1.0
    next_state = np.array([1.1, 2.1, 3.1, 4.1])
    done = False
    
    buffer.push(state, action, reward, next_state, done)
    print("\nTest 2: Single push")
    assert len(buffer) == 1, "Buffer should contain one transition"
    print("✓ Passed single push test")

    # Test the can_sample method
    print("\nTest 3: can_sample method")
    assert not buffer.can_sample(2), "Should not be able to sample 2 transitions"
    assert buffer.can_sample(1), "Should be able to sample 1 transition"
    print("✓ Passed can_sample test")

    # Test the sample method
    print("\nTest 4: sample method")
    sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = buffer.sample(1)
    assert np.array_equal(sampled_states[0], state), "Sampled state should match pushed state"
    assert np.array_equal(sampled_actions[0], action), "Sampled action should match pushed action"
    assert sampled_rewards[0] == reward, "Sampled reward should match pushed reward"
    assert np.array_equal(sampled_next_states[0], next_state), "Sampled next_state should match pushed next_state"
    assert sampled_dones[0] == done, "Sampled done should match pushed done"
    print("✓ Passed sample test")

    # Test the __len__ method
    print("\nTest 5: __len__ method")
    assert len(buffer) == 1, "Buffer length should be 1"
    print("✓ Passed __len__ test")

    # Once capacity is reached, test that the oldest transitions are discarded
    print("\nTest 6: Capacity limit")
    # Add two more transitions
    buffer.push(state * 2, action * 2, reward * 2, next_state * 2, not done)
    buffer.push(state * 3, action * 3, reward * 3, next_state * 3, done)
    assert len(buffer) == 3, "Buffer should be at capacity"
    
    # Add one more, should discard the oldest
    buffer.push(state * 4, action * 4, reward * 4, next_state * 4, not done)
    assert len(buffer) == 3, "Buffer should still be at capacity"
    
    # Sample all transitions and verify the oldest was discarded
    all_states, _, _, _, _ = buffer.sample(3)
    assert not np.array_equal(all_states[0], state), "Oldest transition should have been discarded"
    print("✓ Passed capacity limit test")
    
    print("\nAll ReplayBuffer tests passed successfully!")
    