import gymnasium as gym
import numpy as np

class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    Wraps an environment with a continuous Box action space to make it discrete.

    Maps discrete actions (0 to n_bins-1) to continuous values within the
    original Box action space boundaries.
    """
    def __init__(self, env, n_bins):
        """
        Initializes the wrapper.

        Args:
            env: The environment to wrap (must have a Box action space).
            n_bins: The number of discrete actions to create.
        """
        super().__init__(env)
        
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("DiscretizeActionWrapper requires an environment with a Box action space.")
        if not len(env.action_space.shape) == 1:
             raise ValueError(f"DiscretizeActionWrapper currently only supports 1D Box action spaces, got shape {env.action_space.shape}")

        self.n_bins = n_bins
        self.original_action_space = env.action_space
        self.low = self.original_action_space.low
        self.high = self.original_action_space.high

        # Redefine the action space to be discrete
        self.action_space = gym.spaces.Discrete(self.n_bins)

        print(f"Applied DiscretizeActionWrapper: Mapping Discrete({n_bins}) actions to Box{self.original_action_space.shape} space [{self.low[0]:.2f}, {self.high[0]:.2f}]")


    def action(self, action):
        """
        Maps a discrete action index back to a continuous value.

        Args:
            action: The discrete action index (0 to n_bins-1).

        Returns:
            A numpy array containing the corresponding continuous action value.
        """
        # Linearly interpolate between low and high based on the discrete action index
        # For n_bins=1, output the middle value
        if self.n_bins == 1:
            continuous_action = self.low + (self.high - self.low) / 2.0
        else:
            continuous_action = self.low + (self.high - self.low) * action / (self.n_bins - 1)
        
        # Ensure the action is within the original bounds (due to potential float precision)
        # And return as a numpy array matching original space dtype and shape
        return np.clip(continuous_action, self.low, self.high).astype(self.original_action_space.dtype) 