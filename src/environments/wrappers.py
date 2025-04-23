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

class InvertedPendulumComparisonWrapper(gym.Wrapper):
    """
    A wrapper for the InvertedPendulum environment specifically for 
    long-term simulation comparison against an analytical model.
    
    Features:
    - Disables the default 'terminated' condition (pole falling too far).
    - Allows setting a specific initial state via `reset(initial_state=...)`.
    - *Attempts* to remove the angle limits on the hinge joint for comparison.
    """
    def __init__(self, env):
        super().__init__(env)
        print("Initialized InvertedPendulumComparisonWrapper.")
        
        # Attempt to remove hinge joint limits
        try:
            model = self.env.unwrapped.model
            hinge_index = -1
            for i in range(model.njnt):
                if model.joint(i).name == 'hinge':
                    hinge_index = i
                    break
            
            if hinge_index != -1:
                # Set limited flag to False (0)
                model.jnt_limited[hinge_index] = 0 
                # Optionally update range visually (though jnt_limited is key)
                model.jnt_range[hinge_index] = [-1e10, 1e10] 
                print(f"Wrapper: Successfully removed limits for joint '{model.joint(hinge_index).name}' (index {hinge_index})")
            else:
                print("Warning: Wrapper could not find 'hinge' joint to remove limits.")
        except Exception as e:
            print(f"Warning: Wrapper encountered exception trying to remove joint limits: {e}")

    def reset(self, *, seed=None, options=None, initial_state=None):
        """
        Resets the environment.
        If initial_state is provided (as [x, theta, x_dot, theta_dot]),
        it attempts to set the MuJoCo state accordingly.
        """
        # Call the parent reset first to ensure proper setup
        observation, info = super().reset(seed=seed, options=options)
        
        if initial_state is not None:
            initial_state = np.array(initial_state, dtype=np.float64)
            if len(initial_state) == 4:
                qpos = initial_state[:2] # [x, theta]
                qvel = initial_state[2:] # [x_dot, theta_dot]
                try:
                    # Access the underlying MuJoCo environment to set state
                    self.env.unwrapped.set_state(qpos, qvel)
                    # Re-get the observation *from the environment* after setting the state
                    observation = self.env.unwrapped._get_obs()
                    print(f"Successfully set initial state via wrapper: qpos={qpos}, qvel={qvel}")
                    print(f" -> Resulting observation: {observation}")
                except Exception as e:
                    print(f"Warning: Wrapper failed to set initial state: {e}. Using default reset state.")
                    # observation, info already contain the default reset results
            else:
                print("Warning: initial_state provided to wrapper has incorrect length. Using default reset state.")
                # observation, info already contain the default reset results

        return observation, info

    def step(self, action):
        """
        Steps the environment but forces the 'terminated' flag to False.
        Truncation based on time limit still applies.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Override termination signal for long simulation comparison
        terminated = False 
        
        return observation, reward, terminated, truncated, info 

# --- Wrapper for Double Pendulum Comparison --- 
class InvertedDoublePendulumComparisonWrapper(gym.Wrapper):
    """
    A wrapper for InvertedDoublePendulum-v5 for comparison.

    Features:
    - Allows setting initial state [x, th1, th2, x_d, th1_d, th2_d].
    - Returns the 6D physical state as the observation on reset() and step().
    - Disables the default 'terminated' condition.
    """
    def __init__(self, env):
        super().__init__(env)
        print("Initialized InvertedDoublePendulumComparisonWrapper.")
        
        # Store qpos/qvel sizes for convenience
        # Need to access the unwrapped env's model
        self.nq = self.env.unwrapped.model.nq
        self.nv = self.env.unwrapped.model.nv
        assert self.nq == 3, f"Expected nq=3, got {self.nq}"
        assert self.nv == 3, f"Expected nv=3, got {self.nv}"

        # Modify the observation space to reflect the 6D physical state
        # Assuming all states are unbounded for the wrapper
        inf = np.inf
        obs_low = np.array([-inf] * 6, dtype=np.float64)
        obs_high = np.array([inf] * 6, dtype=np.float64)
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float64)

    def _get_physical_state(self):
        """Extracts the 6D state [x, th1, th2, xd, th1d, th2d]."""
        # Access data from the unwrapped environment
        qpos = self.env.unwrapped.data.qpos # Shape (nq,)
        qvel = self.env.unwrapped.data.qvel # Shape (nv,)
        # Assuming qpos = [x, th1, th2] and qvel = [x_d, th1_d, th2_d]
        return np.concatenate([qpos, qvel]).astype(np.float64)

    def reset(self, *, seed=None, options=None, initial_state=None):
        """
        Resets the environment.
        If initial_state is provided (as 6D physical state),
        it sets the MuJoCo state accordingly.
        Returns the 6D physical state.
        """
        # Call the parent reset first
        _, info = super().reset(seed=seed, options=options)
        
        if initial_state is not None:
            initial_state = np.array(initial_state, dtype=np.float64)
            if len(initial_state) == 6:
                qpos = initial_state[:self.nq] # First nq elements
                qvel = initial_state[self.nq:] # Last nv elements
                try:
                    self.env.unwrapped.set_state(qpos, qvel)
                    print(f"Wrapper: Set initial state: qpos={qpos}, qvel={qvel}")
                except Exception as e:
                    print(f"Warning: Wrapper failed to set initial state: {e}. Using default reset state.")
            else:
                print(f"Warning: initial_state ({len(initial_state)}D) != 6D. Using default reset state.")

        # Return the physical state after reset (and potential state setting)
        physical_state = self._get_physical_state()
        return physical_state, info

    def step(self, action):
        """
        Steps the environment, forces 'terminated' to False, and returns the 6D physical state.
        """
        _, reward, terminated, truncated, info = self.env.step(action)
        
        # Override termination signal
        terminated = False 
        
        # Get the physical state after stepping
        physical_state = self._get_physical_state()
        
        return physical_state, reward, terminated, truncated, info


# --- Wrapper for setting initial state for swing-up --- 
class SwingUpResetWrapper(gym.Wrapper):
    """Simple wrapper to reset the pendulum environment to the bottom state."""
    def __init__(self, env):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped # Get the unwrapped env
        self.initial_qpos = np.array([0.0, -np.pi]) # [cart_pos, pole_angle]
        self.initial_qvel = np.array([0.0, 0.0]) # [cart_vel, pole_ang_vel]

    def reset(self, **kwargs):
        # Reset the base environment first (might randomize slightly)
        obs, info = self.env.reset(**kwargs)
        
        # Force the specific initial state for swing-up
        try:
            self.unwrapped_env.set_state(self.initial_qpos, self.initial_qvel)
            # Get the observation corresponding to the reset state
            obs = self.unwrapped_env._get_obs()
        except Exception as e:
            print(f"Warning: SwingUpResetWrapper could not set initial state: {e}")
            # Return the observation from the initial reset call
            
        return obs, info 