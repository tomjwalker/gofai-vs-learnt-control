"""
Custom Gymnasium environment wrapper for the Double Pendulum Swing-Up task.

This module defines a wrapper `DoublePendulumSwingUp` around the standard
Gymnasium `InvertedDoublePendulum-v5` environment. The goal is modified from
simply balancing the poles upright to starting with both poles hanging down
and actively swinging them up to the balanced position.

Modifications from the base `InvertedDoublePendulum-v5` environment:

1.  **Initial State (`reset` method):**
    *   The starting angles for both poles (`qpos[1]` and `qpos[2]`) are modified.
        *   `qpos[1]` (angle of first pole relative to cart) is set to -π radians (downwards).
        *   `qpos[2]` (angle of second pole relative to first pole) is set to 0 radians (aligned with first pole, so also downwards).
    *   Initial angular and linear velocities (`qvel`) are set to zero.
    *   This overrides the base environment's reset which starts near the upright
      position with small random noise.

2.  **Reward Function (`step` method):**
    *   The reward is recalculated to incentivize the swing-up behavior.
    *   The reward is based on the cosine of the angles of both poles relative to the vertical.
        *   We need the absolute angles: `theta1 = qpos[1]`, `theta2 = qpos[1] + qpos[2]`.
        *   Reward = `(cos(theta1) + cos(theta2)) / 2`. This yields:
            *   +1 when both poles are perfectly upright (angles = 0).
            *   -1 when both poles are hanging straight down.
            *   Values between -1 and +1 for intermediate states.
    *   This replaces the base environment's complex reward involving alive bonus,
      distance penalty, and velocity penalty, which is designed for stabilization.

3.  **Termination Conditions (`step` method):**
    *   The primary termination condition of the base environment (y-coordinate of the
      second pole's tip dropping below a threshold, `y <= 1.0`) is disabled.
      This is crucial for swing-up, as the poles start below this height.
    *   Termination is achieved by setting `terminated = False` within the `step` method
      after the base step is called, unless the state becomes non-finite.
    *   The environment still terminates if the state becomes non-finite (inherited
      from the base environment check) or truncates based on `max_episode_steps`.

4.  **Observation Space:**
    *   The observation space from the base environment is used directly. It includes:
      `[cart_pos, sin(pole1_angle), sin(pole2_angle), cos(pole1_angle), cos(pole2_angle), cart_vel, pole1_ang_vel, pole2_ang_vel, constraint_force]`.
    *   Note: While the reward uses `qpos` angles directly, the agent receives `sin`/`cos`.

5.  **Action Space:**
    *   The action space from the base environment is used directly: `Box(-1.0, 1.0, (1,))`.

6.  **Registration:**
    *   The wrapped environment is registered with the ID `DoublePendulum-SwingUp`
      using a factory function `make_env`.
    *   `max_episode_steps` is set to 500 (can be adjusted).

This wrapper facilitates reinforcement learning agents learning the double pendulum
swing-up maneuver.
"""
import gymnasium as gym
import numpy as np
import time
from gymnasium.envs.registration import register


class DoublePendulumSwingUp(gym.Wrapper):
    """
    Wrapper for InvertedDoublePendulum-v5 that modifies the reset state
    and reward function for a swing-up task.
    """
    
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped
        self.debug = debug
        
        # No specific parameter modifications needed in __init__ for termination,
        # as it's handled robustly in the step() method override.
        # Attempting to modify joint ranges is likely unnecessary due to sin/cos observations.
        if self.debug:
            print("DoublePendulumSwingUp wrapper initialized.")
            try:
                print(f"Base env action space: {self.action_space}")
                print(f"Base env obs space: {self.observation_space}")
            except Exception as e:
                 print(f"Could not print base env spaces: {e}")

    def print_state_details(self):
        """Print detailed state information for debugging"""
        if not self.debug or not hasattr(self.unwrapped_env, 'data'):
            return
            
        try:
            qpos = self.unwrapped_env.data.qpos.copy()
            qvel = self.unwrapped_env.data.qvel.copy()
            obs = self.unwrapped_env._get_obs() # Get current observation
            
            print(f"qpos (positions): {qpos}")
            print(f"qvel (velocities): {qvel}")
            print(f"Cart position: {qpos[0]:.3f}")
            print(f"Pole 1 angle: {qpos[1]:.3f} rad ({np.degrees(qpos[1]):.2f} deg)")
            print(f"Pole 2 angle (rel): {qpos[2]:.3f} rad ({np.degrees(qpos[2]):.2f} deg)")
            print(f"Observation: {obs}")
            
        except Exception as e:
            print(f"Error printing state details: {e}")
    
    def reset(self, **kwargs):
        # Temporarily disable rendering during reset if needed
        orig_render_mode = None
        if hasattr(self.unwrapped_env, 'render_mode') and self.unwrapped_env.render_mode is not None:
             orig_render_mode = self.unwrapped_env.render_mode
             self.unwrapped_env.render_mode = None
            
        # Call the base reset
        obs, info = self.env.reset(**kwargs)
        
        # Restore the render mode
        if orig_render_mode is not None:
            self.unwrapped_env.render_mode = orig_render_mode
        
        if self.debug:
            print("\n===== RESET STATE BEFORE MODIFICATION (Double Pendulum) =====")
            self.print_state_details()
        
        try:
            # Modify the state to have both poles pointing downward
            qpos = self.unwrapped_env.data.qpos.copy()
            qvel = self.unwrapped_env.data.qvel.copy()
            
            # Ensure qpos has expected length (cart + 2 poles = 3)
            if len(qpos) >= 3:
                qpos[1] = -np.pi  # Pole 1 angle relative to cart (downward)
                qpos[2] = 0.0     # Pole 2 angle relative to pole 1 (aligned, so also downward)
            else:
                 if self.debug: print(f"Warning: qpos length is {len(qpos)}, expected at least 3. Cannot set pole angles.")

            qvel[:] = 0.0     # Set all velocities to zero
            
            # Set the new state in the environment
            self.unwrapped_env.set_state(qpos, qvel)
            
            if self.debug:
                print("\n===== RESET STATE AFTER MODIFICATION (Double Pendulum) =====")
                self.print_state_details()
                print(f"Initial qpos set to: {qpos}")
                print(f"Initial qvel set to: {qvel}")

            # Get the observation corresponding to the new state
            obs = self.unwrapped_env._get_obs()
            if self.debug:
                 print(f"Observation after reset modification: {obs}")

        except Exception as e:
            if self.debug:
                print(f"Error setting state during reset: {e}")
            # Re-get observation just in case setting state failed but obs is needed
            obs = self.unwrapped_env._get_obs() 
        
        return obs, info
    
    def step(self, action):
        # Call the base environment step - ignore base reward and termination
        obs, _, base_terminated, truncated, info = self.env.step(action)

        # Calculate swing-up reward based on height (cosine of absolute angles)
        reward = 0.0
        terminated = False # Default to not terminated by this wrapper

        try:
            qpos = self.unwrapped_env.data.qpos.copy()
            if len(qpos) >= 3:
                # Absolute angles from vertical
                theta1 = qpos[1]
                theta2 = theta1 + qpos[2] # Angle of second pole relative to vertical
                
                # Reward = average cosine of absolute angles. Ranges from -1 (down) to +1 (up).
                reward = (np.cos(theta1) + np.cos(theta2)) / 2.0 
            else:
                 if self.debug: print("Warning: qpos length too short in step(), cannot calculate reward.")
                 reward = -1.0 # Penalize if state is weird

        except Exception as e:
            if self.debug: print(f"Error calculating reward: {e}")
            reward = -1.0 # Penalize if error occurs

        # Disable termination based on pole height (y_coordinate <= 1.0 in base env)
        # We explicitly set terminated = False unless state is non-finite.
        terminated = False 

        # Retain the base environment's termination condition for instability (non-finite state)
        if not np.isfinite(obs).all():
            terminated = True
            reward = -100.0 # Apply a large negative reward for instability
            if self.debug: print("Terminating due to non-finite observation.")
        # We could potentially also check base_terminated here if we wanted to keep 
        # other base termination conditions, but for swing-up, we usually only care about non-finite.

        # Add the custom reward to the info dict (optional, for monitoring)
        info['swing_up_reward'] = reward
        info['base_terminated'] = base_terminated # Keep base termination info if needed

        return obs, reward, terminated, truncated, info


# Create the factory function
def make_env(render_mode=None, debug=False, camera_config=None):
    """
    Factory function to create the DoublePendulumSwingUp environment.
    
    Args:
        render_mode (str, optional): The render mode for the base environment.
        debug (bool, optional): Enable debug printing in the wrapper.
        camera_config (dict, optional): Configuration for the MuJoCo camera.

    Returns:
        DoublePendulumSwingUp: The wrapped environment instance.
    """
    # Creates the base InvertedDoublePendulum environment, passing camera_config
    env_kwargs = {'render_mode': render_mode}
    if render_mode is not None and camera_config is not None:
        # Use the correct argument name for the base MuJoCo env
        env_kwargs['default_camera_config'] = camera_config
        
    base_env = gym.make('InvertedDoublePendulum-v5', **env_kwargs)
    # Wraps it with our swing-up modifications
    return DoublePendulumSwingUp(base_env, debug=debug)

# Function to perform registration
def register_double_pendulum_swing_up():
    """Registers the DoublePendulum-SwingUp-v0 environment."""
    if 'DoublePendulumSwingUp-v0' not in gym.envs.registry:
        register(
            id='DoublePendulumSwingUp-v0', # Corrected ID with version
            entry_point='src.environments.swing_up_envs.double_pendulum_su:make_env', # Use full path
            max_episode_steps=500,     # Truncation length (adjust if needed, 500 is short for double pendulum)
        )
        # print("Registered DoublePendulumSwingUp-v0") # Optional debug
    # else: print("DoublePendulumSwingUp-v0 already registered") # Optional debug

# Initial registration when module is imported (for non-subprocess use)
# register_double_pendulum_swing_up()
# We comment this out, registration will be handled by explicit calls

def main(perturb=False, debug=False):
    """
    Test the DoublePendulumSwingUp environment.
    Requires manual registration if not called via training script.
    
    Args:
        perturb (bool): If True, apply forces to the cart to see dynamics.
        debug (bool): If True, print detailed debug information.
    """
    print(f"Testing DoublePendulum-SwingUp environment (Perturb: {perturb}, Debug: {debug})")
    
    # Use the registered ID for the double pendulum
    env = gym.make('DoublePendulum-SwingUp', render_mode="human", debug=debug) 
    
    # Find the actual wrapper instance if debugging is enabled
    swing_up_env = None
    if debug:
        e = env 
        while True:
            if isinstance(e, DoublePendulumSwingUp):
                swing_up_env = e
                swing_up_env.debug = True # Ensure debug is set on the wrapper
                print("Found DoublePendulumSwingUp wrapper instance.")
                break
            if hasattr(e, 'env'):
                 e = e.env
            else:
                 # Reached the base env or a wrapper without .env
                 print("Warning: Could not find DoublePendulumSwingUp wrapper instance.")
                 break

    # Reset the environment
    try:
        obs, info = env.reset()
        if debug: print(f"Initial observation: {obs}")
    except Exception as e:
        print(f"Error during env.reset(): {e}")
        env.close()
        return

    step_count = 0
    max_steps = 300 # Limit test duration (might need more for double pendulum)
    
    # Main simulation loop
    for i in range(max_steps):
        # Action space is [-1, 1] for double pendulum
        action = np.array([0.0]) # Default action: no force
        force_str = "no force"
        
        if perturb:
            # Apply oscillating force to encourage swing-up
            if i < 150:
                 action = np.array([np.sin(i * 0.1) * 1.0]) 
                 force_str = f"sine force ({action[0]:.2f})"
            elif i < 250: 
                 action = np.array([1.0 if (i // 10) % 2 == 0 else -1.0]) # Alternating max force
                 force_str = f"bang-bang force ({action[0]:.1f})"
            # else: no force

        if debug and i % 10 == 0 and swing_up_env:
             print(f"\nStep {i} ({force_str})")
             swing_up_env.print_state_details()
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if debug:
                print(f"Step: {i}, Action: {action[0]:.2f}, Reward: {reward:.3f}, Term: {terminated}, Trunc: {truncated}")
                if terminated: print("  TERMINATED")
                if truncated: print("  TRUNCATED")

            env.render()
            time.sleep(0.02)
            
            if terminated or truncated:
                print(f"Episode finished after {step_count} steps. Terminated: {terminated}, Truncated: {truncated}")
                break
        except Exception as e:
             print(f"Error during env.step() or env.render() at step {i}: {e}")
             break
             
    if step_count == max_steps:
         print(f"Episode reached max test steps ({max_steps}).")

    print("Closing environment.")
    env.close()


if __name__ == "__main__":
    # Manually register if running this file directly for testing
    import gymnasium as gym
    from gymnasium.envs.registration import register
    register_double_pendulum_swing_up()
    # Set perturb=True to observe dynamics with applied forces
    # Set debug=True for detailed state info
    main(perturb=True, debug=False) 