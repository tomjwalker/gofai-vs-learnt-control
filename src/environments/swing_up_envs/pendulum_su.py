"""
Custom Gymnasium environment wrapper for the Pendulum Swing-Up task.

This module defines a wrapper `PendulumSwingUp` around the standard
Gymnasium `InvertedPendulum-v5` environment. The goal is modified from simply
balancing the pole upright to starting with the pole hanging down and actively
swinging it up to the balanced position.

Modifications from the base `InvertedPendulum-v5` environment:

1.  **Initial State (`reset` method):**
    *   The pole's starting angle (`qpos[1]`) is deterministically set to -π radians
      (pointing straight down).
    *   The initial angular and linear velocities (`qvel`) are set to zero.
    *   This overrides the base environment's reset which starts near the upright
      position (angle ≈ 0) with small random noise.

2.  **Reward Function (`step` method):**
    *   The reward is recalculated based on the pole's angle to incentivize the
      swing-up behavior.
    *   The reward is `cos(pole_angle)`, where `pole_angle` is `obs[1]`. This yields:
        *   +1 when the pole is perfectly upright (angle = 0).
        *   -1 when the pole is hanging straight down (angle = ±π).
        *   Values between -1 and +1 for intermediate angles.
    *   This replaces the base environment's reward, which was +1 only for
      maintaining the pole within a small angle (±0.2 rad) near the top.

3.  **Termination Conditions (`step` method & `__init__`):**
    *   The termination condition based on the pole angle exceeding a threshold
      (±0.2 radians in the base environment) is disabled. This is achieved by
      setting `terminated = False` within the `step` method after the base step
      is called.
    *   An attempt is made in `__init__` to set the underlying model's theta
      threshold (`unwrapped_env.theta_threshold_radians`) to infinity, although
      modifying underlying model parameters post-initialization might not always
      be effective; the explicit override in `step` is the guaranteed method.
    *   The environment still terminates if the state becomes non-finite (as in
      the base environment) or truncates based on `max_episode_steps`.

4.  **Joint Limits (`__init__` method - attempted):**
    *   An attempt is made to modify the hinge joint's range (`model.jnt_range[1]`)
      to `[-π, π]` to explicitly allow full rotation. The effectiveness of
      modifying `mjModel` attributes directly after initialization can vary.
      However, the default MuJoCo model likely allows sufficient rotation.

5.  **Registration:**
    *   The wrapped environment is registered with the ID `Pendulum-SwingUp`
      using a factory function `make_env`.
    *   `max_episode_steps` is set to 500 for the registered environment.

This wrapper facilitates reinforcement learning agents learning the swing-up
maneuver, which requires different dynamics and reward signals compared to
simply balancing the pendulum.
"""
import gymnasium as gym
import numpy as np
import time
from gymnasium.envs.registration import register


class PendulumSwingUp(gym.Wrapper): # Renamed class
    """
    Wrapper for InvertedPendulum environment that modifies the reset state
    to start with the pole pointing downward instead of upward.
    """
    
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped
        self.debug = debug
        
        # Attempt to modify parameters to prevent termination based on angle limits
        # Note: Overriding terminated=False in step() is the more robust way
        try:
            self.unwrapped_env.theta_threshold_radians = float('inf')
            if self.debug:
                print("Set theta_threshold_radians to infinity (attempted).")
        except Exception as e:
            if self.debug:
                print(f"Could not set theta_threshold_radians: {e}")

        # Attempt to modify joint range (may not be necessary/effective)
        try:
            self.unwrapped_env.model.jnt_range[1][0] = -np.pi  # Lower limit: -π
            self.unwrapped_env.model.jnt_range[1][1] = np.pi   # Upper limit: π
            if self.debug:
                print("Modified joint ranges (attempted):", self.unwrapped_env.model.jnt_range[1])
        except Exception as e:
            if self.debug:
                print(f"Could not modify joint ranges: {e}")
    
    def print_state_details(self):
        """Print detailed state information for debugging"""
        if not self.debug:
            return
            
        try:
            qpos = self.unwrapped_env.data.qpos.copy()
            qvel = self.unwrapped_env.data.qvel.copy()
            
            print(f"qpos (positions): {qpos}")
            print(f"qvel (velocities): {qvel}")
            print(f"Cart position: {qpos[0]}, Pole angle: {qpos[1]} radians ({np.degrees(qpos[1]):.2f} degrees)")
            
            try: # Attempt to print joint ranges if accessible
                joint_ranges = []
                for i in range(self.unwrapped_env.model.nq):
                    joint_ranges.append((
                        self.unwrapped_env.model.jnt_range[i][0],
                        self.unwrapped_env.model.jnt_range[i][1]
                    ))
                print(f"Joint ranges: {joint_ranges}")
            except Exception as e:
                print(f"Could not retrieve joint ranges: {e}")
                
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
            print("\n===== RESET STATE BEFORE MODIFICATION =====")
            self.print_state_details()
        
        try:
            # Modify the state to have the pole pointing downward
            qpos = self.unwrapped_env.data.qpos.copy()
            qvel = self.unwrapped_env.data.qvel.copy()
            
            # Set pole angle to downward (-pi)
            qpos[1] = -np.pi 
            # Set all velocities to zero
            qvel[:] = 0.0  
            
            # Set the new state in the environment
            self.unwrapped_env.set_state(qpos, qvel)
            
            if self.debug:
                print("\n===== RESET STATE AFTER MODIFICATION =====")
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
        # Call the base environment step
        obs, _, terminated, truncated, info = self.env.step(action) # Base reward is ignored

        # Extract pole angle (assuming standard observation space: [cart_pos, pole_angle, cart_vel, pole_ang_vel])
        # Need to handle potential observation space differences if base env changes
        try:
            pole_angle = obs[1] 
        except IndexError:
             if self.debug: print("Error: Observation space does not seem to contain pole angle at index 1.")
             pole_angle = 0 # Default to upright if error occurs

        # Calculate swing-up reward: cos(angle)
        # Reward is +1 when upright (angle=0), -1 when hanging down (angle=±pi)
        reward = np.cos(pole_angle)

        # Ensure termination based on angle limit is disabled
        # Base environment terminates if abs(pole_angle) > 0.2 radians
        terminated = False

        # Retain the base environment's termination condition for instability
        if not np.isfinite(obs).all():
            terminated = True
            reward = -100.0 # Apply a large negative reward for instability
            if self.debug: print("Terminating due to non-finite observation.")


        # Add the custom reward to the info dict (optional, for monitoring)
        info['swing_up_reward'] = reward

        return obs, reward, terminated, truncated, info


# Create the factory function
def make_env(render_mode=None, debug=False):
    # Creates the base InvertedPendulum environment
    base_env = gym.make('InvertedPendulum-v5', render_mode=render_mode)
    # Wraps it with our swing-up modifications
    return PendulumSwingUp(base_env, debug=debug) # Use renamed class

# Register the environment
register(
    id='Pendulum-SwingUp', # Renamed ID
    entry_point=make_env, # Points to our factory function
    max_episode_steps=500, # Standard truncation length
)


def main(perturb=False, debug=False):
    """
    Test the PendulumSwingUp environment.
    
    Args:
        perturb (bool): If True, apply forces to the cart to see dynamics.
        debug (bool): If True, print detailed debug information.
    """
    print(f"Testing Pendulum-SwingUp environment (Perturb: {perturb}, Debug: {debug})")
    
    # Use the registered ID
    env = gym.make('Pendulum-SwingUp', render_mode="human", debug=debug) 
    
    # Find the actual wrapper instance if debugging is enabled
    # This allows calling methods like print_state_details directly on the wrapper
    swing_up_env = None
    if debug:
        # Need to access the wrapper instance potentially buried under other wrappers
        e = env 
        while hasattr(e, 'env') or isinstance(e, PendulumSwingUp): # Iterate through wrappers
             if isinstance(e, PendulumSwingUp):
                 swing_up_env = e
                 swing_up_env.debug = True # Ensure debug is set on the wrapper instance
                 print("Found PendulumSwingUp wrapper instance.")
                 break
             if not hasattr(e, 'env'): # Should not happen if gym.make worked
                 print("Error: Could not find PendulumSwingUp wrapper.")
                 break
             e = e.env # Move to the next inner environment/wrapper
        if swing_up_env is None:
             print("Warning: Could not find the SwingUp wrapper instance for debug prints.")


    # Reset the environment to get the initial state
    try:
        obs, info = env.reset()
        if debug: print(f"Initial observation: {obs}")
    except Exception as e:
        print(f"Error during env.reset(): {e}")
        env.close()
        return

    step_count = 0
    max_steps = 200 # Limit test duration
    
    # Main simulation loop
    for i in range(max_steps):
        action = np.array([0.0]) # Default action: no force
        force_str = "no force"
        
        if perturb:
            if i < 50:
                action = np.array([-1.0]) # Apply left force
                force_str = "left force (-1.0)"
            elif i < 100:
                 action = np.array([1.0]) # Apply right force
                 force_str = "right force (1.0)"
            # else: no force for the rest

        if debug and i % 10 == 0 and swing_up_env:
             print(f"\nStep {i} ({force_str})")
             swing_up_env.print_state_details() # Call method on the wrapper instance
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if debug:
                print(f"Step: {i}, Action: {action[0]:.2f}, Reward: {reward:.3f}, Term: {terminated}, Trunc: {truncated}")
                # print(f"  Obs: {obs}") # Can be very verbose
                if terminated: print("  TERMINATED")
                if truncated: print("  TRUNCATED")

            env.render()
            time.sleep(0.02) # Slow down rendering slightly
            
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
    # Set perturb=True to observe the pendulum dynamics with applied forces
    # Set debug=True to see detailed state information during reset and steps
    main(perturb=True, debug=False) 