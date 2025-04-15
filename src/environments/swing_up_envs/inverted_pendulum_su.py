"""
Custom Gymnasium environment wrapper for the Inverted Pendulum Swing-Up task.

This module defines a wrapper `InvertedPendulumSwingUp` around the standard
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
    *   The wrapped environment is registered with the ID `InvertedPendulum-SwingUp`
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


class InvertedPendulumSwingUp(gym.Wrapper):
    """
    Wrapper for InvertedPendulum environment that modifies the reset state
    to start with the pole pointing downward instead of upward.
    """
    
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped
        self.debug = debug
        
        # Modify the joint range to allow full rotation and prevent termination
        try:
            self.unwrapped_env.theta_threshold_radians = float('inf')
            self.unwrapped_env.model.jnt_range[1][0] = -np.pi  # Lower limit: -π
            self.unwrapped_env.model.jnt_range[1][1] = np.pi   # Upper limit: π
            if self.debug:
                print("Modified joint ranges:", self.unwrapped_env.model.jnt_range)
        except Exception as e:
            if self.debug:
                print(f"Could not modify environment parameters: {e}")
    
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
            
            try:
                joint_ranges = []
                for i in range(self.unwrapped_env.model.nq):
                    joint_ranges.append((
                        self.unwrapped_env.model.jnt_range[i][0],
                        self.unwrapped_env.model.jnt_range[i][1]
                    ))
                print(f"Joint ranges: {joint_ranges}")
            except:
                pass
                
        except Exception as e:
            print(f"Error printing state details: {e}")
    
    def reset(self, **kwargs):
        # Temporarily disable rendering during reset
        orig_render_mode = None
        try:
            orig_render_mode = self.unwrapped_env.render_mode
            self.unwrapped_env.render_mode = None
        except:
            pass
            
        # Call the base reset
        obs, info = self.env.reset(**kwargs)
        
        # Restore the render mode
        if orig_render_mode is not None:
            try:
                self.unwrapped_env.render_mode = orig_render_mode
            except:
                pass
        
        if self.debug:
            print("\n===== RESET STATE BEFORE MODIFICATION =====")
            self.print_state_details()
        
        try:
            # Modify the state to have the pole pointing downward
            qpos = self.unwrapped_env.data.qpos.copy()
            qvel = self.unwrapped_env.data.qvel.copy()
            
            # Set to the most negative value allowed (-π)
            qpos[1] = -np.pi
            qvel[:] = 0.0  # Set all velocities to zero
            
            # Set the new state
            self.unwrapped_env.set_state(qpos, qvel)
            
            if self.debug:
                print("\n===== RESET STATE AFTER MODIFICATION =====")
                self.print_state_details()
            
            # Get the new observation
            obs = self.unwrapped_env._get_obs()
        except Exception as e:
            if self.debug:
                print(f"Error setting state: {e}")
        
        return obs, info
    
    def step(self, action):
        # Call the base environment step
        obs, _, terminated, truncated, info = self.env.step(action) # Base reward is ignored

        # Extract pole angle (assuming standard observation space: [cart_pos, pole_angle, cart_vel, pole_ang_vel])
        pole_angle = obs[1]

        # Calculate swing-up reward: cos(angle)
        # Reward is +1 when upright (angle=0), -1 when hanging down (angle=±pi)
        reward = np.cos(pole_angle)

        # Ensure termination based on angle limit is disabled
        # Base environment terminates if abs(pole_angle) > 0.2 radians
        terminated = False

        # Optional: Check for non-finite states and terminate if found
        # This retains the base environment's termination condition for instability
        if not np.isfinite(obs).all():
            terminated = True
            reward = -100.0 # Apply a large negative reward for instability

        # Add the custom reward to the info dict (optional)
        info['swing_up_reward'] = reward

        return obs, reward, terminated, truncated, info


# Create the factory function
def make_env(render_mode=None, debug=False):
    base_env = gym.make('InvertedPendulum-v5', render_mode=render_mode)
    return InvertedPendulumSwingUp(base_env, debug=debug)

# Register the environment
register(
    id='InvertedPendulum-SwingUp',
    entry_point=make_env,
    max_episode_steps=500,
)


def main(perturb=False, debug=False):
    """
    Test the InvertedPendulumSwingUp environment.
    
    Args:
        perturb (bool): If True, apply forces to the cart to see dynamics.
        debug (bool): If True, print debug information.
    """
    # First create with no rendering to set up the environment
    setup_env = make_env(render_mode=None, debug=debug)
    setup_env.reset()
    
    # Now create the environment with rendering
    env = gym.make('InvertedPendulum-SwingUp', render_mode="human")
    
    # Find the actual wrapper in the environment stack if debugging is enabled
    swing_up_env = None
    if debug:
        e = env
        while hasattr(e, 'env'):
            if isinstance(e, InvertedPendulumSwingUp):
                swing_up_env = e
                break
            e = e.env
    
    # Reset with proper initialization
    obs, info = env.reset()
    
    # Main simulation loop
    if perturb:
        # Apply left force for 50 steps
        for i in range(50):
            if debug and i % 10 == 0 and swing_up_env:
                print(f"\nStep {i} (left force)")
                swing_up_env.print_state_details()
            
            obs, reward, terminated, truncated, info = env.step(np.array([-1.0]))
            env.render()
            time.sleep(0.05)
            
            if terminated or truncated:
                break
        
        # Apply right force for 50 steps
        for i in range(50):
            if debug and i % 10 == 0 and swing_up_env:
                print(f"\nStep {i} (right force)")
                swing_up_env.print_state_details()
            
            obs, reward, terminated, truncated, info = env.step(np.array([1.0]))
            env.render()
            time.sleep(0.05)
            
            if terminated or truncated:
                break
    else:
        # Just show the pendulum in its starting position for 100 timesteps
        for i in range(100):
            if debug and i % 10 == 0 and swing_up_env:
                print(f"\nStep {i} (no force)")
                swing_up_env.print_state_details()
            
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
            env.render()
            time.sleep(0.05)
            
            if terminated or truncated:
                break
    
    env.close()


if __name__ == "__main__":
    # Set perturb=True to observe the pendulum dynamics with applied forces
    # Set debug=True to see detailed state information
    main(perturb=True, debug=False)
