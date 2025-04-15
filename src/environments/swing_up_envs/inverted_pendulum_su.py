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
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Never terminate due to angle limits
        terminated = False
        
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
