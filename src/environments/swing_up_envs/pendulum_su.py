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
from pathlib import Path
from gymnasium.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import utils
import os

# Define path to custom XML near the top of the file
# Assumes this script is in src/environments/swing_up_envs
current_dir = Path(__file__).parent # Now assumes XML is in the same directory
CUSTOM_XML_PATH = current_dir / "inverted_pendulum_swingup.xml" # Updated path

# --- Custom Env Class using Unlimited XML ---
class InvertedPendulumUnlimitedEnv(MujocoEnv, utils.EzPickle):
    """ Subclass of MujocoEnv that directly loads the custom swingup XML. """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25, # 1/0.04
    }
    
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # Define observation space (copied from standard InvertedPendulumEnv)
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )
        # Calculate and print absolute path
        absolute_xml_path = os.path.abspath(CUSTOM_XML_PATH)
        print(f"--- DEBUG: Initializing MujocoEnv with absolute_model_path: {absolute_xml_path} ---")
        # Call the MUJOCO base class constructor directly
        MujocoEnv.__init__(
            self,
            model_path=absolute_xml_path,
            frame_skip=2, # Standard frame skip for InvertedPendulum
            observation_space=observation_space,
            **kwargs
        )
        
        # --- Post-Initialization Model Modification --- 
        # Workaround for model_path loading issues: Modify the loaded model directly.
        try:
            hinge_idx = self.model.joint('hinge').id # Get ID by name
            if hinge_idx >= 0:
                 # Apply desired unlimited range
                 self.model.jnt_range[hinge_idx] = [-10000.0, 10000.0] 
                 # Ensure joint is marked as not limited
                 self.model.jnt_limited[hinge_idx] = 0 # 0 means false
                 print(f"--- Post-init MODIFIED hinge joint (idx {hinge_idx}) range to: {self.model.jnt_range[hinge_idx]}, limited: {self.model.jnt_limited[hinge_idx]} ---")
            else:
                 print("--- Post-init WARNING: Could not find hinge joint by name to modify. ---")
        except Exception as e:
            print(f"--- Post-init WARNING: Error modifying hinge joint: {e} ---")
        # ----------------------------------------------
        
        # --- Extract bounds immediately after modification ---
        self.extracted_joint_bounds = {} # Store on instance
        if hasattr(self.model, 'jnt_range') and hasattr(self.model, 'jnt_qposadr'):
            for i in range(self.model.njnt):
                jnt_name = self.model.joint(i).name
                qpos_adr = self.model.jnt_qposadr[i]
                low, high = self.model.jnt_range[i]
                state_name = None
                if jnt_name == 'slider': state_name = 'cart_pos'
                elif jnt_name == 'hinge': state_name = 'pole_angle'
                if state_name:
                    self.extracted_joint_bounds[state_name] = [float(low), float(high)]
                else:
                    self.extracted_joint_bounds[f"{jnt_name}_qpos{qpos_adr}"] = [float(low), float(high)]
            print(f"--- Post-init EXTRACTED bounds: {self.extracted_joint_bounds} ---")
        else:
            print("--- Post-init WARNING: Could not extract joint ranges immediately. ---")
        # --------------------------------------------------
        
    def step(self, action):
        # Standard step logic, but reward/termination is handled by the wrapper
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        # In this base class, reward and terminated are simple
        reward = 1.0 # Base reward (will be overwritten by wrapper)
        terminated = False # Base termination (will be overwritten by wrapper)
        info = {}
        if self.render_mode == "human":
            self.render()
        # The PendulumSwingUp wrapper will modify reward and termination
        return observation, reward, terminated, False, info # Return truncated=False always

    def _get_obs(self):
        # Standard observation: qpos[1], qvel
        # Need to ensure correct order [cart_pos, pole_angle, cart_vel, pole_ang_vel]
        qpos = self.data.qpos
        qvel = self.data.qvel
        # MuJoCo state: [cart_pos, pole_angle] -> [qpos[0], qpos[1]]
        # Observation:   [cart_pos, pole_angle, cart_vel, pole_ang_vel]
        return np.concatenate((qpos, qvel)).ravel() # This seems wrong, should match space
        # Correction: Standard observation is just velocities + pole pos sin/cos
        # Let's match the standard InvertedPendulum-v5 obs directly
        # return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        # return np.concatenate([self.data.qpos, np.clip(self.data.qvel, -10, 10)]).ravel()
        # Actually, v5 uses: position + velocity
        # return np.concatenate((self.data.qpos, self.data.qvel)).ravel()
        # Final check based on standard implementation:
        return np.concatenate((qpos, qvel)).ravel()
        
    def reset_model(self):
        # Standard reset logic (randomizes slightly near top)
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        # The PendulumSwingUp wrapper will override this initial state
        return self._get_obs()
# -------------------------------------------

class PendulumSwingUp(gym.Wrapper): # Renamed class
    """
    Wrapper for InvertedPendulum environment that modifies the reset state
    to start with the pole pointing downward instead of upward.
    Adds different reward modes.
    """
    
    def __init__(self, env, debug=False, 
                 reward_mode='cos_theta', 
                 center_penalty_weight=0.1, 
                 limit_penalty=10.0):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped
        self.debug = debug
        self.reward_mode = reward_mode
        self.center_penalty_weight = center_penalty_weight
        self.limit_penalty = limit_penalty
        self.cart_limit = 1.0 # Known from XML
        
        if self.debug:
            print(f"PendulumSwingUp Initialized. Reward Mode: {self.reward_mode}, Center Penalty: {self.center_penalty_weight}, Limit Penalty: {self.limit_penalty}")

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

        # Calculate swing-up reward based on mode
        base_reward = np.cos(pole_angle)
        reward = base_reward
        
        info['reward_cos_theta'] = base_reward
        center_penalty = 0.0
        at_limit_penalty = 0.0

        if self.reward_mode == 'cos_theta_centered':
            # Get cart position
            try:
                cart_pos = obs[0]
            except IndexError:
                if self.debug: print("Error: Observation space does not seem to contain cart position at index 0.")
                cart_pos = 0.0
                
            # Centering penalty
            center_penalty = self.center_penalty_weight * abs(cart_pos)
            reward -= center_penalty
            
            # Limit penalty (apply if very close to or at the limit)
            if abs(cart_pos) >= (self.cart_limit - 0.01):
                 at_limit_penalty = self.limit_penalty
                 reward -= at_limit_penalty
                 if self.debug and abs(cart_pos) >= self.cart_limit: print("Limit penalty applied!")
        
        info['reward_center_penalty'] = center_penalty
        info['reward_limit_penalty'] = at_limit_penalty

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
def make_env(render_mode=None, debug=False, camera_config=None, 
             reward_mode='cos_theta', center_penalty_weight=0.1, limit_penalty=10.0,
             use_unlimited_xml=False): # Added flag
    """
    Factory function to create the PendulumSwingUp environment.
    Can optionally load a custom XML with wider joint limits.
    
    Args:
        render_mode (str, optional): The render mode for the base environment.
        debug (bool, optional): Enable debug printing in the wrapper.
        camera_config (dict, optional): Configuration for the MuJoCo camera.
        reward_mode (str): The reward calculation mode ('cos_theta' or 'cos_theta_centered').
        center_penalty_weight (float): Weight for the cart centering penalty.
        limit_penalty (float): Penalty for hitting the cart position limits.
        use_unlimited_xml (bool): If True, load from custom XML file.

    Returns:
        PendulumSwingUp: The wrapped environment instance.
    """
    # Only pass camera_config if render_mode is also set, as it's MuJoCo specific
    env_kwargs = {'render_mode': render_mode}
    if render_mode is not None and camera_config is not None:
        env_kwargs['default_camera_config'] = camera_config 
        
    if use_unlimited_xml:
        # Load base env from custom XML
        print(f"Using custom XML for swing-up: {CUSTOM_XML_PATH}")
        if not CUSTOM_XML_PATH.exists():
            raise FileNotFoundError(f"Custom XML not found at: {CUSTOM_XML_PATH}")
        base_env = InvertedPendulumUnlimitedEnv(**env_kwargs)
    else:
        # Load standard base env
        print("Using standard InvertedPendulum-v5 for swing-up.")
        base_env = gym.make('InvertedPendulum-v5', **env_kwargs)
        
    # Wraps it with our swing-up modifications
    return PendulumSwingUp(base_env, debug=debug, 
                         reward_mode=reward_mode, 
                         center_penalty_weight=center_penalty_weight, 
                         limit_penalty=limit_penalty)

# Function to perform registration
def register_pendulum_swing_up():
    """Registers Pendulum-SwingUp environments (standard and unlimited)."""
    # Standard version (for DRL compatibility)
    std_id = 'Pendulum-SwingUp-v0'
    if std_id not in gym.envs.registry:
        print(f"Registering standard swing-up: {std_id}")
        register(
            id=std_id,
            entry_point='src.environments.swing_up_envs.pendulum_su:make_env',
            max_episode_steps=500,
            kwargs={ # Explicitly set use_unlimited_xml=False (or rely on default)
                'use_unlimited_xml': False,
                'reward_mode': 'cos_theta',
                'center_penalty_weight': 0.1,
                'limit_penalty': 10.0,
                'debug': False,
                'camera_config': None
            } 
        )

    # Unlimited version (for MPC)
    unlimited_id = 'Pendulum-SwingUpUnlimited-v0'
    if unlimited_id not in gym.envs.registry:
        print(f"Registering unlimited swing-up: {unlimited_id}")
        register(
            id=unlimited_id,
            entry_point='src.environments.swing_up_envs.pendulum_su:make_env',
            max_episode_steps=1000, # Allow longer runs for MPC if needed
            kwargs={ 
                'use_unlimited_xml': True, # Use the custom XML!
                'reward_mode': 'cos_theta', # Keep other defaults consistent for now
                'center_penalty_weight': 0.1,
                'limit_penalty': 10.0,
                'debug': False,
                'camera_config': None
            } 
        )

# Initial registration when module is imported (for non-subprocess use)
# register_pendulum_swing_up()
# We comment this out, registration will be handled by explicit calls

def main(perturb=False, debug=False):
    """
    Test the PendulumSwingUp environment.
    Requires manual registration if not called via training script.
    
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
    # Manually register if running this file directly for testing
    import gymnasium as gym
    from gymnasium.envs.registration import register
    register_pendulum_swing_up()
    # Set perturb=True to observe the pendulum dynamics with applied forces
    # Set debug=True to see detailed state information during reset and steps
    main(perturb=True, debug=False) 