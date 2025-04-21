"""
Train an RL agent using Stable Baselines3 library.

Handles environment creation, agent initialization (PPO), training,
model saving, configuration logging, and multi-session runs.
Logs detailed metrics to TensorBoard.

Usage Examples:
    # Train PPO on InvertedDoublePendulum-v5 for 1M timesteps
    python scripts/train_sb3_agent.py --env-id InvertedDoublePendulum-v5 --total-timesteps 1000000

    # Train PPO on Pendulum-SwingUp with centered reward for 500k timesteps across 3 seeds
    python scripts/train_sb3_agent.py --env-id Pendulum-SwingUp --total-timesteps 500000 --reward-mode cos_theta_centered --num-sessions 3 --base-seed 42

    # Train with custom PPO hyperparameters and policy network
    python scripts/train_sb3_agent.py --env-id InvertedDoublePendulum-v5 --total-timesteps 1000000 \
        --learning-rate 0.0001 --n-steps 2048 --batch-size 128 --gamma 0.995 \
        --policy-kwargs "dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))"
"""

import sys
import argparse
import json
import datetime
import hashlib
import random
import ast
from pathlib import Path
import functools # Import functools
import os # Import os for getpid

import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import registration functions explicitly
from src.environments.swing_up_envs.pendulum_su import register_pendulum_swing_up
from src.environments.swing_up_envs.double_pendulum_su import register_double_pendulum_swing_up

# Helper function from train_rl_agent.py
def get_run_id(config):
    """Generate a unique run ID based on key parameters and timestamp."""
    # Use different keys relevant to SB3 PPO
    config_str_parts = [
        config.env_id,
        config.agent.lower(),
        f"ts{config.total_timesteps}",
        f"lr{config.learning_rate}",
        f"gamma{config.gamma}",
        f"bs{config.batch_size}",
        f"nsteps{config.n_steps}",
    ]
    if config.policy_kwargs: # Add network arch if specified
        try:
            # Try to represent network arch concisely
            policy_dict = ast.literal_eval(config.policy_kwargs)
            arch_str = str(policy_dict.get('net_arch','def')).replace(' ','')
            config_str_parts.append(f"arch{arch_str}")
        except: pass # Ignore if parsing fails
        
    if config.env_id == 'Pendulum-SwingUp':
        config_str_parts.append(config.reward_mode)
        
    config_str = "_".join(config_str_parts)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f"{timestamp}_{config_hash}"


def create_env_factory(env_id, rank, seed=0, env_kwargs=None):
    """
    Utility function for multiprocessed envs.
    
    :param env_id: the environment ID
    :param rank: index of the subprocess
    :param seed: the inital seed for RNG
    :param env_kwargs: Keyword arguments to pass to gym.make
    :return: a function object that creates the environment
    """
    env_kwargs = env_kwargs or {}
    def _init():
        # Important: use env_kwargs here
        env = gym.make(env_id, **env_kwargs)
        # Note: SB3 handles seeding internally when using VecEnv
        # env.seed(seed + rank) # Not needed with SB3 VecEnv wrappers usually
        # env.action_space.seed(seed + rank) # Might be needed for some envs
        return env
    # set_global_seeds(seed) # SB3 handles global seeds
    return _init

class SummaryWriterCallback(BaseCallback):
    """ Custom callback to save summary stats """
    def __init__(self, run_dir, config_file_path, verbose=0):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.config_file_path = config_file_path
        self.summary = {}

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        We don't need to do anything here for this callback, but it must exist.
        Returning True ensures training continues.
        """
        return True

    def _on_training_end(self) -> None:
        ep_info_buffer = self.model.ep_info_buffer
        if ep_info_buffer:
            final_mean_reward = np.mean([ep_info["r"] for ep_info in ep_info_buffer])
            final_mean_len = np.mean([ep_info["l"] for ep_info in ep_info_buffer])
            self.summary = {
                "run_id": self.run_dir.name,
                "final_mean_reward_100ep": float(final_mean_reward),
                "final_mean_len_100ep": float(final_mean_len),
                "total_timesteps": self.num_timesteps,
                "config_file": str(self.config_file_path),
                "tensorboard_log": str(self.run_dir / "logs")
            }
            summary_file = self.run_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self.summary, f, indent=2)
            print(f"\nRun summary saved to: {summary_file}")
        else:
             print("\nWarning: No episode info found in buffer to save summary.")


def make_env_subprocess(env_id, env_kwargs=None):
    """Creates an env in a subprocess, ensuring custom envs are imported/registered.
    
    Args:
        env_id (str): The environment ID.
        env_kwargs (dict, optional): Keyword arguments for gym.make.
        
    Returns:
        gym.Env: The created environment instance.
    """
    # Ensure env_kwargs is a dictionary
    env_kwargs = env_kwargs or {}
    print(f"Subprocess PID {os.getpid()}: Attempting imports and registration...") # Debug
    try:
        # Import gymnasium AND explicitly call registration functions
        import gymnasium as gym
        # Need to import the functions themselves for the call
        from src.environments.swing_up_envs.pendulum_su import register_pendulum_swing_up
        from src.environments.swing_up_envs.double_pendulum_su import register_double_pendulum_swing_up
        
        register_pendulum_swing_up()       # Explicitly register
        register_double_pendulum_swing_up() # Explicitly register
        
        print(f"Subprocess PID {os.getpid()}: Registration called. Registry contains 'DoublePendulumSwingUp-v0': {'DoublePendulumSwingUp-v0' in gym.envs.registry}") # Debug
    except Exception as e:
        print(f"Subprocess PID {os.getpid()}: ERROR during import/registration: {e}")
        raise e
        
    print(f"Subprocess PID {os.getpid()}: Attempting gym.make('{env_id}') with kwargs {env_kwargs}") # Debug
    try:
        env = gym.make(env_id, **env_kwargs)
        print(f"Subprocess PID {os.getpid()}: gym.make successful.") # Debug
        return env
    except Exception as e:
        print(f"Subprocess PID {os.getpid()}: ERROR during gym.make: {e}") # Debug
        # Also print registry keys on error
        if 'gym' in locals() or 'gymnasium' in locals():
             registry_keys = list(gym.envs.registry.keys())
             print(f"Subprocess PID {os.getpid()}: Current registry keys ({len(registry_keys)}): {registry_keys[:20]}...")
        raise e # Re-raise the exception

def train_session(config, seed, run_dir):
    """Train one SB3 agent session."""
    print(f"\n--- Starting SB3 Training Session --- Run ID: {run_dir.name}, Seed: {seed} ---")

    # --- Environment Setup ---
    env_kwargs = {}
    if config.env_id == 'Pendulum-SwingUp':
        env_kwargs['reward_mode'] = config.reward_mode
        env_kwargs['center_penalty_weight'] = config.center_penalty_weight
        env_kwargs['limit_penalty'] = config.limit_penalty
        print(f"Using Pendulum-SwingUp specific kwargs: {env_kwargs}")
    elif config.env_id == 'DoublePendulum-SwingUp':
        # No extra kwargs needed for DoublePendulumSwingUp currently
        pass 
        
    # Create vectorized environment (uses multiple processes for speed)
    n_envs = os.cpu_count() if config.num_envs == -1 else config.num_envs
    if n_envs <= 0: n_envs = 1 # Ensure at least one env
    print(f"Using {n_envs} parallel environments.")

    # Create a list of env-making functions using functools.partial
    env_fns = []
    for i in range(n_envs):
        # Use partial to 'bake in' the arguments for the subprocess function
        # Pass a copy of env_kwargs to be safe
        partial_fn = functools.partial(make_env_subprocess, 
                                       env_id=config.env_id, 
                                       env_kwargs=env_kwargs.copy()) 
        env_fns.append(partial_fn)

    # Use DummyVecEnv for debugging or if SubprocVecEnv causes issues
    # Force DummyVecEnv to bypass multiprocessing issues on Windows for now
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv # Restore original logic
    # vec_env_cls = DummyVecEnv 
    # print("Forcing use of DummyVecEnv to bypass potential SubprocVecEnv issues.")
    
    # Instantiate the VecEnv with the list of functions
    # Do not use make_vec_env here, as we need the import inside the subprocess function
    env = vec_env_cls(env_fns)

    # --- Agent Setup ---
    tensorboard_log_path = run_dir / "logs"
    model_save_path = run_dir / "models" / f"{config.agent.lower()}_{config.env_id}_seed{seed}.zip"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle policy_kwargs string -> dict
    policy_kwargs_dict = None
    if config.policy_kwargs:
        try:
            policy_kwargs_dict = ast.literal_eval(config.policy_kwargs)
            print(f"Using custom policy kwargs: {policy_kwargs_dict}")
        except Exception as e:
            print(f"Warning: Could not parse policy_kwargs: {e}. Using defaults.")

    if config.agent.lower() == 'ppo':
        # Common PPO hyperparameters: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters
        model = PPO("MlpPolicy", 
                    env, 
                    learning_rate=config.learning_rate,
                    n_steps=config.n_steps,
                    batch_size=config.batch_size,
                    n_epochs=config.n_epochs,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    clip_range=config.clip_range,
                    ent_coef=config.ent_coef,
                    vf_coef=config.vf_coef,
                    max_grad_norm=config.max_grad_norm,
                    policy_kwargs=policy_kwargs_dict, 
                    tensorboard_log=str(tensorboard_log_path), 
                    seed=seed, 
                    verbose=1 # 0 = no output, 1 = training info, 2 = debug
                   )
    else:
        raise ValueError(f"Unsupported SB3 agent: {config.agent}")

    # --- Training --- 
    print(f"Starting training for {config.total_timesteps} timesteps...")
    print(f"TensorBoard logs will be saved to: {tensorboard_log_path}")
    print("Run `tensorboard --logdir runs` to view logs.")
    
    summary_callback = SummaryWriterCallback(run_dir, run_dir / "config.json")
    
    try:
        model.learn(total_timesteps=config.total_timesteps, callback=summary_callback, progress_bar=True)
        print("Training finished.")
    except Exception as e:
         print(f"\nERROR during model.learn: {e}")
         import traceback
         traceback.print_exc()
         print("Attempting to save model before exiting...")
    
    # --- Saving Model --- 
    try:
        model.save(str(model_save_path))
        print(f"Trained model saved to: {model_save_path}")
    except Exception as e:
        print(f"\nERROR saving model: {e}")
        
    # Close the environment
    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent using Stable Baselines3.")

    # --- General Args ---
    parser.add_argument("--env-id", type=str, required=True, help="Gymnasium environment ID")
    parser.add_argument("--agent", type=str, default="PPO", choices=['PPO'], help="SB3 agent type")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total number of training timesteps")
    parser.add_argument("--num-envs", type=int, default=-1, help="Number of parallel environments (-1 for auto cpu count)")

    # --- PPO Hyperparameters (Matching SB3 defaults where sensible) ---
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps to run for each environment per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs when optimizing the surrogate loss")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator")
    parser.add_argument("--clip-range", type=float, default=0.2, help="Clipping parameter")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient for the loss calculation")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient for the loss calculation")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="The maximum value for the gradient clipping")
    parser.add_argument("--policy-kwargs", type=str, default=None, help="Policy keyword arguments as a dictionary string (e.g. \"dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))\")")

    # --- Custom Env Args (Pendulum-SwingUp specific) ---
    parser.add_argument("--reward-mode", type=str, default='cos_theta', choices=['cos_theta', 'cos_theta_centered'], help="Reward mode for Pendulum-SwingUp env")
    parser.add_argument("--center-penalty-weight", type=float, default=0.1, help="Weight for center penalty in Pendulum-SwingUp")
    parser.add_argument("--limit-penalty", type=float, default=10.0, help="Penalty for hitting limits in Pendulum-SwingUp")

    # --- Multi-session Args ---
    parser.add_argument("--num-sessions", type=int, default=1, help="Number of independent training sessions to run")
    parser.add_argument("--base-seed", type=int, default=None, help="Base random seed for reproducibility. If None, random seeds are used.")

    args = parser.parse_args()
    
    # --- Run Setup & Config Saving ---
    # Note: Run ID generated inside the loop now for multi-session
    base_run_dir = Path("runs")
    all_run_ids = []

    for i in range(args.num_sessions):
        session_args = argparse.Namespace(**vars(args)) # Copy args for this session
        session_seed = args.base_seed + i if args.base_seed is not None else random.randint(0, 1_000_000)
        session_args.seed = session_seed # Add seed to session config

        # Generate unique run ID for this specific session
        run_id = get_run_id(session_args)
        all_run_ids.append(run_id)
        run_dir = base_run_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration for this session
        config_to_save = vars(session_args).copy()
        config_file = run_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_to_save, f, indent=2, sort_keys=True)
        print(f"\nSession {i+1}/{args.num_sessions}, Seed: {session_seed}, Run ID: {run_id}")
        print(f"Configuration saved to: {config_file}")
        
        # --- Run Training for One Session ---
        # Wrap in try-except to catch potential errors in one session
        # Especially useful with SubprocVecEnv which can have pickling issues on Windows
        try:
            train_session(session_args, session_seed, run_dir)
        except Exception as e:
            print(f"\nERROR occurred in training session {i+1} (Seed: {session_seed}, Run ID: {run_id}):")
            import traceback
            traceback.print_exc()
            print("Continuing to next session if any...")

    print(f"\n--- Completed All {args.num_sessions} Training Session(s) --- ")
    print("Run IDs:")
    for rid in all_run_ids:
        print(f"  {rid}")
    print("\nUse `tensorboard --logdir runs` to view training logs.") 