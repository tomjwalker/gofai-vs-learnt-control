"""
Evaluate a trained Stable Baselines3 agent.

Loads a trained model and its configuration, then runs the agent
with a deterministic policy for a specified number of episodes,
optionally recording video.

Usage:
    python scripts/evaluate_sb3_agent.py --run-id <RUN_ID> --eval-episodes 50
    python scripts/evaluate_sb3_agent.py --model-path <PATH_TO_MODEL.zip> --eval-episodes 30 --record-video
"""

import sys
import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch # Check if SB3 model uses torch
import numpy as np
from stable_baselines3 import PPO # Or other SB3 algos if needed
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import RecordVideo

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom environments to register them
import src.environments.swing_up_envs

# Imports from project
from src.algorithms.learnt.dqn_controller import DQNController # Keep if evaluating DQN later?
from src.environments.wrappers import DiscretizeActionWrapper
# Import registration functions explicitly
from src.environments.swing_up_envs.pendulum_su import register_pendulum_swing_up
from src.environments.swing_up_envs.double_pendulum_su import register_double_pendulum_swing_up

def evaluate_sb3(config, model_path, eval_episodes, run_dir):
    """Evaluate the SB3 agent deterministically."""
    print(f"\n--- Starting SB3 Evaluation ({eval_episodes} episodes) ---")
    print(f"Loading model from: {model_path}")
    
    # --- Load Model ---
    # Determine agent class based on config (or assume PPO for now)
    agent_class = PPO # Extend this later if supporting other SB3 agents
    if config.get('agent','PPO').upper() != 'PPO':
         print(f"Warning: Config agent is {config.get('agent')}, but only PPO loading is implemented.")
         
    # Load the trained agent
    # device='cpu' ensures it runs even if trained on GPU
    model = agent_class.load(model_path, device='cpu')
    print(f"Model loaded successfully. Policy architecture: {model.policy}")

    # --- Environment Setup ---
    render_mode = 'rgb_array' if config.get('record_video', False) else None
    
    camera_config = None
    if render_mode == 'rgb_array':
        # Use a default zoomed-out config, can be overridden if saved in config
        # Set azimuth to 90 for a side view, better for double pendulum
        camera_config = config.get('default_camera_config', 
                                 {"distance": 3.5, "azimuth": 90, "elevation": -20})
        print(f"Using camera config for recording: {camera_config}")
        
    env_kwargs = {}
    env_id = config['env_id']
    if env_id == 'Pendulum-SwingUp':
        env_kwargs['reward_mode'] = config.get('reward_mode', 'cos_theta')
        env_kwargs['center_penalty_weight'] = config.get('center_penalty_weight', 0.1)
        env_kwargs['limit_penalty'] = config.get('limit_penalty', 10.0)
        print(f"Using Pendulum-SwingUp specific kwargs: {env_kwargs}")
        
    try:
        # Create a single dummy environment for evaluation
        env_kwargs['render_mode'] = render_mode
        if render_mode and camera_config:
             # Pass camera_config, which our make_env factories expect
             env_kwargs['camera_config'] = camera_config
             
        # Need to use the non-vectorized env for standard eval loop
        env = gym.make(env_id, **env_kwargs)
        
    except gym.error.Error as e:
        print(f"Error creating environment {env_id} with kwargs {env_kwargs}: {e}")
        sys.exit(1)
        
    print(f"Environment: {env_id}")

    # --- Video Recording Setup (Optional) ---
    if config.get('record_video', False):
        video_folder = run_dir / "videos"
        video_folder.mkdir(exist_ok=True)
        print(f"Recording videos to: {video_folder}")
        # Record the first 3 episodes
        env = RecordVideo(env, video_folder=str(video_folder), 
                          episode_trigger=lambda ep_id: ep_id < 3, 
                          name_prefix=f"eval-{env_id}")

    # --- Evaluation Loop ---
    episode_rewards = []
    episode_lengths = []
    
    for i_episode in range(1, eval_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        while not terminated and not truncated:
            # Use model.predict for deterministic actions
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        if i_episode % 10 == 0 or i_episode == eval_episodes:
             print(f"Evaluation Episode: {i_episode}/{eval_episodes} | Length: {step} | Reward: {episode_reward:.2f}")

    env.close()

    # --- Results ---
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    print("\n--- SB3 Evaluation Complete ---")
    print(f"Episodes: {eval_episodes}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Length: {mean_length:.2f} +/- {std_length:.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    return mean_reward, std_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Stable Baselines3 agent.")
    
    # --- Explicitly Register Custom Environments --- 
    # Necessary because registration no longer happens automatically on import
    print("Registering custom environments...")
    register_pendulum_swing_up()
    register_double_pendulum_swing_up()
    print("Custom environments registered.")
    # --- End Registration ---
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", type=str, default=None, help="Run ID (e.g., 20250419...) to find model and config in runs/RUN_ID")
    group.add_argument("--model-path", type=str, default=None, help="Direct path to the trained model (.zip file)")
    
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of episodes for evaluation.")
    parser.add_argument('--record-video', action='store_true', help="Record videos of the first few evaluation episodes.")
    
    # Allow overriding env_id just in case
    parser.add_argument("--env-id", type=str, default=None, help="Override environment ID from config (use with caution). behavioural cloning")

    args = parser.parse_args()

    # Determine model path and config path
    model_path = None
    config_path = None
    run_dir = None

    if args.run_id:
        run_dir = Path("runs") / args.run_id
        if not run_dir.is_dir():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
        config_path = run_dir / "config.json"
        # Find the model file (SB3 saves as .zip)
        model_files = list((run_dir / "models").glob("*.zip"))
        if not model_files:
            print(f"Error: No model (.zip) files found in {run_dir / 'models'}")
            sys.exit(1)
        # Use the first model found (assuming one model per run_id)
        model_path = model_files[0] 
        print(f"Using run ID: {args.run_id}")
        print(f"Found model: {model_path}")

    elif args.model_path:
        model_path = Path(args.model_path)
        if not model_path.is_file():
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)
        # Assume config is in the parent directory of the model's directory
        run_dir = model_path.parent.parent # e.g., runs/ID/models/ -> runs/ID
        config_path = run_dir / "config.json"
        print(f"Using model path: {model_path}")

    if not run_dir or not run_dir.is_dir():
        print("Error: Could not determine run directory.")
        sys.exit(1)
        
    # Load configuration
    if not config_path or not config_path.is_file():
        print(f"Error: config.json not found at {config_path}")
        sys.exit(1) # Config is essential for evaluation setup
    else:
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            
    # Add record_video flag to config for eval function
    loaded_config['record_video'] = args.record_video 
        
    # Override env_id if specified (use carefully)
    if args.env_id:
        print(f"Warning: Overriding env_id from config with {args.env_id}")
        loaded_config['env_id'] = args.env_id
        
    # --- Run Evaluation ---
    evaluate_sb3(loaded_config, model_path, args.eval_episodes, run_dir) 