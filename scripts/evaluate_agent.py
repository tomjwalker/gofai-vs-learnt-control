"""
Evaluate a trained DQN agent on a Gymnasium environment.

Loads a trained model and its configuration, then runs the agent
with a deterministic policy (epsilon=0) for a specified number of episodes.

Usage:
    python scripts/evaluate_agent.py --run-id <RUN_ID> --eval-episodes 100
    python scripts/evaluate_agent.py --model-path <PATH_TO_MODEL.pth> --eval-episodes 50
"""

import sys
import argparse
import json
from pathlib import Path
import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import RecordVideo # Added for video recording

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Imports from project
from src.algorithms.learnt.dqn_controller import DQNController
from src.environments.swing_up_envs.pendulum_su import register_pendulum_swing_up # Explicit import
register_pendulum_swing_up() # Call registration function early
from src.environments.wrappers import DiscretizeActionWrapper
from src.utils.plotting import plot_diagnostics

def evaluate(config, model_state_dict, eval_episodes, run_dir):
    """Evaluate the agent with a deterministic policy."""
    print(f"\n--- Starting Evaluation ({eval_episodes} episodes) ---")
    
    # --- Environment Setup ---
    render_mode = 'rgb_array' if config.get('record_video', False) else None
    
    camera_config = None
    if render_mode == 'rgb_array':
        camera_config = {
            "distance": 3.5,  # Zoom out further than default
            # "azimuth": 0,     # Default is usually fine
            # "elevation": -20, # Default is usually fine
            # "lookat": np.array([0.0, 0.0, 0.0]), # Default lookat
        }
        print(f"Using custom camera config for recording: {camera_config}")
        
    env_kwargs = {}
    if config.get('env_id') == 'Pendulum-SwingUp':
        # Get reward mode and penalty weights from loaded config, using defaults if missing
        env_kwargs['reward_mode'] = config.get('reward_mode', 'cos_theta')
        env_kwargs['center_penalty_weight'] = config.get('center_penalty_weight', 0.1)
        env_kwargs['limit_penalty'] = config.get('limit_penalty', 10.0)
        print(f"Using Pendulum-SwingUp specific kwargs: {env_kwargs}")
        
    try:
        env_kwargs['render_mode'] = render_mode
        if render_mode and camera_config:
             env_kwargs['camera_config'] = camera_config 
             
        env = gym.make(config['env_id'], **env_kwargs)

    except gym.error.Error as e:
        print(f"Error creating environment {config['env_id']}: {e}")
        sys.exit(1)
        
    # Get state dimension
    state_dim = env.observation_space.shape[0]

    # Apply wrapper if needed (based on loaded config)
    action_dim = config['action_dim'] # Will be overwritten if wrapped
    if config.get('n_discrete_actions') and isinstance(env.action_space, gym.spaces.Box):
        print(f"Applying action discretisation with {config['n_discrete_actions']} bins.")
        env = DiscretizeActionWrapper(env, n_bins=config['n_discrete_actions'])
        action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        # Assuming continuous action space was handled or agent supports it
        # If original action dim wasn't saved, we might have issues here for non-DQN agents
        pass 
        
    print(f"Environment: {config['env_id']}, State dim: {state_dim}, Action dim: {action_dim}")

    # --- Video Recording Setup (Optional) ---
    if config.get('record_video', False):
        video_folder = run_dir / "videos"
        video_folder.mkdir(exist_ok=True)
        print(f"Recording videos to: {video_folder}")
        # Record the first 3 episodes
        env = RecordVideo(env, video_folder=str(video_folder), 
                          episode_trigger=lambda ep_id: ep_id < 3, 
                          name_prefix=f"eval-{config['env_id']}")
        # Note: RecordVideo might handle render calls, check behavior if using manual render

    # --- Agent Setup ---
    # Use loaded config parameters
    agent = DQNController(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=tuple(config['hidden_dims']), # Ensure tuple
        lr=config['lr'], # Not used for eval, but needed for init
        gamma=config['gamma'], # Not used for eval
        epsilon=0.0, # Force greedy policy for evaluation
        epsilon_min=0.0,
        epsilon_decay=0.0,
        buffer_capacity=1, # Not used for eval
        batch_size=1, # Not used for eval
        target_update_interval=1000000 # Effectively disable target updates
    )
    
    # Load the trained weights
    agent.q_net.load_state_dict(model_state_dict)
    agent.q_net.eval() # Set the network to evaluation mode (important for dropout, batchnorm etc.)
    print("Loaded trained model weights.")

    # --- Evaluation Loop ---
    episode_rewards = []
    for i_episode in range(1, eval_episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        history = [] # Initialize history list for this episode

        while not terminated and not truncated:
            # Select action greedily using eval_mode=True
            action = agent.select_action(state, eval_mode=True)
            
            # --- Store state and action BEFORE stepping --- 
            # Use the state *before* the step and the action *taken*
            history.append({
                'obs': state,
                'action': action 
            })
            # ----------------------------------------------
            
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)
        if i_episode % 10 == 0 or i_episode == eval_episodes:
             print(f"Evaluation Episode: {i_episode}/{eval_episodes} | Reward: {episode_reward:.2f}")
             
        # --- Plot diagnostics after the episode --- 
        plots_dir = run_dir / "plots"
        plot_diagnostics(history, plots_dir, episode=i_episode, plot_cost=False)
        # ----------------------------------------

    env.close()

    # --- Results ---
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print("\n--- Evaluation Complete ---")
    print(f"Episodes: {eval_episodes}")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Standard Deviation: {std_reward:.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent.")
    
    # Option 1: Specify run ID
    parser.add_argument("--run-id", type=str, default=None, help="Run ID of the training run to evaluate (looks for model and config in runs/RUN_ID)")
    # Option 2: Specify model path directly (config should be in parent dir)
    parser.add_argument("--model-path", type=str, default=None, help="Direct path to the trained model (.pth file). Assumes config.json is in the parent directory.")
    
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of episodes to run evaluation for.")
    parser.add_argument('--record-video', action='store_true', help="Record videos of the first few evaluation episodes.")
    
    # Optional overrides if config.json is missing or needs changing
    parser.add_argument("--env-id", type=str, default=None, help="Override environment ID from config.")
    parser.add_argument("--n-discrete-actions", type=int, default=None, help="Override number of discrete actions from config.")
    # Add args to override reward params for Pendulum-SwingUp if config is missing/wrong
    parser.add_argument("--reward-mode", type=str, default=None, choices=['cos_theta', 'cos_theta_centered'], help="Override reward mode for Pendulum-SwingUp env")
    parser.add_argument("--center-penalty-weight", type=float, default=None, help="Override center penalty weight for Pendulum-SwingUp")
    parser.add_argument("--limit-penalty", type=float, default=None, help="Override limit penalty for Pendulum-SwingUp")

    args = parser.parse_args()

    if not args.run_id and not args.model_path:
        parser.error("Either --run-id or --model-path must be specified.")
    if args.run_id and args.model_path:
        parser.error("Specify either --run-id or --model-path, not both.")

    # Determine model path and config path
    model_path = None
    config_path = None
    run_dir = None

    if args.run_id:
        run_dir = Path("runs") / "DRL" / args.run_id
        if not run_dir.is_dir():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
        config_path = run_dir / "config.json"
        # Find the model file (assumes only one .pth per session, take first found)
        model_files = list((run_dir / "models").glob("*.pth"))
        if not model_files:
            print(f"Error: No model (.pth) files found in {run_dir / 'models'}")
            sys.exit(1)
        model_path = model_files[0] # Use the first model found in the run
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

    if not run_dir:
        print("Error: Could not determine run directory for saving videos/configs.")
        sys.exit(1)

    # Load configuration
    if not config_path or not config_path.is_file():
        print(f"Warning: config.json not found at {config_path}")
        # Try to proceed with defaults or command-line overrides if provided
        loaded_config = {}
    else:
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            
    # Add record_video flag to config
    loaded_config['record_video'] = args.record_video 

    # --- Handle Legacy Env ID --- 
    if loaded_config.get('env_id') == 'Pendulum-SwingUp':
        print("Warning: Found legacy env ID 'Pendulum-SwingUp'. Using 'Pendulum-SwingUp-v0' instead.")
        loaded_config['env_id'] = 'Pendulum-SwingUp-v0'

    # --- Determine Action Dimension from Config --- 
    # Necessary for agent init before loading state_dict
    # Attempt to find original action_dim if saved, otherwise infer based on discretisation
    if 'action_dim' not in loaded_config: 
        print("Warning: 'action_dim' not found in config. Inferring...")
        # --- Explicitly apply legacy fix here too ---
        env_id_for_inference = loaded_config.get('env_id', args.env_id or 'InvertedPendulum-v5')
        if env_id_for_inference == 'Pendulum-SwingUp':
             env_id_for_inference = 'Pendulum-SwingUp-v0' # Apply fix
             print(f"Applying legacy fix inside action_dim inference: using {env_id_for_inference}")
        # -----------------------------------------------
        print(f"--- DEBUG: Attempting gym.make with env_id: {env_id_for_inference} for action_dim inference ---") # DEBUG
        temp_env = gym.make(env_id_for_inference) # Use default if missing
        if loaded_config.get('n_discrete_actions') and isinstance(temp_env.action_space, gym.spaces.Box):
             loaded_config['action_dim'] = loaded_config['n_discrete_actions']
        elif isinstance(temp_env.action_space, gym.spaces.Discrete):
             loaded_config['action_dim'] = temp_env.action_space.n
        else:
            print(f"Error: Cannot determine action dimension for space {type(temp_env.action_space)}")
            sys.exit(1)
        temp_env.close()
        print(f"Inferred action_dim: {loaded_config['action_dim']}")
        
    # Override config with command-line args if provided
    if args.env_id:
        loaded_config['env_id'] = args.env_id
    if args.n_discrete_actions:
        loaded_config['n_discrete_actions'] = args.n_discrete_actions
        loaded_config['action_dim'] = args.n_discrete_actions # Assume discretisation overrides
    if args.reward_mode:
        loaded_config['reward_mode'] = args.reward_mode
    if args.center_penalty_weight is not None:
        loaded_config['center_penalty_weight'] = args.center_penalty_weight
    if args.limit_penalty is not None:
        loaded_config['limit_penalty'] = args.limit_penalty
        
    # Ensure required keys for DQNController are present
    required_keys = ['hidden_dims', 'lr', 'gamma', 'epsilon', 'epsilon_min', 'epsilon_decay', 'buffer_capacity', 'batch_size', 'target_update_interval']
    # Use defaults from train script's parser if missing in config
    defaults = {
        'hidden_dims': [128, 64], 'lr': 0.0005, 'gamma': 0.99, 'epsilon': 1.0, 
        'epsilon_min': 0.005, 'epsilon_decay': 0.0005, 'buffer_capacity': 50000, 
        'batch_size': 128, 'target_update_interval': 200
    }
    for key in required_keys:
         if key not in loaded_config:
            # Check if the key exists as an argument override first
            if hasattr(args, key) and getattr(args, key) is not None:
                print(f"Using command-line override for '{key}': {getattr(args, key)}")
                loaded_config[key] = getattr(args, key)
            else:
                print(f"Warning: '{key}' not found in config. Using default value: {defaults[key]}")
                loaded_config[key] = defaults[key]

    # Load model state dict
    # Ensure model is loaded to the correct device (CPU for simplicity here)
    # If you train on GPU and eval on CPU, add map_location='cpu'
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu')) 

    # Run evaluation, passing the run_dir for potential video saving
    evaluate(loaded_config, model_state_dict, args.eval_episodes, run_dir) 