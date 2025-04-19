"""
Script to train a learnt controller (e.g., DQN) on a Gymnasium environment.

Usage Examples:
    # Basic training on InvertedPendulum-v4 with default parameters
    python scripts/train_rl_agent.py

    # Train on custom environment with more episodes and custom save path
    python scripts/train_rl_agent.py --env-id PendulumCustomEnv-v0 --episodes 1000 --save-path models/learnt/dqn_pendulum_custom.pth

    # Train with custom hyperparameters
    python scripts/train_rl_agent.py --env-id InvertedPendulum-v4 \
        --episodes 500 \
        --hidden-dims 128 64 32 \
        --lr 0.0005 \
        --epsilon 0.9 \
        --epsilon-min 0.05 \
        --epsilon-decay 0.0001 \
        --batch-size 32 \
        --target-update-interval 500

    # Train with more frequent logging
    python scripts/train_rl_agent.py --env-id InvertedPendulum-v4 --log-interval 5

Available Arguments:
    --env-id: Gymnasium environment ID (default: InvertedPendulum-v5)
    --agent: RL agent type (default: dqn)
    --episodes: Total number of training episodes (default: 500)
    --log-interval: Interval for printing training progress (default: 10)
    --save-path: Path to save the trained model (default: models/learnt/dqn_inverted_pendulum.pth)
    --hidden-dims: Hidden layer dimensions (default: [128, 64])
    --lr: Learning rate (default: 0.0005)
    --gamma: Discount factor (default: 0.99)
    --epsilon: Initial exploration rate (default: 1.0)
    --epsilon-min: Minimum exploration rate (default: 0.005)
    --epsilon-decay: Epsilon decay rate (default: 0.00005)
    --buffer-capacity: Replay buffer capacity (default: 50000)
    --batch-size: Training batch size (default: 128)
    --target-update-interval: Frequency of target network updates (default: 200)
    --n-discrete-actions: Number of discrete actions for continuous environments (default: 15)
    --plot-save-path: Path to save the reward plot (default: plots/dqn_rewards.png)

Running Multiple Sessions:
    Use --num-sessions > 1 to run multiple training sessions sequentially with different
    random seeds. This is useful for observing the variance in training outcomes.
    An aggregate plot showing the mean and standard deviation of rewards across
    sessions will also be saved.

    --num-sessions: Number of independent training sessions to run (default: 1)
    --base-seed: Base random seed for reproducibility across sessions (default: None, generates random seeds)
"""

import sys
import argparse
import json
import datetime
import hashlib
from pathlib import Path
from collections import deque
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import random # Added for seeding

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the relevant controller (assuming only DQN for now)
from src.algorithms.learnt.dqn_controller import DQNController
# Import custom environments if needed (register them in __init__)
import src.environments.swing_up_envs
# Import the wrapper
from src.environments.wrappers import DiscretizeActionWrapper

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        # The two lines below might slow down training but ensure reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def get_run_id(config):
    """Generate a unique run ID based on key parameters and timestamp."""
    # Create a string with key configuration parameters
    config_str = f"{config.env_id}_{config.agent}_h{'-'.join(map(str, config.hidden_dims))}_e{config.episodes}_lr{config.lr}_g{config.gamma}_eps{config.epsilon_decay}_na{config.n_discrete_actions}"
    
    # Add a timestamp to ensure uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate a short hash to keep filename length reasonable
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    return f"{timestamp}_{config_hash}"

def train(config, seed, run_dir):
    """Train the RL agent for one session."""
    
    # --- Set Seed ---
    set_seed(seed)
    print(f"\n--- Starting Training Session with Seed: {seed} ---")

    # --- Environment Setup ---
    env_kwargs = {}
    if config.env_id == 'Pendulum-SwingUp':
        env_kwargs['reward_mode'] = config.reward_mode
        env_kwargs['center_penalty_weight'] = config.center_penalty_weight
        env_kwargs['limit_penalty'] = config.limit_penalty
        print(f"Using Pendulum-SwingUp specific kwargs: {env_kwargs}")
        
    try:
        env = gym.make(config.env_id, **env_kwargs)
    except gym.error.Error as e:
        print(f"Error creating environment {config.env_id} with kwargs {env_kwargs}: {e}")
        sys.exit(1)

    # Get state dimension (assuming Box observation space)
    if isinstance(env.observation_space, gym.spaces.Box):
        state_dim = env.observation_space.shape[0]
    else:
        raise NotImplementedError(f"Observation space {type(env.observation_space)} not supported.")

    # --- Action Space Handling ---
    # Check if action space is continuous (Box) and apply wrapper if needed
    if isinstance(env.action_space, gym.spaces.Box):
        print(f"Environment '{config.env_id}' has a continuous action space.")
        if config.agent == 'dqn':
            print(f"Applying action discretisation for DQN with {config.n_discrete_actions} bins.")
            env = DiscretizeActionWrapper(env, n_bins=config.n_discrete_actions)
            action_dim = env.action_space.n # Action dim is now n_bins
        else:
             # If we add other agents later that support continuous actions
             raise NotImplementedError(f"Agent '{config.agent}' not configured for continuous action space {type(env.action_space)}.")
             
    elif isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        raise NotImplementedError(f"Action space {type(env.action_space)} not supported.")

    print(f"Training agent '{config.agent}' on environment '{config.env_id}'")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Training for {config.episodes} episodes.")

    # --- Agent Setup ---
    if config.agent == 'dqn':
        agent = DQNController(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
            lr=config.lr,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_min=config.epsilon_min,
            epsilon_decay=config.epsilon_decay,
            buffer_capacity=config.buffer_capacity,
            batch_size=config.batch_size,
            target_update_interval=config.target_update_interval
        )
    else:
        raise ValueError(f"Unknown agent type: {config.agent}")

    # --- Training Loop ---
    all_episode_rewards = [] # Store all rewards for plotting
    episode_rewards_deque = deque(maxlen=config.log_interval) # Store recent rewards for logging
    total_steps = 0

    for i_episode in range(1, config.episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        steps_in_episode = 0

        while not terminated and not truncated:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store transition
            # Note: DQNController expects action to be a single value, not array
            # We retrieve it using .item() if it's a tensor, but here it's direct int
            agent.store_transition(state, action, reward, next_state, terminated)

            # Update agent
            agent.update()

            state = next_state
            episode_reward += reward
            total_steps += 1
            steps_in_episode += 1

        all_episode_rewards.append(episode_reward) # Store reward for this episode
        episode_rewards_deque.append(episode_reward) # Add to deque for logging average
        
        # --- Logging ---
        if i_episode % config.log_interval == 0:
            avg_reward = sum(episode_rewards_deque) / len(episode_rewards_deque) # Use deque for avg
            print(f"Episode: {i_episode}/{config.episodes} | "
                  f"Total Steps: {total_steps} | "
                  f"Avg Reward (Last {config.log_interval}): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print(f"Training finished after {config.episodes} episodes and {total_steps} steps.")

    # --- Saving Model ---
    session_model_save_path = run_dir / "models" / f"{Path(config.save_path).stem}_seed{seed}{Path(config.save_path).suffix}"
    session_model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.q_net.state_dict(), session_model_save_path)
    print(f"Trained model saved to: {session_model_save_path}")

    # --- Plotting Rewards ---
    plot_save_path = run_dir / "plots" / Path(config.plot_save_path).name # Use run_dir
    session_plot_save_path = plot_save_path.parent / plot_save_path.name
    session_plot_save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, config.episodes + 1), all_episode_rewards, label='Episode Reward')
    # Add a simple moving average
    if len(all_episode_rewards) >= config.log_interval:
        moving_avg = np.convolve(all_episode_rewards, np.ones(config.log_interval)/config.log_interval, mode='valid')
        plt.plot(range(config.log_interval, config.episodes + 1), moving_avg, label=f'{config.log_interval}-Episode Moving Avg', linestyle='--')
    
    plt.title(f"Training Rewards for {config.agent.upper()} on {config.env_id}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(session_plot_save_path)
    print(f"Individual session reward plot saved to: {session_plot_save_path}")
    plt.close() # Close the figure to free memory

    env.close()
    return all_episode_rewards # Return rewards for aggregation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent on a Gymnasium environment")

    # General Args
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v5", help="Gymnasium environment ID")
    parser.add_argument("--agent", type=str, default="dqn", choices=['dqn'], help="RL agent type")
    parser.add_argument("--episodes", type=int, default=500, help="Total number of training episodes")
    parser.add_argument("--log-interval", type=int, default=10, help="Interval for printing training progress")
    parser.add_argument("--save-path", type=str, default="models/learnt/dqn_inverted_pendulum.pth", help="Path to save the trained model")

    # DQN Hyperparameters (allow overriding defaults from DQNController)
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=[128, 64], help="Hidden layer dimensions")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-min", type=float, default=0.005, help="Minimum exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.0005, help="Epsilon decay rate")
    parser.add_argument("--buffer-capacity", type=int, default=50000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--target-update-interval", type=int, default=200, help="Frequency of target network updates (in steps)")

    # Action discretisation Args
    parser.add_argument("--n-discrete-actions", type=int, default=15, help="Number of discrete actions if wrapping a continuous environment")
    
    # Plotting Args
    parser.add_argument("--plot-save-path", type=str, default="plots/dqn_rewards.png", help="Base path to save the reward plot(s)")

    # --- Custom Env Args (Pendulum-SwingUp specific) ---
    parser.add_argument("--reward-mode", type=str, default='cos_theta', choices=['cos_theta', 'cos_theta_centered'], help="Reward mode for Pendulum-SwingUp env")
    parser.add_argument("--center-penalty-weight", type=float, default=0.1, help="Weight for center penalty in Pendulum-SwingUp")
    parser.add_argument("--limit-penalty", type=float, default=10.0, help="Penalty for hitting limits in Pendulum-SwingUp")

    # Multi-session Args
    parser.add_argument("--num-sessions", type=int, default=1, help="Number of independent training sessions to run")
    parser.add_argument("--base-seed", type=int, default=None, help="Base random seed for reproducibility. If None, random seeds are used.")

    args = parser.parse_args()

    # Convert hidden_dims list back to tuple for the controller
    args.hidden_dims = tuple(args.hidden_dims)
    
    # Generate a unique run ID for this training run
    run_id = get_run_id(args)
    
    # Create directories for this run
    run_dir = Path("runs") / run_id
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Update save paths to use the run directory
    original_save_path = Path(args.save_path)
    original_plot_path = Path(args.plot_save_path)
    
    # Use original filenames but in the new run directory
    args.save_path = str(models_dir / original_save_path.name)
    args.plot_save_path = str(plots_dir / original_plot_path.name)
    
    # Save configuration to a JSON file
    config_to_save = vars(args).copy()
    config_to_save['hidden_dims'] = list(config_to_save['hidden_dims'])  # Convert tuple to list for JSON
    # Clean up args that are not hyperparameters before saving
    config_to_save.pop('num_sessions', None)
    config_to_save.pop('base_seed', None)
    
    config_file = run_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"Training run ID: {run_id}")
    print(f"Configuration saved to: {config_file}")

    # --- Run Multiple Sessions ---
    all_sessions_rewards = []
    base_save_path = Path(args.save_path)
    base_plot_path = Path(args.plot_save_path)

    for i in range(args.num_sessions):
        session_seed = args.base_seed + i if args.base_seed is not None else random.randint(0, 1000000)
        
        # Run training for one session, passing run_dir
        session_rewards = train(args, session_seed, run_dir)
        all_sessions_rewards.append(session_rewards)

    print(f"\n--- Completed {args.num_sessions} Training Session(s) ---")

    # --- Aggregate Plotting (if more than one session) ---
    if args.num_sessions > 1 and all_sessions_rewards:
        print("Generating aggregate reward plot...")
        # Ensure all sessions have the same length (should be args.episodes)
        min_len = min(len(r) for r in all_sessions_rewards)
        rewards_array = np.array([r[:min_len] for r in all_sessions_rewards])

        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)

        episodes = np.arange(1, min_len + 1)

        agg_plot_save_path = plots_dir / f"{original_plot_path.stem}_aggregate{original_plot_path.suffix}"
        agg_plot_save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.plot(episodes, mean_rewards, label='Mean Reward')
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, label='Std Dev')
        
        plt.title(f"Aggregate Training Rewards ({args.num_sessions} Sessions) for {args.agent.upper()} on {args.env_id}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(agg_plot_save_path)
        print(f"Aggregate reward plot saved to: {agg_plot_save_path}")
        
        # Save run summary with key metrics
        summary = {
            "run_id": run_id,
            "final_mean_reward": float(mean_rewards[-1]),
            "final_std_reward": float(std_rewards[-1]),
            "max_mean_reward": float(np.max(mean_rewards)),
            "config_file": str(config_file),
            "plot_file": str(agg_plot_save_path)
        }
        
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Run summary saved to: {summary_file}")
        
        plt.close() 