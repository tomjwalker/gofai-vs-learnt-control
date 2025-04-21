#!/usr/bin/env python
import sys
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import imageio  # For saving video
import argparse # Added
import datetime # Added
import hashlib  # Added
import json     # Added
import ast      # Added
import os       # Added

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import gymnasium as gym
import numpy as np
import casadi as ca

# Import swing-up environments to register them
import src.environments.swing_up_envs # Ensure registration happens
from src.environments.wrappers import InvertedPendulumComparisonWrapper # Keep for potential use

import matplotlib
matplotlib.use('TkAgg') # Consider making this conditional or configurable
import matplotlib.pyplot as plt

from src.algorithms.classic.mpc_controller import MPCController

def create_animated_diagnostics(history, episode=0):
    """
    Create an animated plot showing the evolution of states, forecasts, and controls over time.
    """
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    
    # Set up the plots
    state_labels = ["Cart Position (m)", "Pole Angle (rad)", "Cart Velocity (m/s)", "Pole Angular Velocity (rad/s)"]
    colors = ['b', 'g', 'r', 'c']
    
    # Initialize lines for actual states
    actual_lines = []
    for i in range(4):
        line, = axs[0].plot([], [], color=colors[i], label=f'Actual {state_labels[i]}')
        actual_lines.append(line)
    
    # Initialize lines for forecasts
    forecast_lines = []
    for i in range(4):
        line, = axs[0].plot([], [], '--', color=colors[i], alpha=0.5, label=f'Forecast {state_labels[i]}')
        forecast_lines.append(line)
    
    # Initialize control lines
    actual_control, = axs[1].plot([], [], 'b-', label='Actual Control')
    forecast_control, = axs[1].plot([], [], 'b--', alpha=0.5, label='Forecast Control')
    
    # Initialize cost and constraint lines
    cost_line, = axs[2].plot([], [], 'b-', label='Cost')
    constraint_line, = axs[2].plot([], [], 'r-', label='Constraint Violation')
    
    # Set up axes
    axs[0].set_title(f"Episode {episode}: Environment States")
    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("State Value")
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].set_title("Control Input")
    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Force (N)")
    axs[1].grid(True)
    axs[1].legend()
    
    axs[2].set_title("Cost and Constraint Violation")
    axs[2].set_xlabel("Timestep")
    axs[2].set_ylabel("Value")
    axs[2].grid(True)
    axs[2].legend()
    
    # Pre-compute axis limits
    all_states = np.array([step["obs"] for step in history])
    all_controls = np.array([step["u_next"] for step in history])
    all_costs = np.array([step["cost"] for step in history])
    all_constraints = np.array([step["constraint_violation"] for step in history])
    
    # Set fixed axis limits
    axs[0].set_xlim(0, len(history))
    axs[0].set_ylim(all_states.min() - 0.1, all_states.max() + 0.1)
    
    axs[1].set_xlim(0, len(history))
    axs[1].set_ylim(all_controls.min() - 0.1, all_controls.max() + 0.1)
    
    axs[2].set_xlim(0, len(history))
    axs[2].set_ylim(min(all_costs.min(), all_constraints.min()) - 0.1, 
                   max(all_costs.max(), all_constraints.max()) + 0.1)
    
    # Animation update function
    def update(frame):
        # Update actual state lines
        time_indices = np.arange(frame + 1)
        for i in range(4):
            actual_lines[i].set_data(time_indices, [step["obs"][i] for step in history[:frame+1]])
        
        # Update forecast lines
        if frame < len(history):
            forecast_times = np.arange(frame, frame + history[frame]["X_sol"].shape[1])
            for i in range(4):
                forecast_lines[i].set_data(forecast_times, history[frame]["X_sol"][i, :])
        
        # Update control lines
        actual_control.set_data(time_indices, [step["u_next"] for step in history[:frame+1]])
        if frame < len(history):
            forecast_control.set_data(forecast_times[:-1], history[frame]["U_sol"][0, :])
        
        # Update cost and constraint lines
        cost_line.set_data(time_indices, [step["cost"] for step in history[:frame+1]])
        constraint_line.set_data(time_indices, [step["constraint_violation"] for step in history[:frame+1]])
        
        return actual_lines + forecast_lines + [actual_control, forecast_control, cost_line, constraint_line]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(history), interval=200, blit=True)
    
    return anim

def get_run_id(config):
    """Generate a unique run ID based on key parameters and timestamp."""
    # Use relevant MPC parameters
    config_str_parts = [
        config.env_id,
        f"N{config.horizon}",
        # Represent Q and R concisely if they were modified
        # (This assumes Q is diagonal and R is scalar for simplicity)
        f"Qdiag{config.q_diag}".replace(" ", ""), 
        f"R{config.r_val}",
        f"eps{config.num_episodes}"
    ]
        
    config_str = "_".join(config_str_parts)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use PID for potential parallel runs
    config_hash = hashlib.md5(f"{config_str}_{os.getpid()}".encode()).hexdigest()[:8]
    return f"{timestamp}_{config_hash}"

def save_video(frames, filename="episode_video.mp4"):
    """Save a list of frames as a video file."""
    print(f"Saving video with {len(frames)} frames to {filename}...")
    try:
        imageio.mimsave(filename, frames, fps=50)  # 50 fps approx real-time for 0.02s step
        print("Video saved successfully.")
    except Exception as e:
        print(f"Error saving video: {e}")

def plot_diagnostics(history, plots_dir, episode=0):
    """Generates and saves diagnostic plots for a single episode."""
    print(f"Generating diagnostic plots for Episode {episode}...")
    plots_dir.mkdir(parents=True, exist_ok=True) # Ensure plots dir exists
    time_indices = np.arange(len(history))
    env_states = np.array([step["obs"] for step in history])
    controls = np.array([step["u_next"] for step in history])
    costs = np.array([step["cost"] for step in history if "cost" in step]) # Handle potential missing keys
    constraint_violations = np.array([step["constraint_violation"] for step in history if "constraint_violation" in step])

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot States
    state_labels = [f"State_{i}" for i in range(env_states.shape[1])] # Generic labels
    if history:
        # Use specific labels if obs shape matches known envs
        if env_states.shape[1] == 4: # InvertedPendulum
             state_labels = ["Cart Pos (x)", "Pole Angle (th)", "Cart Vel (x_dot)", "Pole Vel (th_dot)"]
        # Add elif for other envs if needed
    
    for i in range(env_states.shape[1]):
        axs[0].plot(time_indices, env_states[:, i], label=state_labels[i])
    axs[0].set_title(f"Episode {episode}: Environment States")
    axs[0].set_ylabel("State Value")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Control
    axs[1].plot(time_indices, controls, marker='.', linestyle='-')
    axs[1].set_title("MPC Control Input Over Time")
    axs[1].set_ylabel("Force (N)") # Assuming force control
    axs[1].grid(True)

    # Plot Cost and Constraint Violation
    if len(costs) == len(time_indices) and len(constraint_violations) == len(time_indices):
        ax2 = axs[2].twinx()
        axs[2].plot(time_indices, costs, 'b-', label='Cost')
        axs[2].set_ylabel('Cost', color='b')
        axs[2].tick_params(axis='y', labelcolor='b')
        ax2.plot(time_indices, constraint_violations, 'r-', label='Constraint Violation')
        ax2.set_ylabel('Constraint Violation', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        axs[2].set_title("MPC Cost and Constraint Violation")
        lines1, labels1 = axs[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        axs[2].set_title("Cost/Constraint Data Unavailable or Mismatched Length")
    axs[2].set_xlabel("Timestep")
    axs[2].grid(True)

    fig.tight_layout()
    plot_filename = plots_dir / f"episode_{episode}_plots.png" # Save to plots_dir
    plt.savefig(plot_filename)
    print(f"Diagnostic plot saved to: {plot_filename}")
    plt.close(fig) # Close figure to free memory

def run_mpc_experiment(args):
    """Runs the MPC experiment based on parsed arguments."""

    # --- Run Setup --- 
    run_id = get_run_id(args)
    run_dir = Path(args.save_dir) / run_id
    # Create subdirectories for outputs
    videos_dir = run_dir / "videos"
    plots_dir = run_dir / "plots" 
    videos_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    # No models_dir needed for MPC
    
    print(f"Starting MPC Run: {run_id}")
    print(f"Saving results to: {run_dir}")

    # Save configuration
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"Configuration saved to {config_path}")

    # --- Environment Setup --- 
    # Use wrapper for swing-up task if specified
    if args.env_id == "Pendulum-SwingUp":
        # Swing-up env has its own reset logic, use standard make
        try:
            # Need to ensure Pendulum-SwingUp is registered
            src.environments.swing_up_envs.pendulum_su.register_pendulum_swing_up()
            env = gym.make(args.env_id, render_mode=args.render_mode)
            print("Using Pendulum-SwingUp environment.")
        except gym.error.Error as e:
            print(f"Error creating {args.env_id}. Is it registered? Error: {e}")
            sys.exit(1)
    else:
        # For standard balance or comparison, use wrapper to control reset/termination
        try:
             base_env = gym.make(args.env_id, render_mode=args.render_mode)
             # Apply wrapper to allow setting initial state and prevent termination
             env = InvertedPendulumComparisonWrapper(base_env)
             print(f"Using standard {args.env_id} with ComparisonWrapper.")
        except gym.error.Error as e:
             print(f"Error creating {args.env_id}. Error: {e}")
             sys.exit(1)
             
    # Determine state/action dimensions (assuming Box spaces)
    try:
        state_dim = env.observation_space.shape[0]
        # MPC assumes continuous action space
        control_dim = env.action_space.shape[0] 
        print(f"Env: {args.env_id}, State Dim: {state_dim}, Control Dim: {control_dim}")
    except Exception as e:
        print(f"Could not determine state/control dimensions: {e}")
        env.close()
        sys.exit(1)
        
    # Get timestep for MPC
    try:
        mpc_dt = env.unwrapped.dt
        print(f"Using environment timestep for MPC: dt = {mpc_dt}")
    except AttributeError:
        print("Warning: Could not determine env dt. Using default dt=0.02 for MPC.")
        mpc_dt = 0.02 

    # --- MPC Controller Setup --- 
    controller = MPCController(
        N=args.horizon,
        dt=mpc_dt,
        param_path=args.param_path 
    )
    print(f"MPC Initialized with N={controller.N}, dt={controller.dt}")

    # Override Q and R if provided via CLI args
    try:
        if args.q_diag:
             q_diag_vals = [float(x) for x in args.q_diag]
             if len(q_diag_vals) == state_dim:
                 controller.Q = ca.diag(q_diag_vals)
                 controller.Q_terminal = 5.0 * controller.Q # Keep proportional
                 print(f"Overrode MPC Q matrix (diag): {q_diag_vals}")
             else:
                 print(f"Warning: --q-diag length ({len(q_diag_vals)}) != state_dim ({state_dim}). Using default Q.")
        if args.r_val:
             if control_dim == 1: # Scalar R
                 controller.R = ca.DM([args.r_val])
                 print(f"Overrode MPC R matrix (scalar): {args.r_val}")
             else: # Need diagonal R if control_dim > 1
                 r_diag = [args.r_val] * control_dim
                 controller.R = ca.diag(r_diag)
                 print(f"Overrode MPC R matrix (diag): {r_diag}")
    except Exception as e:
        print(f"Error overriding Q/R matrices: {e}. Using defaults.")

    # --- Run Episodes --- 
    episode_rewards = []
    episode_lengths = []

    for episode in range(args.num_episodes):
        print(f"\n--- Starting Episode {episode} --- ")
        # Reset environment - use wrapper's initial state if applicable
        if isinstance(env, InvertedPendulumComparisonWrapper) and args.env_id != "Pendulum-SwingUp":
            # Force starting state for non-swingup tasks using the wrapper
            initial_state = np.array([0.0, 0.1, 0.0, 0.0]) 
            obs, info = env.reset(initial_state=initial_state)
            print(f"Resetting env with wrapper state: {initial_state}")
        else:
            obs, info = env.reset() # Use env's default reset (e.g., for swing-up)
            print(f"Resetting env with default state: {obs}")

        obs = np.array(obs, dtype=np.float64)
        history = []
        frames = []
        ep_reward = 0.0

        for step in range(args.max_steps):
            # Compute action
            solver_outputs = controller.solve(obs)
            u_next = solver_outputs["u_next"] 
            action_to_apply = [u_next] # Gym expects list/array
            if control_dim > 1:
                 action_to_apply = u_next # If solve returns multi-dim control

            # Step environment
            obs_next, reward, terminated, truncated, info = env.step(action_to_apply)
            obs_next = np.array(obs_next, dtype=np.float64)
            ep_reward += reward

            # Render and store frame
            if args.render_mode == "rgb_array":
                frame = env.render()
                frames.append(frame)

            # Record data
            step_data = solver_outputs.copy() # Start with solver outputs
            step_data.update({
                "obs": obs,
                "obs_next": obs_next,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })
            history.append(step_data)

            obs = obs_next
            if terminated or truncated:
                break

        # End of episode
        episode_rewards.append(ep_reward)
        episode_lengths.append(step + 1)
        print(f"Episode {episode} Finished: Length={step+1}, Reward={ep_reward:.2f}")

        # Save video for this episode to videos_dir
        if frames:
            video_filename = videos_dir / f"episode_{episode}.mp4" # Save to videos_dir
            save_video(frames, str(video_filename))
            
        # Save diagnostic plots for this episode
        if args.plot_diagnostics:
             plot_diagnostics(history, plots_dir, episode=episode) # Pass plots_dir

        # Save animated diagnostics for this episode
        if args.save_animated_diagnostics:
            print(f"Generating animated diagnostics for Episode {episode}...")
            try:
                anim = create_animated_diagnostics(history, episode=episode)
                anim_filename = plots_dir / f"episode_{episode}_animated_diagnostics.gif"
                anim.save(str(anim_filename), writer='pillow', fps=10) # Adjust fps as needed
                print(f"Animated diagnostics saved to: {anim_filename}")
                plt.close(anim._fig) # Close the animation figure
            except Exception as e:
                print(f"Error saving animated diagnostics: {e}")

    # --- Save Summary --- 
    summary = {
        "run_id": run_id,
        "env_id": args.env_id,
        "num_episodes": args.num_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "config": vars(args) # Include args used for the run
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, 'w') as f:
        # Convert numpy types for JSON compatibility if necessary
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    print(f"\nRun summary saved to {summary_path}")

    env.close()
    print("Experiment Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MPC controller on a Gymnasium environment.")

    # --- Experiment Args --- 
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v5", help="Gymnasium environment ID")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["human", "rgb_array", None], help="Rendering mode")
    parser.add_argument("--save-dir", type=str, default="runs/MPC", help="Base directory to save run results")
    parser.add_argument("--plot-diagnostics", action="store_true", default=True, help="Generate diagnostic plots for each episode (default: True)")
    parser.add_argument("--save-animated-diagnostics", action="store_true", default=True, help="Generate and save animated diagnostic plots for each episode (default: True)")

    # --- MPC Args --- 
    parser.add_argument("--param-path", type=str, default="src/environments/inverted_pendulum_params.json", help="Path to environment parameters JSON for MPC")
    parser.add_argument("--horizon", "-N", type=int, default=30, help="MPC prediction horizon")
    # Allow overriding Q (diagonal) and R (scalar/diagonal)
    parser.add_argument("--q-diag", type=str, nargs='+', default=None, help="Diagonal elements for Q matrix (e.g., 1.0 20.0 5.0 10.0)")
    parser.add_argument("--r-val", type=float, default=None, help="Value for R matrix (scalar or diagonal if control_dim > 1)")

    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    run_mpc_experiment(args)
