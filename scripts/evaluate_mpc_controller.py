#!/usr/bin/env python
"""
Runs Model Predictive Control (MPC) experiments on Gymnasium environments, 
primarily focusing on Inverted Pendulum tasks (stabilization and swing-up).

This script handles environment setup, MPC controller initialization, 
episode execution, data logging, and generation of diagnostic plots and videos.

It uses sensible default configurations (cost function, initial guess strategy, 
parameter file) based on the specified environment ID, but these can be 
overridden via command-line arguments.

Example Usage:

1. Standard Inverted Pendulum Stabilization (defaults handle cost/guess/params):
   -----------------------------------------------------------------------------
   python scripts/evaluate_mpc_controller.py \
       --env-id InvertedPendulum-v5 \
       --num-episodes 3 \
       --max-steps 500 \
       --q-diag 10.0 1.0 0.1 0.1 \
       --r-val 0.1 \
       --plot-diagnostics \
       --save-animated-diagnostics

2. Inverted Pendulum Swing-Up (defaults handle cost/guess/params):
   -----------------------------------------------------------------
   # First, ensure the swing-up parameters with unlimited pole angle exist:
   # python src/environments/extract_params.py --unlimit-pole --output-suffix _swingup

   python scripts/evaluate_mpc_controller.py \
       --env-id Pendulum-SwingUp \
       --num-episodes 1 \
       --max-steps 500 \
       --horizon 80 \
       --q-diag 0.1 5 0.1 0.1 \
       --r-val 0.01 \
       --q-terminal-multiplier 10 \
       --plot-diagnostics \
       --save-animated-diagnostics

Notes:
- Use '--render-mode human' for real-time visualization (can be slow).
- Adjust '--q-diag', '--r-val', '--q-terminal-multiplier', and '--horizon' 
  to tune MPC performance for specific tasks.
- Check the 'runs/MPC/' directory for saved results (plots, videos, configs, summaries).
"""

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
# import src.environments.swing_up_envs # Old way
# Explicitly import and call registration function
from src.environments.swing_up_envs.pendulum_su import register_pendulum_swing_up
register_pendulum_swing_up() 

from src.environments.wrappers import InvertedPendulumComparisonWrapper # Keep for potential use

import matplotlib
matplotlib.use('TkAgg') # Consider making this conditional or configurable
import matplotlib.pyplot as plt

from src.algorithms.classic.mpc_controller import MPCController
# Import the refactored plotting function
from src.utils.plotting import plot_diagnostics 

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
    all_controls = np.array([step.get("u_next", np.nan) for step in history]) # Use get for safety
    all_costs = np.array([step.get("cost", np.nan) for step in history])
    all_constraints = np.array([step.get("constraint_violation", np.nan) for step in history])
    finite_costs_constraints = np.concatenate([all_costs, all_constraints])

    # Set fixed axis limits, using finite values or defaults
    axs[0].set_xlim(0, len(history))
    axs[0].set_ylim((all_states.min() - 0.1) if all_states.size > 0 else -1,
                    (all_states.max() + 0.1) if all_states.size > 0 else 1)
    
    axs[1].set_xlim(0, len(history))
    axs[1].set_ylim((all_controls.min() - 0.1) if all_controls.size > 0 else -1,
                    (all_controls.max() + 0.1) if all_controls.size > 0 else 1)
    
    axs[2].set_xlim(0, len(history))
    axs[2].set_ylim((finite_costs_constraints.min() - 0.1 * abs(finite_costs_constraints.min())) if finite_costs_constraints.size > 0 else 0, 
                   (finite_costs_constraints.max() + 0.1 * abs(finite_costs_constraints.max())) if finite_costs_constraints.size > 0 else 1)
    
    # Animation update function
    def update(frame):
        # Update actual state lines
        time_indices = np.arange(frame + 1)
        current_history = history[:frame+1]
        for i in range(4):
            actual_lines[i].set_data(time_indices, [step["obs"][i] for step in current_history])
        
        # Update forecast lines - Check if data exists for this frame
        forecast_data_missing = False # Flag
        if frame < len(history) and "X_sol" in history[frame] and history[frame]["X_sol"] is not None:
            forecast_steps = history[frame]["X_sol"].shape[1]
            forecast_times = np.arange(frame, frame + forecast_steps)
            for i in range(4):
                forecast_lines[i].set_data(forecast_times, history[frame]["X_sol"][i, :])
        else: # Clear forecast lines if no data
             forecast_data_missing = True
             for i in range(4):
                 forecast_lines[i].set_data([], [])

        # Update control lines
        actual_control.set_data(time_indices, [step.get("u_next", np.nan) for step in current_history]) # Use get
        # Check if data exists
        if frame < len(history) and "U_sol" in history[frame] and history[frame]["U_sol"] is not None:
            if 'forecast_steps' in locals(): 
                 forecast_control.set_data(forecast_times[:-1], history[frame]["U_sol"][0, :])
            else: 
                 forecast_data_missing = True # Mark as missing if X_sol wasn't plotted
                 forecast_control.set_data([], [])
        else: # Clear forecast control line if no data
             forecast_data_missing = True
             forecast_control.set_data([], [])

        # Print warning if forecast data was missing for this frame
        if forecast_data_missing and frame > 0: # Avoid warning on frame 0 if solver fails instantly
             # Print only occasionally to avoid spamming console
             if frame % 50 == 0: 
                 print(f"\nWarning: Forecast data missing for frame {frame}. Solver might be failing.")

        # Update cost and constraint lines - Check for existence
        costs = [step.get("cost", np.nan) for step in current_history]
        constraints = [step.get("constraint_violation", np.nan) for step in current_history]
        cost_line.set_data(time_indices, costs)
        constraint_line.set_data(time_indices, constraints)
        
        # Dynamically update axes limits for cost/constraint if needed (optional)
        # This prevents extreme values from making plot useless if solver fails early
        valid_costs = [c for c in costs if c is not None and np.isfinite(c)]
        valid_constraints = [c for c in constraints if c is not None and np.isfinite(c)]
        if valid_costs or valid_constraints:
            min_val = min(valid_costs + valid_constraints) if (valid_costs + valid_constraints) else 0
            max_val = max(valid_costs + valid_constraints) if (valid_costs + valid_constraints) else 1
            axs[2].set_ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))
            
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
    # The registration logic now handles applying the correct wrappers based on ID.
    # We simply need to call gym.make.
    try:
        # Ensure registration has happened (called at the top of the script now)
        env = gym.make(args.env_id, render_mode=args.render_mode)
        print(f"Successfully created environment: {args.env_id}")
        # Determine if it's a swing-up task for later logic (e.g., reset messages)
        is_swingup_task = "SwingUp" in args.env_id
    except gym.error.Error as e:
        print(f"Error creating environment '{args.env_id}'. Is it registered correctly? Error: {e}")
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
        dt_env = env.unwrapped.dt
        print(f"Using environment timestep for env steps: dt_env = {dt_env}")
    except AttributeError:
        print("Warning: Could not determine env dt. Assuming dt_env=0.02.")
        dt_env = 0.02 
        
    # Validate controller dt
    dt_controller = args.dt_controller
    if dt_controller < dt_env:
        print(f"Warning: dt_controller ({dt_controller}) is less than dt_env ({dt_env}). Setting dt_controller = dt_env.")
        dt_controller = dt_env
        
    # Calculate ZOH steps
    hold_steps = max(1, int(round(dt_controller / dt_env)))
    print(f"Controller dt: {dt_controller:.4f}, Env dt: {dt_env:.4f} => ZOH for {hold_steps} steps.")

    # --- MPC Controller Setup --- 
    # Convert q_diag from string list to float list if provided
    q_diag_vals = None
    if args.q_diag:
        try:
            q_diag_vals = [float(x) for x in args.q_diag]
            if len(q_diag_vals) != state_dim:
                print(f"Warning: --q-diag length ({len(q_diag_vals)}) != state_dim ({state_dim}). Using default Q in MPCController.")
                q_diag_vals = None # Revert to None so controller uses its default
        except ValueError as e:
            print(f"Warning: Invalid value in --q-diag: {e}. Using default Q in MPCController.")
            q_diag_vals = None

    controller = MPCController(
        N=args.horizon,
        dt_controller=dt_controller, # Use the specified controller dt
        param_path=args.param_path,
        cost_type=args.cost_type,
        guess_type=args.guess_type,
        # Pass weights directly from args (q_diag_vals might be None)
        q_diag=q_diag_vals, 
        r_val=args.r_val, # Pass r_val (might be None)
        q_terminal_multiplier=args.q_terminal_multiplier # Pass multiplier
    )
    # The Q/R weights are now set during __init__ based on args passed
    print(f"MPC Initialized via args with: N={controller.N}, dt={controller.dt_controller}, CostType={controller.cost_type}, GuessType={args.guess_type}")
    # Q/R values are printed within MPCController __init__

    # --- Run Episodes --- 
    episode_rewards = []
    episode_lengths = []

    for episode in range(args.num_episodes):
        print(f"\n--- Starting Episode {episode} --- ")
        # Reset environment - use wrapper's initial state if applicable
        # The PendulumSwingUp wrapper (applied internally via make_env) handles the reset logic.
        obs, info = env.reset() 
        # Print appropriate reset message
        if is_swingup_task:
             print(f"Resetting env for swing-up: {args.env_id}")
             # Obs should be near [0, -pi, 0, 0] after wrapper reset
        else:
             print(f"Resetting env {args.env_id} with default state: {obs}")

        obs = np.array(obs, dtype=np.float64)
        history = []
        frames = []
        ep_reward = 0.0
        last_action = np.zeros(control_dim) # Initialize last action

        for step in range(args.max_steps):
            
            # --- Decide whether to solve or hold --- 
            if step % hold_steps == 0:
                # Time to solve MPC
                # print(f"Step {step}: Solving MPC...") # Optional debug
                solver_outputs = controller.solve(obs)
                u_next = solver_outputs["u_next"] 
                last_action = np.array([u_next]) # Store the new action (as numpy array)
                if control_dim > 1:
                     last_action = u_next # If solve returns multi-dim control
            else:
                # Apply previous action (Zero-Order Hold)
                # print(f"Step {step}: Holding action {last_action}") # Optional debug
                # We need to pass solver_outputs structure from the *last* solve for logging
                # Get it from history if possible, otherwise use placeholders
                if history:
                    solver_outputs = {k: v for k, v in history[-1].items() if k in ["X_solution", "U_solution", "cost", "constraint_violation", "solver_status"]}
                    solver_outputs['u_next'] = last_action.item() if control_dim == 1 else last_action # Log the held action
                else: # Should not happen after first step
                     solver_outputs = {"u_next": last_action.item() if control_dim == 1 else last_action, "cost": np.nan, "constraint_violation": np.nan, "solver_status": "held"}
            # ----------------------------------------

            # Ensure action has correct shape for env.step
            action_to_apply = last_action.reshape((control_dim,)) 
            
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
            
        # Save diagnostic plots for this episode using imported function
        if args.plot_diagnostics:
             # plot_diagnostics(history, plots_dir, episode=episode) # Original call
             plot_diagnostics(history, plots_dir, episode=episode, plot_cost=True) # Pass plot_cost=True for MPC

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
    parser.add_argument("--cost-type", type=str, default=None, choices=['quadratic', 'pendulum_swingup', 'pendulum_atan2'], help="Cost function type for MPC (default: depends on env)")
    parser.add_argument("--guess-type", type=str, default=None, choices=['basic', 'warmstart', 'pendulum_heuristic', 'hybrid'], help="Initial guess strategy for MPC (default: depends on env)")
    parser.add_argument("--param-path", type=str, default=None, help="Path to environment parameters JSON for MPC (default: depends on env)")
    parser.add_argument("--horizon", "-N", type=int, default=30, help="MPC prediction horizon")
    parser.add_argument("--q-diag", type=str, nargs='+', default=None, help="Diagonal elements for Q matrix (e.g., 1.0 20.0 5.0 10.0)")
    parser.add_argument("--r-val", type=float, default=None, help="Value for R matrix (scalar or diagonal if control_dim > 1)")
    parser.add_argument("--q-terminal-multiplier", type=float, default=5.0, help="Multiplier for Q to get Q_terminal (default: 5.0)")
    parser.add_argument("--dt-controller", type=float, default=0.02, help="Timestep used for MPC internal prediction (default: 0.02)")

    args = parser.parse_args()
    
    # --- Set Default Strategies and Param Path based on Environment --- 
    DEFAULT_CONFIGS = {
        "InvertedPendulum-v5": {
            "cost": "quadratic", 
            "guess": "warmstart", 
            "params": "src/environments/inverted_pendulum_params.json"
        },
        "Pendulum-SwingUp": { # This key is now just for reference, logic below uses the unlimited one
            "cost": "pendulum_atan2", 
            "guess": "pendulum_heuristic",
            "params": "src/environments/inverted_pendulum_params_swingup.json" 
        },
        # Add entry for the unlimited version which will actually be used by default logic now
        "Pendulum-SwingUpUnlimited-v0": {
            "cost": "pendulum_atan2", 
            "guess": "pendulum_heuristic",
            # Assumes you generated this file using --unlimit-pole!
            "params": "src/environments/inverted_pendulum_params_swingup.json" 
        },
        # Add entries for DoublePendulum tasks later
    }
    
    # Set defaults if not provided
    # Determine the target env_id for default lookup
    target_env_id_for_defaults = args.env_id
    # Special case: If user asks for Pendulum-SwingUp, use the Unlimited version for MPC defaults
    if args.env_id == "Pendulum-SwingUp-v0": 
         print(f"Note: Env ID '{args.env_id}' requested, but using defaults for 'Pendulum-SwingUpUnlimited-v0' for MPC.")
         target_env_id_for_defaults = "Pendulum-SwingUpUnlimited-v0"
    
    if target_env_id_for_defaults in DEFAULT_CONFIGS:
        defaults = DEFAULT_CONFIGS[target_env_id_for_defaults]
        if args.cost_type is None:
            args.cost_type = defaults["cost"]
            print(f"Using default cost_type '{args.cost_type}' for env '{args.env_id}' (based on {target_env_id_for_defaults})")
        if args.guess_type is None:
            args.guess_type = defaults["guess"]
            print(f"Using default guess_type '{args.guess_type}' for env '{args.env_id}' (based on {target_env_id_for_defaults})")
        if args.param_path is None:
            args.param_path = defaults["params"]
            print(f"Using default param_path '{args.param_path}' for env '{args.env_id}' (based on {target_env_id_for_defaults})")
    else:
        # Fallbacks if env not in map
        print(f"Warning: Env '{args.env_id}' not in DEFAULT_CONFIGS map.")
        if args.cost_type is None: args.cost_type = "quadratic"
        if args.guess_type is None: args.guess_type = "warmstart"
        if args.param_path is None: args.param_path = "src/environments/inverted_pendulum_params.json"
        print(f"Using fallback defaults: cost='{args.cost_type}', guess='{args.guess_type}', params='{args.param_path}'.")
            
    # --- Proceed with Experiment --- 
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    run_mpc_experiment(args)
