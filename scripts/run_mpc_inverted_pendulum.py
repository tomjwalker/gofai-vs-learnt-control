import sys
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import imageio  # For saving video

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import gymnasium as gym
import numpy as np

# Import swing-up environments to register them
import src.environments.swing_up_envs

import matplotlib
matplotlib.use('TkAgg')
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

def save_video(frames, filename="inverted_pendulum_control.mp4"):
    """Save a list of frames as a video file."""
    imageio.mimsave(filename, frames, fps=50)  # 50 fps to match MuJoCo's rendering

def run_mpc_with_diagnostics(num_episodes=1, max_steps=1000, render_mode="rgb_array"):
    """
    Run the InvertedPendulum-v5 environment with your MPCController,
    collecting diagnostic data at each step and plotting afterwards.
    """
    # Create the environment with video-friendly render mode
    env = gym.make("Pendulum-SwingUp", render_mode=render_mode)
    
    # Initialize the viewer by rendering once
    env.reset()
    env.render()
    
    # Now we can access the viewer
    if hasattr(env.unwrapped, 'viewer'):
        print(f"env.unwrapped.viewer attributes: {dir(env.unwrapped.viewer)}")
        # Adjust camera view to zoom out
        env.unwrapped.viewer.cam.distance = 10.0  # Increase distance to zoom out
        env.unwrapped.viewer.cam.elevation = -45  # Adjust elevation angle
        env.unwrapped.viewer.cam.lookat = [0.0, 0.0, 0.0]  # Center the view
    
    controller = MPCController(
        N=100,  # Increased prediction horizon
        dt=0.01,  # Increased timestep for better prediction
        param_path="../src/environments/inverted_pendulum_params.json"
    )

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        obs = np.array(obs, dtype=np.float64)

        # We'll store step-by-step data in a list of dicts
        history = []
        frames = []  # Store video frames

        for step in range(max_steps):
            # 1) Compute action using your MPC's step()
            solver_outputs = controller.solve(obs)
            X_sol = solver_outputs["X_solution"]  # shape (4, N+1)
            U_sol = solver_outputs["U_solution"]  # shape (1, N)
            u_next = solver_outputs["u_next"]     # float

            # Print MPC diagnostics
            print(f"\nStep {step}:")
            print(f"Cost: {solver_outputs['cost']:.2f}")
            print(f"Constraint violation: {solver_outputs['constraint_violation']:.2e}")
            print(f"Max control in horizon: {np.max(np.abs(U_sol)):.2f}")

            # 2) Step environment
            #   Gym expects the action as e.g. [u_next], ensuring shape (1,)
            obs_next, reward, done, truncated, info = env.step([u_next])
            obs_next = np.array(obs_next, dtype=np.float64)

            # Capture frame for video
            frame = env.render()
            frames.append(frame)

            # 3) Record data for plotting
            step_data = {
                "obs": obs,           # the environment's current state
                "obs_next": obs_next,
                "X_sol": X_sol,       # the predicted state trajectory
                "U_sol": U_sol,       # the predicted control trajectory
                "u_next": u_next,     # the immediate control
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "cost": solver_outputs["cost"],
                "constraint_violation": solver_outputs["constraint_violation"]
            }
            history.append(step_data)

            obs = obs_next
            if done or truncated:
                print(f"\n[Episode {episode}] Finished in {step+1} steps, total reward = "
                      f"{sum(d['reward'] for d in history):.2f}")
                break

        # Save video
        video_filename = f"inverted_pendulum_episode_{episode}.mp4"
        save_video(frames, video_filename)
        print(f"Saved video to {video_filename}")

        # End of episode, do some plotting
        plot_diagnostics(history, episode=episode)

        # Create and show animated diagnostics
        anim = create_animated_diagnostics(history, episode)
        plt.show()

    env.close()

def plot_diagnostics(history, episode=0):
    """
    Given the list of step data from run_mpc_with_diagnostics,
    produce a few relevant plots:
    1) Environment states over time
    2) MPC horizon predictions at each step
    3) Control input over time
    4) Cost and constraint violation over time
    """

    # We'll assume the environment's state is 4D, and your X_sol is 4 x (N+1).
    # If your states differ, adjust accordingly.
    time_indices = np.arange(len(history))  # one index per environment step
    # For environment states, let's store them in an array of shape [steps, 4]
    env_states = np.array([step["obs"] for step in history])  # shape (steps, 4)
    # The immediate controls
    controls = np.array([step["u_next"] for step in history]) # shape (steps,)
    # Costs and constraint violations
    costs = np.array([step["cost"] for step in history])
    constraint_violations = np.array([step["constraint_violation"] for step in history])

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # 1) Plot environment states with better labels
    state_labels = ["Cart Position (m)", "Pole Angle (rad)", "Cart Velocity (m/s)", "Pole Angular Velocity (rad/s)"]
    for i in range(env_states.shape[1]):
        axs[0].plot(time_indices, env_states[:, i], label=state_labels[i])

    axs[0].set_title(f"Episode {episode}: Environment States")
    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("State Value")
    axs[0].legend()
    axs[0].grid(True)

    # Overplot horizon predictions as dotted lines
    for step_i, data in enumerate(history):
        X_sol = data["X_sol"]  # shape (4, N+1)
        horizon_times = step_i + np.arange(X_sol.shape[1])
        for i in range(X_sol.shape[0]):
            axs[0].plot(horizon_times, X_sol[i, :], linestyle="dotted", alpha=0.2)

    # 2) Plot the control input over time
    axs[1].plot(time_indices, controls, marker='o')
    axs[1].set_title("MPC Control Input Over Time")
    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Force (N)")
    axs[1].grid(True)

    # 3) Plot cost and constraint violation
    axs[2].plot(time_indices, costs, 'b-', label='Cost')
    axs[2].set_ylabel('Cost', color='b')
    axs[2].tick_params(axis='y', labelcolor='b')
    
    ax2 = axs[2].twinx()
    ax2.plot(time_indices, constraint_violations, 'r-', label='Constraint Violation')
    ax2.set_ylabel('Constraint Violation', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    axs[2].set_title("MPC Cost and Constraint Violation")
    axs[2].set_xlabel("Timestep")
    axs[2].grid(True)

    # Add legends
    lines1, labels1 = axs[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_mpc_with_diagnostics()
