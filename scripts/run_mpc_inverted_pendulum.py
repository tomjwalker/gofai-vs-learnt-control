import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import gymnasium as gym
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.algorithms.classic.mpc_controller import MPCController

def run_mpc_with_diagnostics(num_episodes=1, max_steps=200, render_mode=None):
    """
    Run the InvertedPendulum-v4 environment with your MPCController,
    collecting diagnostic data at each step and plotting afterwards.
    """

    # Create the environment with optional video-friendly render mode
    env = gym.make("InvertedPendulum-v4", render_mode=render_mode)
    controller = MPCController(
        N=15,
        dt=0.01,  # ensure this is consistent with env dt, if you rely on real-time
        param_path="../src/environments/inverted_pendulum_params.json"
    )

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        obs = np.array(obs, dtype=np.float64)

        # We'll store step-by-step data in a list of dicts
        history = []

        for step in range(max_steps):
            # 1) Compute action using your MPC's step()
            solver_outputs = controller.solve(obs)
            X_sol = solver_outputs["X_solution"]  # shape (4, N+1)
            U_sol = solver_outputs["U_solution"]  # shape (1, N)
            u_next = solver_outputs["u_next"]     # float

            # 2) Step environment
            #   Gym expects the action as e.g. [u_next], ensuring shape (1,)
            obs_next, reward, done, truncated, info = env.step([u_next])
            obs_next = np.array(obs_next, dtype=np.float64)

            # 3) Record data for plotting
            step_data = {
                "obs": obs,           # the environment's current state
                "obs_next": obs_next,
                "X_sol": X_sol,       # the predicted state trajectory
                "U_sol": U_sol,       # the predicted control trajectory
                "u_next": u_next,     # the immediate control
                "reward": reward,
                "done": done,
                "truncated": truncated
            }
            history.append(step_data)

            obs = obs_next
            if done or truncated:
                print(f"[Episode {episode}] Finished in {step+1} steps, total reward so far = "
                      f"{sum(d['reward'] for d in history):.2f}")
                break

        # End of episode, do some plotting
        plot_diagnostics(history, episode=episode)

    env.close()

def plot_diagnostics(history, episode=0):
    """
    Given the list of step data from run_mpc_with_diagnostics,
    produce a few relevant plots:
    1) Environment states over time
    2) MPC horizon predictions at each step
    3) Control input over time
    """

    # We'll assume the environment's state is 4D, and your X_sol is 4 x (N+1).
    # If your states differ, adjust accordingly.
    time_indices = np.arange(len(history))  # one index per environment step
    # For environment states, let's store them in an array of shape [steps, 4]
    env_states = np.array([step["obs"] for step in history])  # shape (steps, 4)
    # The immediate controls
    controls = np.array([step["u_next"] for step in history]) # shape (steps,)

    # 1) Plot environment states
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    # top subplot: environment states
    for i in range(env_states.shape[1]):
        ax[0].plot(time_indices, env_states[:, i], label=f"State {i}")

    ax[0].set_title(f"Episode {episode}: Environment States")
    ax[0].set_xlabel("Timestep")
    ax[0].set_ylabel("State Value")
    ax[0].legend()

    # Overplot horizon predictions as dotted lines
    # E.g. at each step, we have a horizon X_sol shape (4, N+1).
    # We'll shift the time axis for plotting the horizon into the future.
    for step_i, data in enumerate(history):
        X_sol = data["X_sol"]  # shape (4, N+1)
        # We'll create a separate time axis offset by 'step_i' for the horizon
        horizon_times = step_i + np.arange(X_sol.shape[1])
        for i in range(X_sol.shape[0]):
            ax[0].plot(horizon_times, X_sol[i, :], linestyle="dotted", alpha=0.2)

    # 2) Plot the control input over time
    ax[1].plot(time_indices, controls, marker='o')
    ax[1].set_title("MPC Control Input Over Time")
    ax[1].set_xlabel("Timestep")
    ax[1].set_ylabel("Action (u)")
    # Possibly also show predicted controls? U_sol at each step_i, shape (1, N).
    # We'll omit for simplicity, or show it similarly as dotted lines.

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_mpc_with_diagnostics()
