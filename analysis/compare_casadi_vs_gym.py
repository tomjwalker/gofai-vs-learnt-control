print("--- Script starting ---")
"""
Compares the dynamics of an Inverted Pendulum simulated using CasADi and Gymnasium.

Runs two simulations:
1. Zero control input: Observes the natural dynamics starting near the upright position.
2. Bang-bang control: Applies a predefined sequence of max positive/negative forces.

For each selected simulation, it generates and optionally saves:
- An animation of the CasADi simulation (as a GIF), using the specified integration method.
- Plots comparing the state variables (position, angle, velocities) over time
  between the CasADi model and the Gymnasium environment.

Usage:
    # Run both simulations (RK4 default), display plots, default 10s unforced duration
    python analysis/compare_casadi_vs_gym.py

    # Run only the unforced simulation with Euler integration for 5 seconds and save
    python analysis/compare_casadi_vs_gym.py --sim-type unforced --integration euler --duration 5 --save

    # Run only the forced (bang-bang) simulation (RK4) and save the animation
    python analysis/compare_casadi_vs_gym.py --sim-type forced --save
"""
# analysis/compare_casadi_vs_gym.py
import sys
import argparse # Added argparse
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium as gym
import casadi as ca

# Add the project root to the Python path to allow importing from src
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary functions/classes from the project
from src.utils.parameters import load_pendulum_params
from src.environments.wrappers import InvertedPendulumComparisonWrapper
from src.environments.pendulum_dynamics import pendulum_dynamics # Import the dynamics function


print("Running CasADi and Gym dynamics simulation and comparison...")

# --- Configuration ---
ENV_ID = "InvertedPendulum-v5"
# Use paths relative to project root
PARAM_PATH = "src/environments/pendulum_params.json"
# SIM_STEPS = 2500 # Removed fixed steps
DEFAULT_DURATION_UNFORCED = 10.0 # Default duration in seconds
CASADI_INITIAL_STATE = np.array([0.0, 0.1, 0.0, 0.0])
CONTROL_INPUT = np.array([0.0])
INTEGRATION_METHOD = 'rk4'

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Compare CasADi and Gym Pendulum simulations.")
parser.add_argument("--save", action='store_true',
                    help="Save animations as GIF files in the 'analysis' directory.")
parser.add_argument("--sim-type", choices=['all', 'unforced', 'forced'], default='all',
                    help="Which simulation type to run and display ('unforced', 'forced', or 'all').")
parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_UNFORCED,
                    help="Duration (in seconds) for the unforced simulation.")
parser.add_argument("--integration", choices=['rk4', 'euler'], default='rk4',
                    help="Integration method for the CasADi simulation ('rk4' or 'euler').")
args = parser.parse_args()

# --- Load Parameters for CasADi ---
if not os.path.exists(PARAM_PATH):
    print(f"Error: Parameter file not found at {PARAM_PATH}")
    exit()
params = load_pendulum_params(PARAM_PATH)
pole_vis_length = params.get('pole_length', 0.6) # Use full length for vis

# --- Simulate Gymnasium Environment ---
env = InvertedPendulumComparisonWrapper(gym.make(ENV_ID))

# --- Get the ACTUAL timestep from the environment ---
DT = env.unwrapped.dt # Access dt from the unwrapped environment
print(f"Using environment timestep DT = {DT}")

# --- Calculate SIM_STEPS based on duration and DT ---
SIM_STEPS = int(args.duration / DT)
print(f"Running unforced simulation for {args.duration}s ({SIM_STEPS} steps).")

# --- Simulation 1: Unforced (Zero Control) ---
if args.sim_type in ['all', 'unforced']:
    print("\n--- Running Simulation 1: Unforced --- ")
    obs_gym, info_gym = env.reset(initial_state=CASADI_INITIAL_STATE)
    print(f"Gym reset state: {obs_gym}")
    gym_states = np.zeros((SIM_STEPS + 1, len(obs_gym))) # Store raw observations
    gym_states[0, :] = obs_gym
    sim_steps_actual = SIM_STEPS # Keep track of actual steps if truncated
    for i in range(SIM_STEPS):
        action_gym = [CONTROL_INPUT[0]]
        obs_gym, reward_gym, terminated_gym, truncated_gym, info_gym = env.step(action_gym)
        gym_states[i + 1, :] = obs_gym
        if truncated_gym:
            print(f"Gym simulation truncated at step {i+1}.")
            sim_steps_actual = i
            gym_states = gym_states[:sim_steps_actual+1, :]
            break
    env.close()
    print("Gym simulation complete.")

    # --- Setup CasADi Function ---
    # (Define this once, but only simulate if needed)
    x_sym = ca.SX.sym('x', 4)
    u_sym = ca.SX.sym('u', 1)
    x_next_sym = pendulum_dynamics(x_sym, u_sym, DT, params, integration_method=args.integration)
    dynamics_func = ca.Function('dynamics', [x_sym, u_sym], [x_next_sym])

    # --- Simulate CasADi Dynamics ---
    print(f"Simulating CasADi model ({args.integration}) for {sim_steps_actual} steps...")
    casadi_states = np.zeros((sim_steps_actual + 1, 4))
    casadi_states[0, :] = CASADI_INITIAL_STATE
    current_x_casadi = CASADI_INITIAL_STATE
    for i in range(sim_steps_actual):
        x_next_dm = dynamics_func(ca.DM(current_x_casadi), ca.DM(CONTROL_INPUT))
        current_x_casadi = x_next_dm.full().flatten()
        casadi_states[i + 1, :] = current_x_casadi
        if np.any(np.isnan(current_x_casadi)):
             print(f"CasADi simulation stopped at step {i+1} due to NaN state.")
             sim_steps_actual = i
             casadi_states = casadi_states[:sim_steps_actual+1, :]
             gym_states = gym_states[:sim_steps_actual+1, :]
             break
    print("CasADi simulation complete.")

    # --- Calculate Max Initial Swing Angle/Height (from CasADi sim) ---
    first_swing_steps = min(sim_steps_actual // 2, int(1.5 / DT)) # e.g., first 1.5s
    if sim_steps_actual > 0:
        max_theta_first_swing = np.max(np.abs(casadi_states[:first_swing_steps, 1]))
        max_height_first_swing = pole_vis_length * np.cos(max_theta_first_swing)
        print(f"Max CasADi angle in first ~{first_swing_steps*DT:.2f}s: {np.rad2deg(max_theta_first_swing):.2f} deg")
    else:
        print("Skipping max angle calculation due to zero simulation steps.")
        max_height_first_swing = pole_vis_length # Default if no sim

    # --- Setup Animation ---
    fig_anim, ax_anim = plt.subplots()
    ax_anim.set_aspect('equal')
    ax_anim.set_xlabel("Cart Position (m)")
    ax_anim.set_ylabel("Vertical Position (m)")
    ax_anim.set_title(f"CasADi ({args.integration}) Inverted Pendulum Simulation (No Control)")

    # Determine plot limits dynamically
    if sim_steps_actual > 0:
        max_cart_pos = np.max(np.abs(casadi_states[:, 0]))
    else:
        max_cart_pos = pole_vis_length # Default if no sim
    ax_anim.set_xlim(CASADI_INITIAL_STATE[0] - max_cart_pos - pole_vis_length * 1.2,
                CASADI_INITIAL_STATE[0] + max_cart_pos + pole_vis_length * 1.2)
    ax_anim.set_ylim(-pole_vis_length * 1.2, pole_vis_length * 1.2)

    # Add horizontal lines for max initial height
    ax_anim.axhline(max_height_first_swing, color='gray', linestyle='--', lw=1, label='Approx Max Height')
    ax_anim.axhline(-max_height_first_swing, color='gray', linestyle='--', lw=1) # Symmetric line
    ax_anim.legend(loc='upper right')

    # Create plot elements to update
    cart_width = 0.2
    cart_height = 0.1
    cart = plt.Rectangle((0, -cart_height/2), cart_width, cart_height, fc='blue')
    ax_anim.add_patch(cart)
    pole, = ax_anim.plot([], [], 'r-', lw=3) # Pole line
    pivot, = ax_anim.plot([], [], 'ko', ms=5) # Pivot point
    time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes, va='top')

    # --- Animation Update Function ---
    def update(frame):
        if frame >= casadi_states.shape[0]: # Prevent index error if sim stopped early
            return cart, pole, pivot, time_text
        x, theta = casadi_states[frame, 0], casadi_states[frame, 1]

        # Cart position
        cart_x = x - cart_width / 2
        cart.set_xy((cart_x, -cart_height / 2))

        # Pole position (pivot at cart center)
        pivot_x, pivot_y = x, 0
        pole_end_x = pivot_x + pole_vis_length * np.sin(theta)
        pole_end_y = pivot_y + pole_vis_length * np.cos(theta)
        pole.set_data([pivot_x, pole_end_x], [pivot_y, pole_end_y])
        pivot.set_data([pivot_x], [pivot_y])

        # Update time text
        time_text.set_text(f'Time: {frame * DT:.2f}s')

        return cart, pole, pivot, time_text

    # --- Run Animation ---
    if sim_steps_actual > 0:
        ani = animation.FuncAnimation(fig_anim, update, frames=sim_steps_actual + 1,
                                    interval=DT * 1000, blit=True, repeat=False)
        plt.grid(True)

        # --- Save Animation 1 ---
        if args.save:
            output_dir = "analysis"
            os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
            output_path_anim = os.path.join(output_dir, "casadi_no_control_animation.gif") # Changed extension
            print(f"Saving animation to {output_path_anim}...")
            try:
                ani.save(output_path_anim, writer='pillow', fps=int(1/DT)) # Use pillow writer
                print("Animation saving complete.")
            except Exception as e:
                print(f"Error saving animation: {e}")

        plt.show() # Show after saving (if sim ran)
        print("Visualization closed.")
    else:
        print("Skipping animation and plotting due to zero simulation steps.")
        plt.close(fig_anim) # Close the empty figure

    # --- Generate Static State Plots ---
    if sim_steps_actual > 0:
        print("Generating comparison state plots...")
        time_vector = np.arange(sim_steps_actual + 1) * DT
        fig_states, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
        fig_states.suptitle('State Variables vs Time: CasADi Model vs Gym Simulation (Unforced)')

        state_labels = ['Cart Position (x) [m]', 'Pole Angle (theta) [rad]',
                        'Cart Velocity (x_dot) [m/s]', 'Pole Angular Vel (theta_dot) [rad/s]']
        casadi_indices = [0, 1, 2, 3]

        for i, ax_i in enumerate(axs):
            idx = casadi_indices[i]
            ax_i.plot(time_vector, casadi_states[:, idx], label='CasADi Model')
            ax_i.plot(time_vector, gym_states[:, idx], linestyle='--', label='Gym Sim')
            ax_i.set_ylabel(state_labels[i])
            ax_i.grid(True)
            ax_i.legend()

        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        print("State plots closed.")
    # End of if args.sim_type in ['all', 'unforced']

# --- Setup CasADi function if not already done ---
# Needed if only running forced sim
if args.sim_type == 'forced' and 'dynamics_func' not in locals():
    x_sym = ca.SX.sym('x', 4)
    u_sym = ca.SX.sym('u', 1)
    x_next_sym = pendulum_dynamics(x_sym, u_sym, DT, params, integration_method=args.integration)
    dynamics_func = ca.Function('dynamics', [x_sym, u_sym], [x_next_sym])

# --- Simulation 2: Bang-Bang Control ---
if args.sim_type in ['all', 'forced']:
    print("\n--- Running Simulation 2: Bang-Bang Control ---")
    # Keep BB duration fixed for now, unless specified otherwise
    SIM_STEPS_BB = 50 # Example fixed duration for bang-bang
    n_cycles = 5
    half_cycle_duration = max(1, SIM_STEPS_BB // (2 * n_cycles))

    control_pattern = [-3.0] * half_cycle_duration + [3.0] * half_cycle_duration
    CONTROL_SEQUENCE = np.tile(control_pattern, n_cycles)

    if len(CONTROL_SEQUENCE) > SIM_STEPS_BB:
        CONTROL_SEQUENCE = CONTROL_SEQUENCE[:SIM_STEPS_BB]
    elif len(CONTROL_SEQUENCE) < SIM_STEPS_BB:
        fill_value = CONTROL_SEQUENCE[-1] if len(CONTROL_SEQUENCE) > 0 else 0.0
        CONTROL_SEQUENCE = np.append(CONTROL_SEQUENCE, [fill_value] * (SIM_STEPS_BB - len(CONTROL_SEQUENCE)))

    # --- Simulate Gymnasium with Bang-Bang ---
    print(f"Simulating Gym environment '{ENV_ID}' with Bang-Bang control...")
    env_bb = InvertedPendulumComparisonWrapper(gym.make(ENV_ID))
    obs_gym_bb, info_gym_bb = env_bb.reset(initial_state=CASADI_INITIAL_STATE)
    gym_states_bb = np.zeros((SIM_STEPS_BB + 1, len(obs_gym_bb)))
    gym_states_bb[0, :] = obs_gym_bb
    sim_steps_bb_actual = SIM_STEPS_BB # Track actual steps
    for i in range(SIM_STEPS_BB):
        action_gym = [CONTROL_SEQUENCE[i]]
        obs_gym_bb, reward_gym_bb, terminated_gym_bb, truncated_gym_bb, info_gym_bb = env_bb.step(action_gym)
        gym_states_bb[i + 1, :] = obs_gym_bb
        if terminated_gym_bb or truncated_gym_bb:
            print(f"Gym (Bang-Bang) simulation stopped at step {i+1}.")
            sim_steps_bb_actual = i
            gym_states_bb = gym_states_bb[:sim_steps_bb_actual+1, :]
            CONTROL_SEQUENCE = CONTROL_SEQUENCE[:sim_steps_bb_actual]
            break
    env_bb.close()
    print("Gym (Bang-Bang) simulation complete.")

    # --- Simulate CasADi with Bang-Bang ---
    print(f"Simulating CasADi model with Bang-Bang control...")
    casadi_states_bb = np.zeros((sim_steps_bb_actual + 1, 4))
    casadi_states_bb[0, :] = CASADI_INITIAL_STATE
    current_x_casadi_bb = CASADI_INITIAL_STATE
    for i in range(sim_steps_bb_actual):
        control_input_bb = np.array([CONTROL_SEQUENCE[i]])
        x_next_dm_bb = dynamics_func(ca.DM(current_x_casadi_bb), ca.DM(control_input_bb))
        current_x_casadi_bb = x_next_dm_bb.full().flatten()
        casadi_states_bb[i + 1, :] = current_x_casadi_bb
        if np.any(np.isnan(current_x_casadi_bb)):
             print(f"CasADi (Bang-Bang) simulation stopped at step {i+1} due to NaN state.")
             sim_steps_bb_actual = i
             casadi_states_bb = casadi_states_bb[:sim_steps_bb_actual+1, :]
             gym_states_bb = gym_states_bb[:sim_steps_bb_actual+1, :]
             CONTROL_SEQUENCE = CONTROL_SEQUENCE[:sim_steps_bb_actual]
             break
    print("CasADi (Bang-Bang) simulation complete.")

    # --- Animation for Bang-Bang Simulation ---
    if sim_steps_bb_actual > 0:
        print("Generating animation for Bang-Bang control...")
        fig_anim_bb, ax_anim_bb = plt.subplots()
        ax_anim_bb.set_aspect('equal')
        ax_anim_bb.set_xlabel("Cart Position (m)")
        ax_anim_bb.set_ylabel("Vertical Position (m)")
        ax_anim_bb.set_title(f"CasADi ({args.integration}) Simulation (Bang-Bang Control)")
        max_cart_pos_bb = np.max(np.abs(casadi_states_bb[:, 0]))
        ax_anim_bb.set_xlim(casadi_states_bb[0, 0] - max_cart_pos_bb - pole_vis_length * 1.2,
                       casadi_states_bb[0, 0] + max_cart_pos_bb + pole_vis_length * 1.2)
        ax_anim_bb.set_ylim(-pole_vis_length * 1.2, pole_vis_length * 1.2)
        cart_bb = plt.Rectangle((0, -cart_height/2), cart_width, cart_height, fc='blue')
        ax_anim_bb.add_patch(cart_bb)
        pole_bb, = ax_anim_bb.plot([], [], 'r-', lw=3)
        pivot_bb, = ax_anim_bb.plot([], [], 'ko', ms=5)
        time_text_bb = ax_anim_bb.text(0.05, 0.95, '', transform=ax_anim_bb.transAxes, va='top')

        def update_bb(frame):
            if frame >= casadi_states_bb.shape[0]: # Prevent index error
                return cart_bb, pole_bb, pivot_bb, time_text_bb
            x, theta = casadi_states_bb[frame, 0], casadi_states_bb[frame, 1]
            cart_x = x - cart_width / 2
            cart_bb.set_xy((cart_x, -cart_height / 2))
            pivot_x, pivot_y = x, 0
            pole_end_x = pivot_x + pole_vis_length * np.sin(theta)
            pole_end_y = pivot_y + pole_vis_length * np.cos(theta)
            pole_bb.set_data([pivot_x, pole_end_x], [pivot_y, pole_end_y])
            pivot_bb.set_data([pivot_x], [pivot_y])
            control_val_str = f"{CONTROL_SEQUENCE[frame]:.1f}" if frame < len(CONTROL_SEQUENCE) else "N/A"
            time_text_bb.set_text(f'Time: {frame * DT:.2f}s, u: {control_val_str}')
            return cart_bb, pole_bb, pivot_bb, time_text_bb

        ani_bb = animation.FuncAnimation(fig_anim_bb, update_bb, frames=sim_steps_bb_actual + 1,
                                       interval=DT * 1000, blit=True, repeat=False)
        plt.grid(True)

        # --- Save Animation 2 (Bang-Bang) ---
        if args.save:
            output_dir = "analysis"
            os.makedirs(output_dir, exist_ok=True)
            output_path_anim_bb = os.path.join(output_dir, "casadi_bang_bang_animation.gif") # Changed extension
            print(f"Saving Bang-Bang animation to {output_path_anim_bb}...")
            try:
                ani_bb.save(output_path_anim_bb, writer='pillow', fps=int(1/DT)) # Use pillow writer
                print("Bang-Bang animation saving complete.")
            except Exception as e:
                print(f"Error saving Bang-Bang animation: {e}")

        plt.show() # Show after saving
        print("Bang-Bang Animation closed.")
    else:
        print("Skipping Bang-Bang animation and plotting due to zero simulation steps.")

    # --- Static Plots for Bang-Bang Simulation ---
    if sim_steps_bb_actual > 0:
        print("Generating comparison state plots for Bang-Bang control...")
        time_vector_bb = np.arange(sim_steps_bb_actual + 1) * DT
        fig_states_bb, axs_bb = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
        fig_states_bb.suptitle('State & Control vs Time: CasADi Model vs Gym Sim (Bang-Bang)')

        # Define state labels if not already defined (e.g., if only forced sim ran)
        if 'state_labels' not in locals():
             state_labels = ['Cart Position (x) [m]', 'Pole Angle (theta) [rad]',
                        'Cart Velocity (x_dot) [m/s]', 'Pole Angular Vel (theta_dot) [rad/s]']
             casadi_indices = [0, 1, 2, 3]

        for i, ax_i in enumerate(axs_bb[:4]): # Plot first 4 states
            idx = casadi_indices[i]
            ax_i.plot(time_vector_bb, casadi_states_bb[:, idx], label='CasADi Model')
            ax_i.plot(time_vector_bb, gym_states_bb[:, idx], linestyle='--', label='Gym Sim')
            ax_i.set_ylabel(state_labels[i])
            ax_i.grid(True)
            ax_i.legend()

        # Plot control input
        # Ensure CONTROL_SEQUENCE aligns with time_vector_bb[:-1]
        control_plot_len = min(len(CONTROL_SEQUENCE), len(time_vector_bb) - 1)
        axs_bb[4].plot(time_vector_bb[:control_plot_len], CONTROL_SEQUENCE[:control_plot_len], label='Applied Control (u)')
        axs_bb[4].set_ylabel('Control Input [N]')
        axs_bb[4].grid(True)
        axs_bb[4].legend()

        axs_bb[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        print("Bang-Bang State plots closed.")
    # End of if args.sim_type in ['all', 'forced']

print("\nScript finished.") 