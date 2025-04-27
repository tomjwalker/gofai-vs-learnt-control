# analysis/compare_casadi_vs_gym_double.py
"""
Compares the dynamics of the generated CasADi model for the Inverted Double Pendulum
against the Gymnasium environment simulation.

Provides three modes via command-line arguments:
1.  `--mode free` (default): 
    Simulates both CasADi and Gym with zero control input, starting from a 
    slightly offset initial condition. Plots states and shows animation.
2.  `--mode control`:
    Simulates both CasADi and Gym with a predefined bang-bang control sequence.
    Plots states and shows animation.
3.  `--mode multi_angle`:
    Simulates *only* the CasADi model three times with slightly different 
    initial angles (zero control). 
    Plots states and shows animation with all three traces.

Usage:
------
python -m analysis.compare_casadi_vs_gym_double [--mode <mode>] [--steps <N>] [--no-anim] [--no-plot]
"""
import sys
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium as gym
import casadi as ca
import argparse

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary functions/classes for DOUBLE pendulum
from src.utils.parameters import load_double_pendulum_params # Updated
from src.environments.wrappers import InvertedDoublePendulumComparisonWrapper # Updated
from src.environments.double_pendulum_dynamics import get_dynamics_function # Updated
from src.utils.integration import rk4_step # Import the new helper


print("Running DOUBLE Pendulum CasADi and Gym dynamics simulation and comparison...")

# --- Argument Parsing --- 
parser = argparse.ArgumentParser(description="Compare CasADi vs Gym for Double Pendulum.")
parser.add_argument(
    '--mode', type=str, default='free', 
    choices=['free', 'control', 'multi_angle'],
    help="Simulation mode: 'free' (no control comparison), 'control' (bang-bang comparison), 'multi_angle' (CasADi only, multiple initial angles)."
)
parser.add_argument(
    '--steps', type=int, default=200, 
    help="Number of simulation steps."
)
parser.add_argument(
    '--no-anim', action='store_true', 
    help="Skip showing the animation."
)
parser.add_argument(
    '--no-plot', action='store_true', 
    help="Skip showing the static state plots."
)

args = parser.parse_args()
SIM_MODE = args.mode
SIM_STEPS = args.steps
SHOW_ANIM = not args.no_anim
SHOW_PLOT = not args.no_plot

# --- Configuration (Defaults) ---
ENV_ID = "InvertedDoublePendulum-v5"
PARAM_PATH = "src/environments/double_pendulum_params.json"
# Initial state [x, th1, th2, xd, th1d, th2d]
INITIAL_STATE_DEFAULT = np.array([0.0, 0.1, -0.1, 0.0, 0.0, 0.0]) 
CONTROL_INPUT_FREE = np.array([0.0]) # For free simulation
INTEGRATION_METHOD = 'rk4' # CasADi integration method

# --- Load Parameters for CasADi --- 
if not os.path.exists(PARAM_PATH):
    print(f"Error: Parameter file not found at {PARAM_PATH}")
    sys.exit()
# Load the ordered list of params, dt, and joint limits
params_list, dt_loaded, joint_limits = load_double_pendulum_params(PARAM_PATH, return_mapping=False) 
if params_list is None or dt_loaded is None or joint_limits is None: 
    print("Error: Failed to load parameters, dt, or joint limits. Exiting.")
    sys.exit()
DT = dt_loaded
print(f"Using timestep DT = {DT}")
print(f"Using joint limits for clamping: {joint_limits}")

# Define indices based on the known order for extracting lengths
PARAM_ORDER = [
    'M', 'm1', 'm2', 'l1', 'l2', 'd1', 'd2', 'Icm1', 'Icm2', 'g',
    'b_slide', 'b_fric', 'b_joint1', 'b_joint2', 'gear'
]
try:
    idx_l1 = PARAM_ORDER.index('l1')
    idx_l2 = PARAM_ORDER.index('l2')
    vis_pole1_length = params_list[idx_l1]
    vis_pole2_length = params_list[idx_l2]
except (ValueError, IndexError):
    print("Warning: Could not find 'l1' or 'l2' in loaded parameter list. Using defaults for visualization.")
    vis_pole1_length = 0.6 
    vis_pole2_length = 0.6

# Extract slider limits for clamping
x_limits = joint_limits.get('x', [-np.inf, np.inf])
if len(x_limits) != 2: x_limits = [-np.inf, np.inf]
else: print(f"Applying x position clamping between {x_limits[0]} and {x_limits[1]}")

# --- Setup CasADi Dynamics Function --- 
# Load the function f(x, u, p) -> x_dot
casadi_ode_func = get_dynamics_function() 
print("CasADi dynamics ODE function loaded.")
param_values_casadi = ca.DM(params_list)
# Verify parameter vector size
expected_param_size = casadi_ode_func.size_in(2)[0]
if param_values_casadi.shape[0] != expected_param_size:
     print(f"Error: Parameter size mismatch."); sys.exit()

# --- Simulation Functions --- 
def simulate_casadi(initial_state, control_sequence):
    print(f"Simulating CasADi model (using shared RK4 integration, DT={DT})...") # Updated print
    
    # --- DEBUG: Verify parameter values being used --- 
    try:
        b_slide_index = 10 # Based on param_order in parameters.py
        b_fric_index = 11
        loaded_b_slide = param_values_casadi[b_slide_index].full().item()
        loaded_b_fric = param_values_casadi[b_fric_index].full().item()
        print(f"[DEBUG] Using b_slide (param {b_slide_index}): {loaded_b_slide:.4f}")
        print(f"[DEBUG] Using b_fric (param {b_fric_index}): {loaded_b_fric:.4f}")
    except Exception as e:
        print(f"[DEBUG] Error accessing debug params: {e}")
    # --- END DEBUG --- 
    
    n_steps = len(control_sequence)
    casadi_states = np.zeros((n_steps + 1, 6))
    casadi_states[0, :] = initial_state
    current_x_dm = ca.DM(initial_state)
    
    for i in range(n_steps):
        control_input_dm = ca.DM([control_sequence[i]]) 
        try:
            # Use the shared RK4 step function
            current_x_dm = rk4_step(casadi_ode_func, current_x_dm, control_input_dm, DT, param_values_casadi)
            
            current_x = current_x_dm.full().flatten() # Convert back to numpy for storage/clamping

            # --- Clamp the slider position (x) --- 
            current_x[0] = np.clip(current_x[0], x_limits[0], x_limits[1])
            # Optional velocity reset (commented out)
            # if current_x[0] == x_limits[0] or current_x[0] == x_limits[1]:
            #    current_x[3] = 0.0 
            
            if np.any(np.isnan(current_x)):
                print(f"CasADi NaN detected after RK4/clamping at step {i+1}. Stopping.")
                casadi_states[i + 1, :] = np.nan
                return casadi_states[:i+2, :]
            
            casadi_states[i + 1, :] = current_x
            current_x_dm = ca.DM(current_x) # Update DM for next iteration
            
        except Exception as e:
            print(f"Error during CasADi Manual RK4 simulation step {i+1}: {e}")
            casadi_states[i + 1:, :] = np.nan
            return casadi_states[:i+2, :]
            
    print("CasADi simulation complete.")
    return casadi_states

def simulate_gym(initial_state, control_sequence):
    print(f"Simulating Gym environment '{ENV_ID}'...")
    try:
        env = InvertedDoublePendulumComparisonWrapper(gym.make(ENV_ID))
        obs_gym, info_gym = env.reset(initial_state=initial_state)
        print(f"Gym reset state: {obs_gym}")
        n_steps = len(control_sequence)
        gym_states = np.zeros((n_steps + 1, 6)) 
        gym_states[0, :] = obs_gym
        for i in range(n_steps):
            action_gym = [control_sequence[i]] 
            obs_gym, reward_gym, terminated_gym, truncated_gym, info_gym = env.step(action_gym)
            gym_states[i + 1, :] = obs_gym
            # Note: termination is disabled by wrapper, only truncation applies
            if truncated_gym: 
                print(f"Gym simulation truncated at step {i+1}.")
                return gym_states[:i+1, :]
        env.close()
        print("Gym simulation complete.")
        return gym_states
    except Exception as e:
         print(f"Error during Gym simulation: {e}")
         return None # Indicate failure

# --- Plotting Functions --- 
def plot_states(time_vector, states_dict, title):
    n_states = states_dict[list(states_dict.keys())[0]].shape[1]
    assert n_states == 6, "Expected 6 states for double pendulum"
    fig, axs = plt.subplots(n_states, 1, sharex=True, figsize=(10, 12))
    fig.suptitle(title)
    state_labels = ['x (m)', 'th1 (rad)', 'th2 (rad)', 'xd (m/s)', 'th1d (rad/s)', 'th2d (rad/s)']
    line_styles = [':', '--', '-.'] # For multi-angle
    
    for i, label in enumerate(state_labels):
        for j, (name, states) in enumerate(states_dict.items()):
            if states.shape[0] == len(time_vector): # Check lengths match
                 style = line_styles[j % len(line_styles)] if SIM_MODE == 'multi_angle' else ('-' if 'CasADi' in name else '--')
                 axs[i].plot(time_vector, states[:, i], linestyle=style, label=name)
            else:
                 print(f"Warning: Length mismatch for {name}. States: {states.shape[0]}, Time: {len(time_vector)}. Skipping plot.")
        axs[i].set_ylabel(label)
        axs[i].grid(True)
        axs[i].legend()
    
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def setup_animation(title):
    fig_anim, ax_anim = plt.subplots()
    ax_anim.set_aspect('equal')
    ax_anim.set_xlabel("Cart Position (m)")
    ax_anim.set_ylabel("Vertical Position (m)")
    ax_anim.set_title(title.replace('(RK4 Integration)', '(Manual RK4)'))
    cart_width = 0.2
    cart_height = 0.1
    return fig_anim, ax_anim, cart_width, cart_height

def update_animation(frame, states_list, lines_poles1, lines_poles2, points_pivot1, points_pivot2, carts, time_text, dt):
    time_text.set_text(f'Time: {frame * dt:.2f}s')
    artists = [time_text]
    for i, states in enumerate(states_list):
        if frame < states.shape[0]: # Ensure frame is within bounds for this state trajectory
            x, th1, th2 = states[frame, 0], states[frame, 1], states[frame, 2]
            cart_x = x - carts[i].get_width() / 2
            carts[i].set_xy((cart_x, -carts[i].get_height() / 2))
            
            p1_x, p1_y = x, 0 # Pivot 1 (cart top)
            p2_x = p1_x + vis_pole1_length * np.sin(th1)
            p2_y = p1_y + vis_pole1_length * np.cos(th1)
            p3_x = p2_x + vis_pole2_length * np.sin(th2)
            p3_y = p2_y + vis_pole2_length * np.cos(th2)
            
            lines_poles1[i].set_data([p1_x, p2_x], [p1_y, p2_y])
            lines_poles2[i].set_data([p2_x, p3_x], [p2_y, p3_y])
            points_pivot1[i].set_data([p1_x], [p1_y])
            points_pivot2[i].set_data([p2_x], [p2_y])
            artists.extend([carts[i], lines_poles1[i], lines_poles2[i], points_pivot1[i], points_pivot2[i]])
        else:
            # If frame exceeds this state length (e.g., Gym truncated early), keep previous elements visible
             artists.extend([carts[i], lines_poles1[i], lines_poles2[i], points_pivot1[i], points_pivot2[i]])
    return artists


# --- Main Simulation Logic --- 
casadi_states = None
gym_states = None
control_sequence = None
multi_casadi_states = {}

if SIM_MODE == 'free' or SIM_MODE == 'control':
    if SIM_MODE == 'free':
        print("\n--- Running Mode: FREE ---")
        control_sequence = np.repeat(CONTROL_INPUT_FREE, SIM_STEPS)
    elif SIM_MODE == 'control':
        print("\n--- Running Mode: CONTROL (Bang-Bang) ---")
        # Load the full mapping dict *only* to get control limits here
        _, _, _, params_map = load_double_pendulum_params(PARAM_PATH, return_mapping=True)
        if params_map is None:
            print("Error loading param map for control limits. Exiting.")
            sys.exit()
        try:
            control_limits = params_map.get('control_limits_gym', [-1.0, 1.0])
            max_control = float(control_limits[1])
            print(f"Using max control limit from params: {max_control:.2f}")
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"Warning: Could not read 'control_limits_gym' from params mapping: {e}. Using default 1.0")
            max_control = 1.0
            
        control_sequence = np.array(
             [-1.0] * (SIM_STEPS // 2) + 
             [1.0] * (SIM_STEPS - SIM_STEPS // 2)
         ) * max_control

    # Simulate both
    casadi_states = simulate_casadi(INITIAL_STATE_DEFAULT, control_sequence)
    gym_states = simulate_gym(INITIAL_STATE_DEFAULT, control_sequence)

    # Adjust lengths if one simulation stopped early
    if gym_states is not None:
        min_len = min(casadi_states.shape[0], gym_states.shape[0])
        casadi_states = casadi_states[:min_len, :]
        gym_states = gym_states[:min_len, :]
        sim_steps_actual = min_len - 1
        time_vector = np.arange(min_len) * DT
    else:
        sim_steps_actual = casadi_states.shape[0] - 1
        time_vector = np.arange(casadi_states.shape[0]) * DT
        
elif SIM_MODE == 'multi_angle':
    print("\n--- Running Mode: MULTI_ANGLE (CasADi only) ---")
    control_sequence = np.repeat(CONTROL_INPUT_FREE, SIM_STEPS) # Use zero control
    initial_states = [
        INITIAL_STATE_DEFAULT,
        np.array([0.0, 0.15, -0.15, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.05, -0.05, 0.0, 0.0, 0.0])
    ]
    sim_steps_actual = SIM_STEPS
    max_len = 0
    for i, init_state in enumerate(initial_states):
        states = simulate_casadi(init_state, control_sequence)
        multi_casadi_states[f'CasADi_Init{i+1}'] = states
        max_len = max(max_len, states.shape[0])
    # Pad shorter simulations if needed for plotting/animation
    time_vector = np.arange(max_len) * DT
    for name, states in multi_casadi_states.items():
        if states.shape[0] < max_len:
            padding = np.full((max_len - states.shape[0], 6), np.nan)
            multi_casadi_states[name] = np.vstack((states, padding))
            print(f"Padded {name} trajectory to length {max_len}")

# --- Visualization --- 
if SHOW_ANIM:
    states_to_animate = []
    labels = [] # Store labels for the legend
    if SIM_MODE == 'multi_angle':
        title = f"CasADi Double Pendulum Simulation (RK4, Multiple Init Angles, dt={DT:.3f})" # Updated title
        states_to_animate = list(multi_casadi_states.values())
        labels = list(multi_casadi_states.keys())
    elif casadi_states is not None:
        title = f"CasADi (RK4) Double Pendulum Simulation (Mode: {SIM_MODE}, dt={DT:.3f})" # Updated title
        states_to_animate = [casadi_states]
        labels = ["CasADi Model (RK4)"] # Updated label
        if gym_states is not None:
             title = f"CasADi (RK4) vs Gym Double Pendulum (Mode: {SIM_MODE}, dt={DT:.3f})" # Updated title
             states_to_animate.append(gym_states)
             labels.append("Gym Sim")
    else:
        print("No valid states to animate.")

    if states_to_animate:
        fig_anim, ax_anim, cart_w, cart_h = setup_animation(title)
        num_trajectories = len(states_to_animate)
        colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))
        lines_p1, lines_p2, pivots1, pivots2, carts = [], [], [], [], []
        
        # Determine plot limits based on all trajectories
        all_x = np.concatenate([s[:, 0] for s in states_to_animate if s is not None])
        max_abs_x = np.max(np.abs(all_x[~np.isnan(all_x)])) if len(all_x[~np.isnan(all_x)]) > 0 else 1.0
        ax_anim.set_xlim(-max_abs_x - vis_pole1_length - vis_pole2_length, max_abs_x + vis_pole1_length + vis_pole2_length)
        ax_anim.set_ylim(-vis_pole1_length - vis_pole2_length - 0.2, vis_pole1_length + vis_pole2_length + 0.2)

        for i in range(num_trajectories):
            cart = plt.Rectangle((0, -cart_h/2), cart_w, cart_h, fc=colors[i], alpha=0.6 if num_trajectories > 1 else 1.0, label=labels[i])
            ax_anim.add_patch(cart)
            carts.append(cart)
            line_p1, = ax_anim.plot([], [], '-', lw=3, color=colors[i], alpha=0.8, label=f"{labels[i]} Pole 1")
            lines_p1.append(line_p1)
            line_p2, = ax_anim.plot([], [], '-', lw=3, color=colors[i], alpha=0.8, label=f"{labels[i]} Pole 2")
            lines_p2.append(line_p2)
            piv1, = ax_anim.plot([], [], 'o', ms=5, color=colors[i])
            pivots1.append(piv1)
            piv2, = ax_anim.plot([], [], 'o', ms=5, color=colors[i])
            pivots2.append(piv2)
            
        time_text_anim = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, va='top')
        max_frames = max(s.shape[0] for s in states_to_animate) if states_to_animate else 0
        
        # Add legend - using labels from carts should be sufficient
        legend_handles = [c for c in carts] # Use cart patches for legend
        ax_anim.legend(handles=legend_handles, loc='upper right') 

        ani = animation.FuncAnimation(
            fig_anim, update_animation, 
            frames=max_frames, 
            fargs=(states_to_animate, lines_p1, lines_p2, pivots1, pivots2, carts, time_text_anim, DT),
            interval=DT * 1000, 
            blit=True, repeat=False
        )
        plt.grid(True)
        plt.show()
        print("Animation closed.")

# --- Static State Plots --- 
if SHOW_PLOT:
    if SIM_MODE == 'multi_angle':
        plot_states(time_vector, multi_casadi_states, f"CasADi States (RK4, Multi-Angle, dt={DT:.3f})") # Updated title
    elif casadi_states is not None:
        states_to_plot = {"CasADi Model (RK4)": casadi_states} # Updated label
        plot_title = f"CasADi States (RK4, Mode: {SIM_MODE}, dt={DT:.3f})" # Updated title
        if gym_states is not None:
            states_to_plot["Gym Sim"] = gym_states
            plot_title = f"CasADi (RK4) vs Gym States (Mode: {SIM_MODE}, dt={DT:.3f})" # Updated title
        plot_states(time_vector, states_to_plot, plot_title)
    else:
         print("No valid states to plot.")

print("\nComparison script finished.") 