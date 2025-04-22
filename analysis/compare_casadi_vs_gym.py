# analysis/compare_casadi_vs_gym.py
import sys
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
from src.utils.parameters import load_inverted_pendulum_params
from src.environments.wrappers import InvertedPendulumComparisonWrapper
from src.environments.casadi_dynamics import pendulum_dynamics # Import the dynamics function


print("Running CasADi and Gym dynamics simulation and comparison...")

# --- Configuration ---
ENV_ID = "InvertedPendulum-v5"
# Use paths relative to project root
PARAM_PATH = "src/environments/inverted_pendulum_params.json" 
SIM_STEPS = 2500  # Reduced steps for faster visualization (e.g., 100 seconds if dt=0.04)
CASADI_INITIAL_STATE = np.array([0.0, 0.1, 0.0, 0.0]) 
CONTROL_INPUT = np.array([0.0]) 
INTEGRATION_METHOD = 'rk4'

# --- Load Parameters for CasADi ---
if not os.path.exists(PARAM_PATH):
    print(f"Error: Parameter file not found at {PARAM_PATH}")
    exit()
params = load_inverted_pendulum_params(PARAM_PATH)
pole_vis_length = params.get('pole_length', 0.6) # Use full length for vis

# --- Simulate Gymnasium Environment ---
print(f"Simulating Gym environment '{ENV_ID}' for {SIM_STEPS} steps using wrapper...")
env = InvertedPendulumComparisonWrapper(gym.make(ENV_ID))

# --- Get the ACTUAL timestep from the environment ---
DT = env.unwrapped.dt # Access dt from the unwrapped environment
print(f"Using environment timestep DT = {DT}")

obs_gym, info_gym = env.reset(initial_state=CASADI_INITIAL_STATE)
print(f"Gym reset state: {obs_gym}")
gym_states = np.zeros((SIM_STEPS + 1, len(obs_gym))) # Store raw observations
gym_states[0, :] = obs_gym
for i in range(SIM_STEPS):
    action_gym = [CONTROL_INPUT[0]] 
    obs_gym, reward_gym, terminated_gym, truncated_gym, info_gym = env.step(action_gym)
    gym_states[i + 1, :] = obs_gym
    if truncated_gym: 
        print(f"Gym simulation truncated at step {i+1}.")
        SIM_STEPS = i 
        gym_states = gym_states[:SIM_STEPS+1, :]
        break
env.close()
print("Gym simulation complete.")

# --- Setup CasADi Function (Ensure DT is passed correctly) ---
x_sym = ca.SX.sym('x', 4)
u_sym = ca.SX.sym('u', 1)
x_next_sym = pendulum_dynamics(x_sym, u_sym, DT, params, integration_method=INTEGRATION_METHOD)
dynamics_func = ca.Function('dynamics', [x_sym, u_sym], [x_next_sym])

# --- Simulate CasADi Dynamics (uses the correct DT now) ---
print(f"Simulating CasADi model for {SIM_STEPS} steps...")
casadi_states = np.zeros((SIM_STEPS + 1, 4)) 
casadi_states[0, :] = CASADI_INITIAL_STATE 
current_x_casadi = CASADI_INITIAL_STATE
for i in range(SIM_STEPS): # Use potentially updated SIM_STEPS
    x_next_dm = dynamics_func(ca.DM(current_x_casadi), ca.DM(CONTROL_INPUT))
    current_x_casadi = x_next_dm.full().flatten()
    casadi_states[i + 1, :] = current_x_casadi
    if np.any(np.isnan(current_x_casadi)):
         print(f"CasADi simulation stopped at step {i+1} due to NaN state.")
         SIM_STEPS = i 
         casadi_states = casadi_states[:SIM_STEPS+1, :]
         gym_states = gym_states[:SIM_STEPS+1, :] 
         break      
print("CasADi simulation complete.")

# --- Calculate Max Initial Swing Angle/Height (from CasADi sim) ---
first_swing_steps = min(SIM_STEPS // 2, 150) 
max_theta_first_swing = np.max(np.abs(casadi_states[:first_swing_steps, 1]))
max_height_first_swing = pole_vis_length * np.cos(max_theta_first_swing) 
print(f"Max CasADi angle in first ~{first_swing_steps*DT:.2f}s: {np.rad2deg(max_theta_first_swing):.2f} deg")

# --- Setup Animation (Uses DT for interval) ---
fig_anim, ax_anim = plt.subplots()
ax_anim.set_aspect('equal')
ax_anim.set_xlabel("Cart Position (m)")
ax_anim.set_ylabel("Vertical Position (m)")
ax_anim.set_title(f"CasADi ({INTEGRATION_METHOD}) Inverted Pendulum Simulation (No Control)")

# Determine plot limits dynamically
max_cart_pos = np.max(np.abs(casadi_states[:, 0]))
ax_anim.set_xlim(casadi_states[0, 0] - max_cart_pos - pole_vis_length * 1.2, 
            casadi_states[0, 0] + max_cart_pos + pole_vis_length * 1.2)
ax_anim.set_ylim(-pole_vis_length * 1.2, pole_vis_length * 1.2)

# Add horizontal lines for max initial height
ax_anim.axhline(max_height_first_swing, color='gray', linestyle='--', lw=1, label='Initial Max Height')
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
ani = animation.FuncAnimation(fig_anim, update, frames=SIM_STEPS + 1, 
                            interval=DT * 1000, blit=True, repeat=False)
plt.grid(True)
plt.show()

print("Visualization closed.")

# --- Generate Static State Plots (Uses correct DT for time_vector) ---
print("Generating comparison state plots...")
time_vector = np.arange(SIM_STEPS + 1) * DT # Uses correct DT
fig_states, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
fig_states.suptitle('State Variables vs Time: CasADi Model vs Gym Simulation')

state_labels = ['Cart Position (x) [m]', 'Pole Angle (theta) [rad]', 
                'Cart Velocity (x_dot) [m/s]', 'Pole Angular Vel (theta_dot) [rad/s]']
casadi_indices = [0, 1, 2, 3]

for i, ax_i in enumerate(axs):
    idx = casadi_indices[i]
    ax_i.plot(time_vector, casadi_states[:, idx], label='CasADi Model')
    # Directly plot the corresponding column from gym_states
    ax_i.plot(time_vector, gym_states[:, idx], linestyle='--', label='Gym Sim') 
    ax_i.set_ylabel(state_labels[i])
    ax_i.grid(True)
    ax_i.legend()

axs[-1].set_xlabel("Time (s)")
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()

print("State plots closed.") 


# --- Simulation 2: Bang-Bang Control --- 
print("\n--- Starting Simulation 2: Bang-Bang Control ---")
# SIM_STEPS_BB = 40 # Shorter duration for controlled test
# CONTROL_SEQUENCE = np.array(
#     [3.0] * (SIM_STEPS_BB // 2) + 
#     [-3.0] * (SIM_STEPS_BB - SIM_STEPS_BB // 2)
# )
SIM_STEPS_BB = 20 
n_cycles = 5
half_cycle_duration = max(1, SIM_STEPS_BB // (2 * n_cycles)) # e.g., 100 // 20 = 5

control_pattern = [-3.0] * half_cycle_duration + [3.0] * half_cycle_duration
CONTROL_SEQUENCE = np.tile(control_pattern, n_cycles)

# Adjust length if SIM_STEPS_BB wasn't perfectly divisible
if len(CONTROL_SEQUENCE) > SIM_STEPS_BB:
    CONTROL_SEQUENCE = CONTROL_SEQUENCE[:SIM_STEPS_BB]
elif len(CONTROL_SEQUENCE) < SIM_STEPS_BB:
    # Append the last value to fill up
    fill_value = CONTROL_SEQUENCE[-1] if len(CONTROL_SEQUENCE) > 0 else 0.0
    CONTROL_SEQUENCE = np.append(CONTROL_SEQUENCE, [fill_value] * (SIM_STEPS_BB - len(CONTROL_SEQUENCE)))


# --- Simulate Gymnasium with Bang-Bang --- 
print(f"Simulating Gym environment '{ENV_ID}' with Bang-Bang control...")
env_bb = InvertedPendulumComparisonWrapper(gym.make(ENV_ID))
obs_gym_bb, info_gym_bb = env_bb.reset(initial_state=CASADI_INITIAL_STATE)
gym_states_bb = np.zeros((SIM_STEPS_BB + 1, len(obs_gym_bb)))
gym_states_bb[0, :] = obs_gym_bb
for i in range(SIM_STEPS_BB):
    action_gym = [CONTROL_SEQUENCE[i]] 
    obs_gym_bb, reward_gym_bb, terminated_gym_bb, truncated_gym_bb, info_gym_bb = env_bb.step(action_gym)
    gym_states_bb[i + 1, :] = obs_gym_bb
    if terminated_gym_bb or truncated_gym_bb: 
        print(f"Gym (Bang-Bang) simulation stopped at step {i+1}.")
        SIM_STEPS_BB = i 
        gym_states_bb = gym_states_bb[:SIM_STEPS_BB+1, :]
        CONTROL_SEQUENCE = CONTROL_SEQUENCE[:SIM_STEPS_BB] # Truncate control too
        break
env_bb.close()
print("Gym (Bang-Bang) simulation complete.")

# --- Simulate CasADi with Bang-Bang --- 
print(f"Simulating CasADi model with Bang-Bang control...")
casadi_states_bb = np.zeros((SIM_STEPS_BB + 1, 4))
casadi_states_bb[0, :] = CASADI_INITIAL_STATE
current_x_casadi_bb = CASADI_INITIAL_STATE
for i in range(SIM_STEPS_BB): # Use potentially updated SIM_STEPS_BB
    control_input_bb = np.array([CONTROL_SEQUENCE[i]])
    x_next_dm_bb = dynamics_func(ca.DM(current_x_casadi_bb), ca.DM(control_input_bb))
    current_x_casadi_bb = x_next_dm_bb.full().flatten()
    casadi_states_bb[i + 1, :] = current_x_casadi_bb
    if np.any(np.isnan(current_x_casadi_bb)):
         print(f"CasADi (Bang-Bang) simulation stopped at step {i+1} due to NaN state.")
         SIM_STEPS_BB = i 
         casadi_states_bb = casadi_states_bb[:SIM_STEPS_BB+1, :]
         gym_states_bb = gym_states_bb[:SIM_STEPS_BB+1, :] 
         CONTROL_SEQUENCE = CONTROL_SEQUENCE[:SIM_STEPS_BB]
         break      
print("CasADi (Bang-Bang) simulation complete.")

# --- Animation for Bang-Bang Simulation --- 
print("Generating animation for Bang-Bang control...")
fig_anim_bb, ax_anim_bb = plt.subplots()
ax_anim_bb.set_aspect('equal')
ax_anim_bb.set_xlabel("Cart Position (m)")
ax_anim_bb.set_ylabel("Vertical Position (m)")
ax_anim_bb.set_title(f"CasADi ({INTEGRATION_METHOD}) Simulation (Bang-Bang Control)")
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
    x, theta = casadi_states_bb[frame, 0], casadi_states_bb[frame, 1]
    cart_x = x - cart_width / 2
    cart_bb.set_xy((cart_x, -cart_height / 2))
    pivot_x, pivot_y = x, 0
    pole_end_x = pivot_x + pole_vis_length * np.sin(theta)
    pole_end_y = pivot_y + pole_vis_length * np.cos(theta)
    pole_bb.set_data([pivot_x, pole_end_x], [pivot_y, pole_end_y])
    pivot_bb.set_data([pivot_x], [pivot_y])
    time_text_bb.set_text(f'Time: {frame * DT:.2f}s, u: {CONTROL_SEQUENCE[frame] if frame < len(CONTROL_SEQUENCE) else "N/A"}')
    return cart_bb, pole_bb, pivot_bb, time_text_bb

ani_bb = animation.FuncAnimation(fig_anim_bb, update_bb, frames=SIM_STEPS_BB + 1, 
                               interval=DT * 1000, blit=True, repeat=False)
plt.grid(True)
plt.show()
print("Bang-Bang Animation closed.")

# --- Static Plots for Bang-Bang Simulation --- 
print("Generating comparison state plots for Bang-Bang control...")
time_vector_bb = np.arange(SIM_STEPS_BB + 1) * DT 
fig_states_bb, axs_bb = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
fig_states_bb.suptitle('State & Control vs Time: CasADi Model vs Gym Sim (Bang-Bang)')

for i, ax_i in enumerate(axs_bb[:4]): # Plot first 4 states
    idx = casadi_indices[i]
    ax_i.plot(time_vector_bb, casadi_states_bb[:, idx], label='CasADi Model')
    ax_i.plot(time_vector_bb, gym_states_bb[:, idx], linestyle='--', label='Gym Sim') 
    ax_i.set_ylabel(state_labels[i])
    ax_i.grid(True)
    ax_i.legend()

# Plot control input
axs_bb[4].plot(time_vector_bb[:-1], CONTROL_SEQUENCE, label='Applied Control (u)') # Control applies over interval
axs_bb[4].set_ylabel('Control Input [N]')
axs_bb[4].grid(True)
axs_bb[4].legend()

axs_bb[-1].set_xlabel("Time (s)")
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()

print("Bang-Bang State plots closed.") 