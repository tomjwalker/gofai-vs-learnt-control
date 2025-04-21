# casadi_dynamics.py
import casadi as ca
from typing import Dict, Any


def _euler_integration(x: ca.SX, x_dot: ca.SX, x_ddot: ca.SX, dt: float) -> ca.SX:
    """
    Perform one step of Euler integration.
    
    Args:
        x: Current state
        x_dot: Current velocity
        x_ddot: Current acceleration
        dt: Time step
        
    Returns:
        Next state after Euler integration
    """
    next_pos = x[0] + x_dot[0] * dt
    next_theta = x[1] + x_dot[1] * dt
    next_vel = x_dot[0] + x_ddot[0] * dt
    next_theta_dot = x_dot[1] + x_ddot[1] * dt
    
    return ca.vertcat(next_pos, next_theta, next_vel, next_theta_dot)

def _calculate_state_derivatives(x: ca.SX, u: ca.SX, params: Dict[str, Any]) -> ca.SX:
    """
    Calculates the state derivatives [x_dot, theta_dot, x_ddot, theta_ddot]
    for the inverted pendulum, including all forces (gravity, coriolis, 
    damping, friction, control).
    """
    # Unpack state
    pos, theta, x_dot, theta_dot = x[0], x[1], x[2], x[3]
    
    # Unpack parameters (same as in pendulum_dynamics)
    gravity = params.get('gravity', 9.81)
    pole_half_length = params.get('pole_half_length', 0.25) 
    pole_mass = params.get('pole_mass', 0.1)
    cart_mass = params.get('cart_mass', 1.0)
    actuator_gear_dict = params.get('actuator_gear', {'slide': 1.0})
    actuator_gear = actuator_gear_dict.get('slide', 1.0)
    joint_damping = params.get('joint_damping', 1.0) # Hinge damping
    slider_damping = params.get('slider_damping', 0.0) # Slider damping
    cart_friction = params.get('cart_friction', 0.1) # Cart friction
    pole_inertia_com_y = params.get('pole_inertia_about_y', 0.001) # Inertia about COM

    # Moment of inertia about pivot
    I_pivot = pole_inertia_com_y + pole_mass * pole_half_length ** 2 

    # Trig functions
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)

    # Inertia matrix M(q)
    inertia_matrix = ca.vertcat(
        ca.horzcat(cart_mass + pole_mass, pole_mass * pole_half_length * cos_theta),
        ca.horzcat(pole_mass * pole_half_length * cos_theta, I_pivot)
    )

    # Force vector C(q, q_dot) + G(q) + B*u + F_damp/fric
    coriolis_gravity_cart = pole_mass * pole_half_length * theta_dot**2 * sin_theta
    coriolis_gravity_pole = pole_mass * gravity * pole_half_length * sin_theta
    control_force = actuator_gear * u[0]
    damping_force_cart = -slider_damping * x_dot 
    friction_force_cart = -cart_friction * x_dot 
    damping_torque_pole = -joint_damping * theta_dot 

    force_vector = ca.vertcat(
        control_force + coriolis_gravity_cart + damping_force_cart + friction_force_cart,
        coriolis_gravity_pole + damping_torque_pole
    )

    # Solve for accelerations: q_ddot = M(q)^-1 * force_vector
    accelerations = ca.solve(inertia_matrix, force_vector)
    x_ddot = accelerations[0]
    theta_ddot = accelerations[1]

    # Return state derivative vector
    return ca.vertcat(x_dot, theta_dot, x_ddot, theta_ddot)

def _rk4_integration(x: ca.SX, u: ca.SX, dt: float, params: Dict[str, Any]) -> ca.SX:
    """
    Perform one step of RK4 integration using the full state derivative.
    """
    # Define the state derivative function locally for cleaner access
    # This function now includes damping/friction.
    f = lambda x_state, u_input: _calculate_state_derivatives(x_state, u_input, params)
    
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def pendulum_dynamics(x: ca.SX, u: ca.SX, dt: float, params: Dict[str, Any], 
                     integration_method: str = 'rk4') -> ca.SX:
    """
    Computes the next state using Euler or RK4 integration.
    Relies on _calculate_state_derivatives for the underlying ODEs.
    State: [x, theta, x_dot, theta_dot].
    (Docstring parameters are the same as before)
    """
    # Unpack state velocities needed for Euler
    x_dot_vel, theta_dot_vel = x[2], x[3]
    
    # Choose integration method
    if integration_method == 'euler':
        # Euler needs accelerations explicitly
        # We can get them from the state derivative function
        state_deriv = _calculate_state_derivatives(x, u, params)
        x_ddot = state_deriv[2]
        theta_ddot = state_deriv[3]
        
        # Perform Euler step
        x_pos_theta = ca.vertcat(x[0], x[1]) # Position states
        x_vel_vec = ca.vertcat(x_dot_vel, theta_dot_vel) # Velocity states
        x_ddot_vec = ca.vertcat(x_ddot, theta_ddot) # Acceleration states
        x_next = _euler_integration(x_pos_theta, x_vel_vec, x_ddot_vec, dt)
        
    elif integration_method == 'rk4':
        # RK4 uses the state derivative function directly
        x_next = _rk4_integration(x, u, dt, params)
    else:
        raise ValueError(f"Unknown integration method: {integration_method}")

    return x_next

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from src.utils.parameters import load_inverted_pendulum_params
    import os

    print("Running CasADi dynamics simulation and visualization...")

    # --- Configuration ---
    PARAM_PATH = "src/environments/inverted_pendulum_params.json"
    DT = 0.02  # Simulation timestep
    SIM_STEPS = 5000  # Number of simulation steps (e.g., 100 seconds) - INCREASED x10
    INITIAL_STATE = np.array([0.0, 0.1, 0.0, 0.0]) # Start near upright
    CONTROL_INPUT = np.array([0.0]) # Zero control
    INTEGRATION_METHOD = 'rk4'

    # --- Load Parameters ---
    if not os.path.exists(PARAM_PATH):
        print(f"Error: Parameter file not found at {PARAM_PATH}")
        exit()
    params = load_inverted_pendulum_params(PARAM_PATH)
    pole_vis_length = params.get('pole_length', 0.6) # Use full length for vis

    # --- Setup CasADi Function ---
    x_sym = ca.SX.sym('x', 4)
    u_sym = ca.SX.sym('u', 1)
    x_next_sym = pendulum_dynamics(x_sym, u_sym, DT, params, integration_method=INTEGRATION_METHOD)
    
    # Create a CasADi function for numerical evaluation
    # Note: Need to use ca.Function which supports numerical inputs/outputs
    dynamics_func = ca.Function('dynamics', [x_sym, u_sym], [x_next_sym])

    # --- Simulate Dynamics ---
    print(f"Simulating {SIM_STEPS} steps with dt={DT} using {INTEGRATION_METHOD}...")
    states = np.zeros((SIM_STEPS + 1, 4))
    states[0, :] = INITIAL_STATE
    
    current_x = INITIAL_STATE
    for i in range(SIM_STEPS):
        # Convert numpy arrays to CasADi DM type for the function call
        x_next_dm = dynamics_func(ca.DM(current_x), ca.DM(CONTROL_INPUT))
        # Convert result back to numpy array
        current_x = x_next_dm.full().flatten()
        states[i + 1, :] = current_x
        if np.any(np.isnan(current_x)):
             print(f"Simulation stopped at step {i+1} due to NaN state.")
             SIM_STEPS = i # Truncate simulation
             states = states[:SIM_STEPS+1, :]
             break
             
    print("Simulation complete.")

    # --- Calculate Max Initial Swing Angle/Height ---
    # Find first peak angle (approx first quarter period)
    # Estimate period T approx 2*pi*sqrt(L/g) -> T/4
    # This is very approximate, maybe just search first ~100 steps
    first_swing_steps = min(SIM_STEPS // 2, 150) # Look within first ~3 seconds
    max_theta_first_swing = np.max(np.abs(states[:first_swing_steps, 1]))
    max_height_first_swing = pole_vis_length * np.cos(max_theta_first_swing) # y=L*cos(theta)
    print(f"Max angle in first ~{first_swing_steps*DT:.2f}s: {np.rad2deg(max_theta_first_swing):.2f} deg")
    print(f"Corresponding max height: {max_height_first_swing:.3f} m")

    # --- Setup Animation ---
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel("Cart Position (m)")
    ax.set_ylabel("Vertical Position (m)")
    ax.set_title(f"CasADi ({INTEGRATION_METHOD}) Inverted Pendulum Simulation (No Control)")

    # Determine plot limits dynamically
    max_cart_pos = np.max(np.abs(states[:, 0]))
    ax.set_xlim(states[0, 0] - max_cart_pos - pole_vis_length * 1.2, 
                states[0, 0] + max_cart_pos + pole_vis_length * 1.2)
    ax.set_ylim(-pole_vis_length * 1.2, pole_vis_length * 1.2)

    # Add horizontal lines for max initial height
    ax.axhline(max_height_first_swing, color='gray', linestyle='--', lw=1, label='Initial Max Height')
    ax.axhline(-max_height_first_swing, color='gray', linestyle='--', lw=1) # Symmetric line
    ax.legend(loc='upper right')

    # Create plot elements to update
    cart_width = 0.2
    cart_height = 0.1
    cart = plt.Rectangle((0, -cart_height/2), cart_width, cart_height, fc='blue')
    ax.add_patch(cart)
    pole, = ax.plot([], [], 'r-', lw=3) # Pole line
    pivot, = ax.plot([], [], 'ko', ms=5) # Pivot point
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top')

    # --- Animation Update Function ---
    def update(frame):
        x, theta = states[frame, 0], states[frame, 1]
        
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
    ani = animation.FuncAnimation(fig, update, frames=SIM_STEPS + 1, 
                                interval=DT * 1000, blit=True, repeat=False)
    plt.grid(True)
    plt.show()

    print("Visualization closed.")

    # --- Generate Static State Plots ---
    print("Generating state plots...")
    time_vector = np.arange(SIM_STEPS + 1) * DT
    fig_states, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    fig_states.suptitle('State Variables vs Time (CasADi Simulation)')

    state_labels = ['Cart Position (x) [m]', 'Pole Angle (theta) [rad]', 
                    'Cart Velocity (x_dot) [m/s]', 'Pole Angular Vel (theta_dot) [rad/s]']
    state_indices = [0, 1, 2, 3]

    for i, ax_i in enumerate(axs):
        idx = state_indices[i]
        ax_i.plot(time_vector, states[:, idx])
        ax_i.set_ylabel(state_labels[i])
        ax_i.grid(True)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    plt.show()

    print("State plots closed.")
