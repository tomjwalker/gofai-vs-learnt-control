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
