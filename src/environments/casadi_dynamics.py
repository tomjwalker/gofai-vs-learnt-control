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

def _rk4_integration(x: ca.SX, u: ca.SX, dt: float, params: Dict[str, Any]) -> ca.SX:
    """
    Perform one step of RK4 integration.
    
    Args:
        x: Current state
        u: Control input
        dt: Time step
        params: System parameters
        
    Returns:
        Next state after RK4 integration
    """
    def f(x_state, u_input):
        pos, theta, x_dot, theta_dot = x_state[0], x_state[1], x_state[2], x_state[3]
        
        # Recompute accelerations for this state
        cos_theta = ca.cos(theta)
        sin_theta = ca.sin(theta)
        inertia_matrix = ca.vertcat(
            ca.horzcat(params['cart_mass'] + params['pole_mass'], 
                      params['pole_mass'] * params['pole_half_length'] * cos_theta),
            ca.horzcat(params['pole_mass'] * params['pole_half_length'] * cos_theta, 
                      params['pole_inertia_about_y'] + params['pole_mass'] * params['pole_half_length'] ** 2)
        )
        force_vector = ca.vertcat(
            u_input + params['pole_mass'] * params['pole_half_length'] * theta_dot ** 2 * sin_theta,
            params['pole_mass'] * params['gravity'] * params['pole_half_length'] * sin_theta
        )
        accels = ca.solve(inertia_matrix, force_vector)
        
        return ca.vertcat(x_dot, theta_dot, accels[0], accels[1])
    
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def pendulum_dynamics(x: ca.SX, u: ca.SX, dt: float, params: Dict[str, Any], 
                     integration_method: str = 'rk4') -> ca.SX:
    """
    Computes the next state of the inverted pendulum using either Euler or RK4 integration
    with CasADi symbolic expressions. The state is assumed to follow the Gymnasium ordering:
    [x, theta, x_dot, theta_dot].

    This function implements the discrete-time dynamics for a cart-pole system where the rod (pole)
    has a distributed mass. The continuous-time equations of motion are expressed in mass matrix form:

        M(q)q_ddot = forces

    where:
        q = [x, theta]^T is the configuration vector
        M(q) = [M + m, m * d * cos(theta); m * d * cos(theta), I_pivot] is the mass matrix
        forces = [u + m * d * theta_dot^2 * sin(theta); m * g * d * sin(theta)] is the force vector

    The parameters are:
        - M is the cart mass.
        - m is the rod (pole) mass.
        - d is the distance from the pivot (cart) to the rod's center of mass.
        - I_pivot is the moment of inertia of the rod about the pivot,
          computed via the parallel axis theorem: I_pivot = I_pole_com + m * d^2,
          with I_pole_com being the inertia about the rod's center of mass.
        - g is the gravitational acceleration.

    The discrete dynamics can be computed using either:
    - Euler integration: x_next = x + dt * f(x, u)
    - RK4 integration: x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        x (ca.SX): Current state vector [x, theta, x_dot, theta_dot].
        u (ca.SX): Control input (scalar force on the cart).
        dt (float): Timestep for integration.
        params (Dict[str, Any]): Dictionary containing the pendulum parameters:
            - 'cart_mass': (M) Cart mass (kg).
            - 'pole_mass': (m) Rod mass (kg).
            - 'pole_half_length': (d) Half-length of the pole (m); used as the distance from pivot to COM.
            - 'pole_inertia_about_y': (I_pole_com) Rod inertia about its COM (kg·m²), for rotation about the y-axis.
            - 'gravity': Gravitational acceleration (m/s²).
        integration_method (str): Either 'euler' or 'rk4' (default).

    Returns:
        ca.SX: Next state vector [x_next, theta_next, x_dot_next, theta_dot_next] as a CasADi symbolic expression.
    """
    # Unpack state (Gymnasium ordering)
    pos, theta, x_dot, theta_dot = x[0], x[1], x[2], x[3]

    # Unpack parameters
    M = params['cart_mass']
    m = params['pole_mass']
    d = params['pole_half_length']  # distance from pivot to COM (half-length)
    I_pole_com = params['pole_inertia_about_y']
    g = params['gravity']

    # Compute the moment of inertia about the pivot using the parallel axis theorem.
    I_pivot = I_pole_com + m * d ** 2

    # Precompute trigonometric functions using CasADi.
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)

    # Build the inertia matrix symbolically.
    # Note: We use ca.vertcat and ca.horzcat to form a 2x2 matrix.
    inertia_matrix = ca.vertcat(
        ca.horzcat(M + m, m * d * cos_theta),
        ca.horzcat(m * d * cos_theta, I_pivot)
    )

    # Build the force vector symbolically.
    force_vector = ca.vertcat(
        u + m * d * theta_dot ** 2 * sin_theta,
        m * g * d * sin_theta
    )

    # Solve for the accelerations: [x_ddot, theta_ddot]
    # ca.solve solves the linear system inertia_matrix * accelerations = force_vector:
    # https://web.casadi.org/docs/#linear-algebra
    accelerations = ca.solve(inertia_matrix, force_vector)
    x_ddot = accelerations[0]
    theta_ddot = accelerations[1]

    # Choose integration method
    if integration_method == 'euler':
        # Euler integration step to obtain next state.
        x_dot_vec = ca.vertcat(x_dot, theta_dot)
        x_ddot_vec = ca.vertcat(x_ddot, theta_ddot)
        x_next = _euler_integration(ca.vertcat(pos, theta), x_dot_vec, x_ddot_vec, dt)
    elif integration_method == 'rk4':
        x_next = _rk4_integration(x, u, dt, params)
    else:
        raise ValueError(f"Unknown integration method: {integration_method}")

    return x_next
