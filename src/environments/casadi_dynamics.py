# casadi_dynamics.py
import casadi as ca
from typing import Dict, Any


def pendulum_dynamics(x: ca.SX, u: ca.SX, dt: float, params: Dict[str, Any]) -> ca.SX:
    """
    Computes the next state of the inverted pendulum using Euler integration
    with CasADi symbolic expressions. The state is assumed to follow the Gymnasium ordering:
    [x, theta, x_dot, theta_dot].

    This function implements the discrete-time dynamics for a cart-pole system where the rod (pole)
    has a distributed mass. The continuous-time equations used are:

        (M + m) * x_ddot + m * d * (theta_ddot * cos(theta) - theta_dot^2 * sin(theta)) = u
        I_pivot * theta_ddot + m * d * x_ddot * cos(theta) + m * g * d * sin(theta) = 0

    where:
        - M is the cart mass.
        - m is the rod (pole) mass.
        - d is the distance from the pivot (cart) to the rod’s center of mass.
        - I_pivot is the moment of inertia of the rod about the pivot,
          computed via the parallel axis theorem: I_pivot = I_pole_com + m * d^2,
          with I_pole_com being the inertia about the rod’s center of mass.
        - g is the gravitational acceleration.

    The discrete dynamics are computed by performing one Euler integration step:
        x_next = x + dt * f(x, u)
    where f(x, u) includes the computed accelerations.

    Args:
        x (ca.SX): Current state vector [x, theta, x_dot, theta_dot].
        u (ca.SX): Control input (scalar force on the cart).
        dt (float): Timestep for Euler integration.
        params (Dict[str, Any]): Dictionary containing the pendulum parameters:
            - 'cart_mass': (M) Cart mass (kg).
            - 'pole_mass': (m) Rod mass (kg).
            - 'pole_half_length': (d) Half-length of the pole (m); used as the distance from pivot to COM.
            - 'pole_inertia_about_y': (I_pole_com) Rod inertia about its COM (kg·m²), for rotation about the y-axis.
            - 'gravity': Gravitational acceleration (m/s²).

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
        -m * g * d * sin_theta
    )

    # Solve for the accelerations: [x_ddot, theta_ddot]
    accelerations = ca.solve(inertia_matrix, force_vector)
    x_ddot = accelerations[0]
    theta_ddot = accelerations[1]

    # Euler integration step to obtain next state.
    next_pos = pos + x_dot * dt
    next_theta = theta + theta_dot * dt
    next_vel = x_dot + x_ddot * dt
    next_theta_dot = theta_dot + theta_ddot * dt

    # Return the next state as a CasADi symbolic expression.
    x_next = ca.vertcat(next_pos, next_theta, next_vel, next_theta_dot)
    return x_next
