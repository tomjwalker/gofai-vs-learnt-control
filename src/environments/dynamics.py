import numpy as np


def pendulum_dynamics(x, u, dt, params):
    """
    Discrete-time inverted pendulum dynamics, clearly matching Gymnasium observation order.

    Args:
        x (array-like): State [x, theta, x_dot, theta_dot] (consistent with Gymnasium).
        u (float): Control input (horizontal force on the cart, in Newtons).
        dt (float): Timestep for Euler discretisation (s).
        params (dict): Pendulum parameters:
            - M (float): Cart mass (kg).
            - m (float): Rod mass (kg).
            - d (float): Distance from pivot to rod COM (m).
            - I_pivot (float): Rod moment of inertia about pivot (kg·m²).
            - g (float): Gravitational acceleration (m/s²).

    Returns:
        np.ndarray: Next state vector [x, theta, x_dot, theta_dot].
    """
    # Unpack clearly (matching Gymnasium API)
    pos, theta, vel, theta_dot = x

    # Unpack parameters clearly
    M = params['cart_mass']
    m = params['pole_mass']
    d = params['pole_half_length']
    I_pole_com = params['pole_inertia_about_y']
    g = params['gravity']

    # Parallel axis theorem to get inertia about pivot
    I_pivot = I_pole_com + m * d ** 2

    # Pre-compute some terms for readability
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Inertia matrix
    inertia_matrix = np.array([[M + m, m * d * cos_theta], [m * d * cos_theta, I_pivot]])

    # Force vector
    force_vector = np.array([u + m * d * theta_dot ** 2 * sin_theta, -m * g * d * sin_theta])

    # Placeholder for accelerations (your next step clearly is to solve these explicitly)
    accelerations = np.linalg.solve(inertia_matrix, force_vector)
    x_ddot = accelerations[0]
    theta_ddot = accelerations[1]

    # Euler integration clearly matching Gymnasium state ordering
    next_pos = pos + vel * dt
    next_theta = theta + theta_dot * dt
    next_vel = vel + x_ddot * dt
    next_theta_dot = theta_dot + theta_ddot * dt

    x_next = np.array([next_pos, next_theta, next_vel, next_theta_dot])

    return x_next
