from __future__ import annotations

from typing import Any

import casadi as ca

PendulumParams = dict[str, Any]


def _euler_integration(
    position_state: ca.SX | ca.DM,
    velocity_state: ca.SX | ca.DM,
    acceleration_state: ca.SX | ca.DM,
    dt: float,
) -> ca.SX | ca.DM:
    """Perform one explicit Euler step for the single-pendulum state."""
    next_pos = position_state[0] + velocity_state[0] * dt
    next_theta = position_state[1] + velocity_state[1] * dt
    next_vel = velocity_state[0] + acceleration_state[0] * dt
    next_theta_dot = velocity_state[1] + acceleration_state[1] * dt
    return ca.vertcat(next_pos, next_theta, next_vel, next_theta_dot)


def _calculate_state_derivatives(
    state: ca.SX | ca.DM,
    control: ca.SX | ca.DM,
    params: PendulumParams,
) -> ca.SX | ca.DM:
    """Return `[x_dot, theta_dot, x_ddot, theta_ddot]` for the single pendulum."""
    theta, x_dot, theta_dot = state[1], state[2], state[3]

    gravity = float(params.get("gravity", 9.81))
    pole_half_length = float(params.get("pole_half_length", 0.25))
    pole_mass = float(params.get("pole_mass", 0.1))
    cart_mass = float(params.get("cart_mass", 1.0))
    actuator_gear_dict = params.get("actuator_gear", {"slide": 1.0})
    actuator_gear = float(actuator_gear_dict.get("slide", 1.0))
    joint_damping = float(params.get("joint_damping", 1.0))
    slider_damping = float(params.get("slider_damping", 0.0))
    cart_friction = float(params.get("cart_friction", 0.1))
    pole_inertia_com_y = float(params.get("pole_inertia_about_y", 0.001))

    inertia_about_pivot = pole_inertia_com_y + pole_mass * pole_half_length**2
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)

    inertia_matrix = ca.vertcat(
        ca.horzcat(cart_mass + pole_mass, pole_mass * pole_half_length * cos_theta),
        ca.horzcat(pole_mass * pole_half_length * cos_theta, inertia_about_pivot),
    )

    coriolis_gravity_cart = pole_mass * pole_half_length * theta_dot**2 * sin_theta
    coriolis_gravity_pole = pole_mass * gravity * pole_half_length * sin_theta
    control_force = actuator_gear * control[0]
    damping_force_cart = -slider_damping * x_dot
    friction_force_cart = -cart_friction * x_dot
    damping_torque_pole = -joint_damping * theta_dot

    force_vector = ca.vertcat(
        control_force + coriolis_gravity_cart + damping_force_cart + friction_force_cart,
        coriolis_gravity_pole + damping_torque_pole,
    )

    accelerations = ca.solve(inertia_matrix, force_vector)
    return ca.vertcat(x_dot, theta_dot, accelerations[0], accelerations[1])


def _rk4_integration(
    state: ca.SX | ca.DM,
    control: ca.SX | ca.DM,
    dt: float,
    params: PendulumParams,
) -> ca.SX | ca.DM:
    """Perform one RK4 step using the full single-pendulum state derivative."""

    def derivative(x_state: ca.SX | ca.DM, u_input: ca.SX | ca.DM) -> ca.SX | ca.DM:
        return _calculate_state_derivatives(x_state, u_input, params)

    k1 = derivative(state, control)
    k2 = derivative(state + dt / 2 * k1, control)
    k3 = derivative(state + dt / 2 * k2, control)
    k4 = derivative(state + dt * k3, control)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def pendulum_dynamics(
    x: ca.SX | ca.DM,
    u: ca.SX | ca.DM,
    dt: float,
    params: PendulumParams,
    *,
    integration_method: str = "rk4",
) -> ca.SX | ca.DM:
    """Compute the next single-pendulum state using Euler or RK4 integration."""
    x_dot_vel, theta_dot_vel = x[2], x[3]

    if integration_method == "euler":
        state_derivative = _calculate_state_derivatives(x, u, params)
        acceleration_vector = ca.vertcat(state_derivative[2], state_derivative[3])
        position_vector = ca.vertcat(x[0], x[1])
        velocity_vector = ca.vertcat(x_dot_vel, theta_dot_vel)
        return _euler_integration(position_vector, velocity_vector, acceleration_vector, dt)

    if integration_method == "rk4":
        return _rk4_integration(x, u, dt, params)

    raise ValueError(f"Unknown integration method: {integration_method}")
