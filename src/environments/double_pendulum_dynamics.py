from __future__ import annotations

import casadi as ca
import numpy as np

CasadiVector = ca.SX | ca.DM


def _calculate_state_derivatives_double(
    state: CasadiVector,
    control: CasadiVector,
    params: CasadiVector,
) -> CasadiVector:
    """Calculate `[x_dot, theta1_dot, theta2_dot, x_ddot, theta1_ddot, theta2_ddot]`."""
    x_d = state[3]
    theta1 = state[1]
    theta2 = state[2]
    theta1_d = state[4]
    theta2_d = state[5]

    control_input = control[0]
    cart_mass = params[0]
    pole1_mass = params[1]
    pole2_mass = params[2]
    pole1_length = params[3]
    _pole2_length = params[4]
    pole1_com_distance = params[5]
    pole2_com_distance = params[6]
    pole1_inertia = params[7]
    pole2_inertia = params[8]
    gravity = params[9]
    slider_damping = params[10]
    cart_friction = params[11]
    joint1_damping = params[12]
    joint2_damping = params[13]
    gear = params[14]

    mass_matrix = ca.vertcat(
        ca.horzcat(
            cart_mass + pole1_mass + pole2_mass,
            (pole1_com_distance * pole1_mass + pole1_length * pole2_mass) * ca.cos(theta1),
            pole2_com_distance * pole2_mass * ca.cos(theta2),
        ),
        ca.horzcat(
            (pole1_com_distance * pole1_mass + pole1_length * pole2_mass) * ca.cos(theta1),
            pole1_inertia + pole1_com_distance**2 * pole1_mass + pole1_length**2 * pole2_mass,
            pole2_com_distance * pole1_length * pole2_mass * ca.cos(theta1 - theta2),
        ),
        ca.horzcat(
            pole2_com_distance * pole2_mass * ca.cos(theta2),
            pole2_com_distance * pole1_length * pole2_mass * ca.cos(theta1 - theta2),
            pole2_inertia + pole2_com_distance**2 * pole2_mass,
        ),
    )

    rhs_vector = ca.vertcat(
        control_input * gear
        - cart_friction * x_d
        - slider_damping * x_d
        + pole1_com_distance * pole1_mass * theta1_d**2 * ca.sin(theta1)
        + pole2_com_distance * pole2_mass * theta2_d**2 * ca.sin(theta2)
        + pole1_length * pole2_mass * theta1_d**2 * ca.sin(theta1),
        -joint1_damping * theta1_d
        + pole1_com_distance * gravity * pole1_mass * ca.sin(theta1)
        - pole2_com_distance * pole1_length * pole2_mass * theta2_d**2 * ca.sin(theta1 - theta2)
        + gravity * pole1_length * pole2_mass * ca.sin(theta1),
        -joint2_damping * theta2_d
        + pole2_com_distance * gravity * pole2_mass * ca.sin(theta2)
        + pole2_com_distance * pole1_length * pole2_mass * theta1_d**2 * ca.sin(theta1 - theta2),
    )

    accelerations = ca.solve(mass_matrix, rhs_vector)
    return ca.vertcat(x_d, theta1_d, theta2_d, accelerations[0], accelerations[1], accelerations[2])


def _rk4_integration_double(
    x: CasadiVector,
    u: CasadiVector,
    dt: float,
    params: CasadiVector,
) -> CasadiVector:
    """Perform one RK4 step using the double-pendulum state derivative."""

    def derivative(x_state: CasadiVector, u_input: CasadiVector) -> CasadiVector:
        return _calculate_state_derivatives_double(x_state, u_input, params)

    k1 = derivative(x, u)
    k2 = derivative(x + dt / 2 * k1, u)
    k3 = derivative(x + dt / 2 * k2, u)
    k4 = derivative(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def get_double_pendulum_ode_function() -> ca.Function:
    """Create a CasADi function for the continuous-time double-pendulum ODE."""
    state_symbol = ca.SX.sym("state", 6)
    control_symbol = ca.SX.sym("control", 1)
    params_symbol = ca.SX.sym("params", 15)
    state_dot = _calculate_state_derivatives_double(state_symbol, control_symbol, params_symbol)
    return ca.Function(
        "double_pendulum_dynamics_ode",
        [state_symbol, control_symbol, params_symbol],
        [state_dot],
        ["state", "control", "params"],
        ["state_dot"],
    )


def double_pendulum_dynamics_step(
    x: CasadiVector,
    u: CasadiVector,
    dt: float,
    params: CasadiVector,
    *,
    integration_method: str = "rk4",
) -> CasadiVector:
    """Compute the next double-pendulum state using the configured integrator."""
    if integration_method == "rk4":
        return _rk4_integration_double(x, u, dt, params)
    raise ValueError(f"Unknown integration method: {integration_method}")


if __name__ == "__main__":
    ode_function = get_double_pendulum_ode_function()
    state_value = np.zeros(6)
    control_value = np.zeros(1)
    parameter_value = np.ones(15)

    print("Testing the generated CasADi function...")
    print("ODE function created:", ode_function)
    print(
        "Parameter order: ['M', 'm1', 'm2', 'l1', 'l2', 'd1', 'd2', "
        "'Icm1', 'Icm2', 'g', 'b_slide', 'b_fric', 'b_joint1', 'b_joint2', 'gear']"
    )
    print("Using placeholder parameters:", parameter_value)
    print("Calculated state_dot:", ode_function(state_value, control_value, parameter_value))
    print(
        "Calculated next_state (RK4, dt=0.01):",
        double_pendulum_dynamics_step(state_value, control_value, 0.01, parameter_value),
    )
