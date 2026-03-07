from __future__ import annotations

import casadi as ca

CasadiVector = ca.SX | ca.DM
CasadiMatrix = ca.SX | ca.DM


class MPCCostBase:
    """Base interface for MPC stage and terminal costs."""

    def calculate_stage_cost(
        self,
        Xk: CasadiVector,
        Uk: CasadiVector,
        Xref: CasadiVector,
        Q: CasadiMatrix,
        R: CasadiMatrix,
    ) -> CasadiMatrix:
        raise NotImplementedError

    def calculate_terminal_cost(
        self,
        XN: CasadiVector,
        Xref: CasadiVector,
        Q_terminal: CasadiMatrix,
    ) -> CasadiMatrix:
        raise NotImplementedError


class QuadraticCost(MPCCostBase):
    """Standard quadratic MPC cost."""

    def calculate_stage_cost(
        self,
        Xk: CasadiVector,
        Uk: CasadiVector,
        Xref: CasadiVector,
        Q: CasadiMatrix,
        R: CasadiMatrix,
    ) -> CasadiMatrix:
        state_error = Xk - Xref
        return ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([Uk.T, R, Uk])

    def calculate_terminal_cost(
        self,
        XN: CasadiVector,
        Xref: CasadiVector,
        Q_terminal: CasadiMatrix,
    ) -> CasadiMatrix:
        terminal_error = XN - Xref
        return ca.mtimes([terminal_error.T, Q_terminal, terminal_error])


class PendulumSwingupCost(MPCCostBase):
    """Swing-up cost using quadratic terms plus `(1 - cos(theta))` for angle error."""

    def calculate_stage_cost(
        self,
        Xk: CasadiVector,
        Uk: CasadiVector,
        Xref: CasadiVector,
        Q: CasadiMatrix,
        R: CasadiMatrix,
    ) -> CasadiMatrix:
        state_error = Xk - Xref
        cost_x = state_error[0] ** 2 * Q[0, 0]
        cost_x_dot = state_error[2] ** 2 * Q[2, 2]
        cost_theta_dot = state_error[3] ** 2 * Q[3, 3]
        cost_theta = (1 - ca.cos(Xk[1])) * Q[1, 1]
        cost_control = ca.mtimes([Uk.T, R, Uk])
        return cost_x + cost_theta + cost_x_dot + cost_theta_dot + cost_control

    def calculate_terminal_cost(
        self,
        XN: CasadiVector,
        Xref: CasadiVector,
        Q_terminal: CasadiMatrix,
    ) -> CasadiMatrix:
        terminal_error = XN - Xref
        cost_x = terminal_error[0] ** 2 * Q_terminal[0, 0]
        cost_x_dot = terminal_error[2] ** 2 * Q_terminal[2, 2]
        cost_theta_dot = terminal_error[3] ** 2 * Q_terminal[3, 3]
        cost_theta = (1 - ca.cos(XN[1])) * Q_terminal[1, 1]
        return cost_x + cost_theta + cost_x_dot + cost_theta_dot


class PendulumSwingupAtan2Cost(MPCCostBase):
    """Swing-up cost using a wrapped-angle error from `atan2(sin(theta), cos(theta))`."""

    def calculate_stage_cost(
        self,
        Xk: CasadiVector,
        Uk: CasadiVector,
        Xref: CasadiVector,
        Q: CasadiMatrix,
        R: CasadiMatrix,
    ) -> CasadiMatrix:
        state_error = Xk - Xref
        cost_x = state_error[0] ** 2 * Q[0, 0]
        cost_x_dot = state_error[2] ** 2 * Q[2, 2]
        cost_theta_dot = state_error[3] ** 2 * Q[3, 3]
        angle_error = ca.atan2(ca.sin(Xk[1]), ca.cos(Xk[1]))
        cost_theta = angle_error**2 * Q[1, 1]
        cost_control = ca.mtimes([Uk.T, R, Uk])
        return cost_x + cost_theta + cost_x_dot + cost_theta_dot + cost_control

    def calculate_terminal_cost(
        self,
        XN: CasadiVector,
        Xref: CasadiVector,
        Q_terminal: CasadiMatrix,
    ) -> CasadiMatrix:
        terminal_error = XN - Xref
        cost_x = terminal_error[0] ** 2 * Q_terminal[0, 0]
        cost_x_dot = terminal_error[2] ** 2 * Q_terminal[2, 2]
        cost_theta_dot = terminal_error[3] ** 2 * Q_terminal[3, 3]
        terminal_angle_error = ca.atan2(ca.sin(XN[1]), ca.cos(XN[1]))
        cost_theta = terminal_angle_error**2 * Q_terminal[1, 1]
        return cost_x + cost_theta + cost_x_dot + cost_theta_dot


COST_FUNCTION_MAP = {
    "quadratic": QuadraticCost,
    "pendulum_swingup": PendulumSwingupCost,
    "pendulum_atan2": PendulumSwingupAtan2Cost,
}
