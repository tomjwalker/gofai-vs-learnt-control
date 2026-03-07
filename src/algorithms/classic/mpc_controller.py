from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import casadi as ca
import numpy as np

from src.algorithms.classic.mpc_costs import COST_FUNCTION_MAP, MPCCostBase
from src.algorithms.classic.mpc_guesses import GUESS_STRATEGY_MAP, MPCGuessBase
from src.environments.pendulum_dynamics import pendulum_dynamics
from src.utils.parameters import load_pendulum_params

PendulumParams: TypeAlias = dict[str, Any]
SolverOutputs: TypeAlias = dict[str, Any]


def build_mpc_bounds(
    state_bounds_obs: list[list[float]],
    control_bounds: list[list[float]],
    N: int,
    joint_bounds: dict[str, list[float]] | None = None,
) -> tuple[list[float], list[float]]:
    """Construct lower and upper bounds for the stacked MPC state/control variables."""
    lbx: list[float] = []
    ubx: list[float] = []
    effective_joint_bounds = joint_bounds or {}
    state_names = ["cart_pos", "pole_angle", "cart_vel", "pole_ang_vel"]

    for _ in range(N + 1):
        for index, state_name in enumerate(state_names):
            if state_name in effective_joint_bounds:
                lower_bound, upper_bound = effective_joint_bounds[state_name]
            else:
                lower_bound, upper_bound = state_bounds_obs[index]

            lbx.append(float(lower_bound))
            ubx.append(float(upper_bound))

    for _ in range(N):
        for lower_bound, upper_bound in control_bounds:
            lbx.append(float(lower_bound))
            ubx.append(float(upper_bound))

    return lbx, ubx


def build_mpc_solver(
    params: PendulumParams,
    N: int,
    dt_controller: float,
    Q: ca.DM,
    R: ca.DM,
    Q_terminal: ca.DM,
    X_ref: ca.DM,
    cost_calculator: MPCCostBase,
) -> tuple[ca.Function, ca.SX, ca.SX, ca.SX, list[float], list[float]]:
    """Build the CasADi NLP for the single-pendulum MPC controller."""
    X = ca.SX.sym("X", 4, N + 1)
    U = ca.SX.sym("U", 1, N)
    X_init = ca.SX.sym("X_init", 4)

    cost = 0
    for k in range(N):
        cost += cost_calculator.calculate_stage_cost(X[:, k], U[:, k], X_ref, Q, R)
    cost += cost_calculator.calculate_terminal_cost(X[:, N], X_ref, Q_terminal)

    initial_state_constraint = X[:, 0] - X_init
    dynamics_constraints = [initial_state_constraint]
    for k in range(N):
        X_next = pendulum_dynamics(
            X[:, k],
            U[:, k],
            dt_controller,
            params,
            integration_method="rk4",
        )
        dynamics_constraints.append(X[:, k + 1] - X_next)

    constraint_vector = ca.vertcat(*dynamics_constraints)
    lbx, ubx = build_mpc_bounds(
        params["state_bounds_obs"],
        params["control_bounds"],
        N,
        params.get("joint_bounds"),
    )
    opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nlp = {"x": opt_vars, "f": cost, "g": constraint_vector, "p": X_init}

    solver_options = {
        "ipopt.print_level": 0,
        "ipopt.tol": 1e-6,
        "ipopt.max_iter": 200,
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "limited-memory",
        "print_time": False,
        "ipopt.warm_start_init_point": "yes",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.ma57_automatic_scaling": "yes",
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, solver_options)
    return solver, X, U, X_init, lbx, ubx


class MPCController:
    """Typed MPC controller for the single inverted pendulum."""

    def __init__(
        self,
        N: int = 30,
        dt_controller: float = 0.02,
        param_path: str = "src/environments/pendulum_params.json",
        cost_type: str = "quadratic",
        guess_type: str = "warmstart",
        q_diag: list[float] | None = None,
        r_val: float | None = None,
        q_terminal_multiplier: float = 5.0,
    ) -> None:
        self.N = N
        self.dt_controller = dt_controller
        self.cost_type = cost_type
        self.guess_type = guess_type
        self.param_path = param_path

        if not Path(self.param_path).exists():
            raise FileNotFoundError(f"Parameter file not found at: {self.param_path}")
        self.params = load_pendulum_params(self.param_path)

        cost_calculator_class = COST_FUNCTION_MAP.get(cost_type)
        if cost_calculator_class is None:
            available = sorted(COST_FUNCTION_MAP)
            raise ValueError(f"Unknown cost_type '{cost_type}'. Available types: {available}")
        self.cost_calculator: MPCCostBase = cost_calculator_class()

        guess_strategy_class = GUESS_STRATEGY_MAP.get(guess_type)
        if guess_strategy_class is None:
            available = sorted(GUESS_STRATEGY_MAP)
            raise ValueError(f"Unknown guess_type '{guess_type}'. Available types: {available}")
        self.guess_strategy: MPCGuessBase = guess_strategy_class()

        default_q_diag = [1.0, 20.0, 5.0, 10.0]
        current_q_diag = q_diag if q_diag is not None and len(q_diag) == 4 else default_q_diag
        current_r_val = r_val if r_val is not None else 50.0

        self.Q = ca.diag(current_q_diag)
        self.R = ca.DM([current_r_val])
        self.Q_terminal = q_terminal_multiplier * self.Q
        self.X_ref = ca.DM.zeros(4, 1)
        self.solver, self.X, self.U, self.X_init, self.lbx, self.ubx = build_mpc_solver(
            self.params,
            self.N,
            self.dt_controller,
            self.Q,
            self.R,
            self.Q_terminal,
            self.X_ref,
            self.cost_calculator,
        )
        self.X_prev: np.ndarray | None = None
        self.U_prev: np.ndarray | None = None

    def solve(self, x0: np.ndarray) -> SolverOutputs:
        """Solve the MPC problem for the current state and return the full solver payload."""
        X_guess, U_guess = self.guess_strategy.get_guess(x0=x0, N=self.N, controller=self)
        initial_vector = np.concatenate([X_guess.T.flatten(), U_guess.T.flatten()])

        try:
            solution = self.solver(
                x0=initial_vector,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=0,
                ubg=0,
                p=x0.ravel(),
            )
        except Exception:
            return {
                "X_solution": None,
                "U_solution": None,
                "u_next": 0.0,
                "cost": np.inf,
                "constraint_violation": np.inf,
                "solver_status": "failed",
            }

        optimal_vars = np.asarray(solution["x"].full()).ravel()
        state_width = len(x0)
        x_block_length = state_width * (self.N + 1)
        X_solution = optimal_vars[:x_block_length].reshape((self.N + 1, state_width)).T
        U_solution = optimal_vars[x_block_length:].reshape((self.N, 1)).T
        u_next = float(U_solution[0, 0])
        cost_value = float(solution["f"].full().item()) if "f" in solution else float("inf")
        constraint_violation = (
            float(np.linalg.norm(solution["g"].full())) if "g" in solution else float("inf")
        )
        solver_status = str(self.solver.stats().get("return_status", "unknown"))

        self.X_prev = X_solution
        self.U_prev = U_solution
        return {
            "X_solution": X_solution,
            "U_solution": U_solution,
            "u_next": u_next,
            "cost": cost_value,
            "constraint_violation": constraint_violation,
            "solver_status": solver_status,
        }

    def step(self, x0: np.ndarray) -> float:
        """Return the first MPC control input, or `0.0` if the solve fails."""
        solver_outputs = self.solve(x0)
        if (
            solver_outputs.get("solver_status") == "failed"
            or solver_outputs.get("X_solution") is None
        ):
            self.X_prev = None
            self.U_prev = None
            return 0.0
        return float(solver_outputs["u_next"])
