"""

"""

import casadi as ca
import numpy as np
from typing import Tuple, Dict

from src.utils.parameters import load_inverted_pendulum_params
from src.utils.mpc_helpers import build_mpc_bounds
from src.environments.casadi_dynamics import pendulum_dynamics

# TODO: remove hardcoded path
# Set working directory to the root of the project, without any relative path logic
import os
os.chdir(os.path.join(os.path.dirname(__file__), "../.."))
print(os.getcwd())


def build_mpc_solver(
    params: Dict,
    N: int,
    dt: float,
    Q: ca.DM,
    R: ca.DM,
    Q_terminal: ca.DM,
    X_ref: ca.DM
) -> Tuple[ca.Function, ca.SX, ca.SX, ca.SX, list, list]:
    """
    Build an MPC solver for the inverted pendulum environment.

    Args:
        params: Dict of physical/environment parameters, including bounds.
        N: Prediction horizon.
        dt: Timestep.
        Q: State weighting matrix.
        R: Control weighting matrix.
        Q_terminal: Terminal cost matrix.
        X_ref: Reference state to track.

    Returns:
        Tuple containing:
            - solver: CasADi NLP solver instance
            - X: Symbolic state trajectory variable (4 x N+1)
            - U: Symbolic control trajectory variable (1 x N)
            - X_init: Symbolic initial condition (4 x 1)
            - lbx, ubx: Lower/upper bounds for the optimisation
    """

    # === Define decision variables (states and control inputs) ===
    # - ca.SX is CasADI's symbolic type for scalar expressions (MX for more complex expressions)
    # - ca.SX.sym creates symbolic variables with a name and dimensions.
    # - This is how we define decision variables for MPC in CasADI.
    X = ca.SX.sym("X", 4, N + 1)  # state trajectory. 4 states: [x, theta, x_dot, theta_dot]. Over N+1 timesteps as
    # include initial state.
    U = ca.SX.sym("U", 1, N)  # control trajectory. 1 control input: horizontal force on the cart. Over N timesteps as
    # only apply control at each timestep.

    # === Define MPC cost function ===

    # Initialize cost
    cost = 0
    # Loop over each timestep in the prediction horizon and accumulate the cost
    for k in range(N):
        # Error between the predicted state and the reference state
        state_error = X[:, k] - X_ref
        control = U[:, k]
        # Calculate timestep cost
        timestep_cost = ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([control.T, R, control])
        # Accumulate cost
        cost += timestep_cost
    # Add terminal cost
    terminal_error = X[:, N] - X_ref
    terminal_cost = ca.mtimes([terminal_error.T, Q_terminal, terminal_error])
    cost += terminal_cost

    # === Define MPC constraints ===

    # Define initial state X_0
    X_init = ca.SX.sym("X_init", 4)  # Initial state: [x, theta, x_dot, theta_dot]

    # Enforce that the first state in the decision variable equals the initial state
    initial_state_constraint = X[:, 0] - X_init

    # Build a list to store the dynamics constraints
    dynamics_constraints = []

    # For each timestep in the prediction horizon, enforce the discrete-time dynamics (that the next state is the result
    # of applying the dynamics function to the current state and control input)
    for k in range(N):
        # X[:, k] is the current state
        # U[:, k] is the current control input
        # X[:, k + 1] = pendulum_dynamics(X[:, k], U[:, k], dt, params) is the next state predicted by the dynamics
        X_next = pendulum_dynamics(X[:, k], U[:, k], dt, params)

        # Enforce that the predicted state equals the decision variable for the next timestep
        dynamics_constraints.append(X[:, k + 1] - X_next)

    # Concatenate all dynamics constraints into a single vector
    dynamics_constraints = ca.vertcat(initial_state_constraint, *dynamics_constraints)

    # === Build bounds on state and control vectors ===

    state_bounds = params["state_bounds"]
    control_bounds = params["control_bounds"]
    lbx, ubx = build_mpc_bounds(state_bounds, control_bounds, N)

    # === Compile the NLP problem ===

    # Reshape the decision variables as a single vector
    opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # Define the constrained non-linear program optimisation. The CasADi NLP dictionary has the form: nlp = {
    #     "x": The `opt_vars` column vector stacking the state X and control U vectors
    #     "f": The `cost` function f(x, p)
    #     "g": The set of constraint functions g(x, p)
    #     "p": Additional CasADi symbolic variables which aren't the variables being optimised over (X, U) but are
    #         nevertheless required for f() and g(); e.g. the initial values X_init for the initial state constraint
    # }
    # (see https://web.casadi.org/docs/#nonlinear-programming for further details)
    nlp = {"x": opt_vars, "f": cost, "g": dynamics_constraints, "p": X_init}

    # === Set up the solver ===

    # Solver options
    opts = {
        "ipopt.print_level": 0,
        "ipopt.tol": 1e-4,
        "ipopt.max_iter": 100,
        "print_time": False
    }

    # Create an IPOPT solver instance
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    return solver, X, U, X_init, lbx, ubx


class MPCController:
    """
    Model Predictive Controller (MPC) for the Gymnasium/MuJoCo Inverted Pendulum.

    Responsibilities:
    - Load system parameters
    - Set up the cost function and dynamics constraints
    - Build and solve the CasADi NLP
    - Provide a `step()` method that returns the first action for closed-loop rollout
    """

    def __init__(self,
                 N: int = 10,
                 dt: float = 0.05,
                 param_path: str = "environments/inverted_pendulum_params.json"):
        """
        Initialise MPC with problem horizon, timestep, and physical parameters.
        - Loads environment parameters
        - Defines cost matrices Q, R, Q_terminal
        - Defines reference state
        - Builds solver using build_mpc_solver
        - Stores symbolic variables and bounds
        """

        # Load parameters
        self.params = load_inverted_pendulum_params(param_path)

        # Set instance variables
        self.N = N
        self.dt = dt

        # Define cost matrices
        self.Q = ca.diag([1.0, 10.0, 0.1, 0.1])
        self.R = ca.DM([0.01])
        self.Q_terminal = self.Q

        # Define reference state
        self.X_ref = ca.DM.zeros(4, 1)

        # Build solver
        self.solver, self.X, self.U, self.X_init, self.lbx, self.ubx = build_mpc_solver(
            self.params, self.N, self.dt, self.Q, self.R, self.Q_terminal, self.X_ref
        )

        # Attributes to store X_prev and U_prev, which can be used to warm-start the next timestep's MPC initial guess
        self.X_prev = None    # Initialise to None - this can then be used as a test for whether it is first timestep
        self.U_prev = None

    def get_guesses_basic(self, x0: np.ndarray) -> np.ndarray:
        """
        Helper function for `solve`. On each real timestep, we pass initial guesses X_guess and U_guess to the
        CasADi solver to help localise it in the search space to ensure it finds a feasible solution.

        Basic version:
        - X_guess is a repeat of the x0 vector (assume system doesn't move from x0), of shape [len(x0), N+1]
        - U_guess is all zeros (0 control input), of shape [1, N]
        """

        # X_guess has shape [len(X), N+1]
        # n.b. it is worth first getting x0 into a columnar vector shape with .reshape(-1, 1)
        x0 = x0.reshape(-1, 1)
        # n.b. np.tile(vector, (1, num_repeats)) repeats `vector` column-wise
        X_guess = np.tile(x0, (1, (self.N + 1)))

        # For the control vector, the initial guess is no control - 0s for all degrees of freedom
        U_guess = np.zeros((1, self.N))

        return X_guess, U_guess

    def get_guesses_warmstart(self, x0: np.ndarray) -> np.ndarray:
        """
        Helper function for `solve`. On each real timestep, we pass initial guesses X_guess and U_guess to the
        CasADi solver to help localise it in the search space to ensure it finds a feasible solution.

        Warm-start version:
        1. X_guess is the output of the previous real timestep's MPC solver output shifted 1 to the left
        a. X_guess(:, 0) = x0 - we have the actual state at this timestep, so no need to guess current state
        b. Then, have X_guess(:,k) ← X_prev(:,k+1) for k=1,…,N
        c. The final X_guess(:,N+1) can then be a repeat of the penultimate guess X_guess(⋅,N)
        2. U_guess follows similar logic from U_prev, except that we can use U_prev even for U_guess[:, 0]
        """

        # Use self.X_prev and self.U_prev

        # Warm start guess for X
        X_guess = np.zeros_like(self.X_prev)  # shape (4, N+1)
        X_guess[:, :-1] = self.X_prev[:, 1:]  # Shift everything left
        X_guess[:, -1] = self.X_prev[:, -1]  # Replicate last column
        # override X[:, 0] with x0
        X_guess[:, 0] = x0

        # Warm start guess for U
        U_guess = np.zeros_like(self.U_prev)  # shape (1, N)
        U_guess[:, :-1] = self.U_prev[:, 1:]  # Shift everything left
        U_guess[:, -1] = self.U_prev[:, -1]  # Replicate last column

        return X_guess, U_guess

    def get_guesses(self, x0: np.ndarray):

        # If no previous X, U available (e.g. self.X_prev is None), do basic guessing
        if self.X_prev is None:
            X_guess, U_guess = self.get_guesses_basic(x0)
        # Else do warm-start
        else:
            X_guess, U_guess = self.get_guesses_warmstart(x0)

        return X_guess, U_guess

    def solve(self, x0: np.ndarray) -> np.ndarray:
        """
        Solve the MPC problem given initial state x0.

        Args:
            x0 (np.ndarray): Current state, shape (4,)

        Returns:
            np.ndarray: Optimal control sequence, shape (1, N)

        TODO:
        - Construct initial guess [x]
        - Prepare constraint RHS
        - Call CasADi solver
        - Extract U from solver output
        """

        # === Construct initial guess ===
        X_guess, U_guess = self.get_guesses(x0)

        # As seen in the helper function `build_mpc_solver`, the X matrix (shape (:, N+1)) and U matrix (1, N) are
        # each flattened into a column vector and then concatenated X over U. In the helper function we defined this
        # for the symbolic CasADi placeholders, now we do it for the real-valued X_guess and U_guess
        initial_x_vec = X_guess.T.flatten()
        initial_u_vec = U_guess.T.flatten()
        initial_vec = np.concatenate([initial_x_vec, initial_u_vec])

        # === Call CasADi solver ===
        solution = self.solver(
            x0=initial_vec,    # solver argument x0 is a vector of all flattened X and U arrays (features x forcast ts)
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=0,    # All dynamics constraints are written as equality constraints LHS - RHS = 0
            ubg=0,
            p=x0.ravel()    # the additional required param for f(x, p) and g(x, p) is the current timestep X_init
        )
        # N.B. x0 should be a 1d vector already, but .ravel() ensures consistency.
        # E.g. a vector of (4,), a vector of (4, 1) and a vector of (1, 4) will all have the same (4, ) shape after
        # ravel

        # === Extract U from solver output ===

        # N.B. <output>.full() converts the typically casadi-type output of the solver to a NumPy array
        # N.B. <output>.flatten() then ensures this is a 1D array
        optimal_vars = solution["x"].full().flatten()

        # - The solver returns the optimal output in the same stacked column vector form as the input
        # - Therefore, the first (len(X) * (N+1) elements are the solution for X, the remainder (1 * N) for U
        len_x = len(x0)
        width_x = self.N + 1
        width_u = self.N
        idx_last_x = len_x * width_x
        X_solution = optimal_vars[:idx_last_x].reshape((width_x, len_x)).T    # TODO why this way then transpose?
        U_solution = optimal_vars[idx_last_x:].reshape((width_u, 1)).T

        # Separate out the immediate timestep's control signal, for convenience
        u_next = float(U_solution[0, 0])

        # Collate all useful outputs into a dict
        solver_outputs = {
            "X_solution": X_solution,
            "U_solution": U_solution,
            "u_next": u_next
        }

        # === Update self.X_prev and self.U_prev attributes ===
        self.X_prev = X_solution
        self.U_prev = U_solution

        return solver_outputs

    def step(self, x0: np.ndarray) -> float:
        """
        Return the first control input from the MPC solution. This method also included to have a common API to
        DRL-type controllers / fitting the general Gymnasium/MDP approach

        Args:
            x0 (np.ndarray): Current state, shape (4,)

        Returns:
            float: First control input u₀
        """
        solver_outputs = self.solve(x0)

        return solver_outputs["u_next"]
