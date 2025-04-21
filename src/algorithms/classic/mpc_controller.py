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
# os.chdir(os.path.join(os.path.dirname(__file__), "../..")) # Commented out, paths should be relative to project root
# print(os.getcwd())

def build_mpc_bounds(state_bounds_obs: list, control_bounds: list, N: int, 
                       joint_bounds: dict = None) -> Tuple[list, list]:
    """
    Construct lists of lower and upper bounds for the stacked state (X) 
    and control (U) trajectory variables used in the NLP.

    Uses physical joint bounds where available, otherwise observation bounds.

    Args:
        state_bounds_obs: Bounds from the environment's observation space [[min, max], ...].
        control_bounds: Bounds for the control inputs [[min, max], ...].
        N: MPC prediction horizon.
        joint_bounds: Dictionary of physical joint limits {state_name: [min, max]}.

    Returns:
        Tuple[list, list]: lbx, ubx lists for the NLP solver.
    """
    lbx, ubx = [], []
    joint_bounds = joint_bounds or {}
    
    # State bounds (X trajectory, shape 4 x N+1)
    num_states = len(state_bounds_obs)
    state_names = ['cart_pos', 'pole_angle', 'cart_vel', 'pole_ang_vel'] # Assuming this order
    
    for k in range(N + 1):  # Iterate over timesteps
        for i in range(num_states):  # Iterate over states (x, theta, x_dot, theta_dot)
            state_name = state_names[i]
            # Use specific joint bound if available, otherwise use observation space bound
            if state_name in joint_bounds:
                lb, ub = joint_bounds[state_name]
            else:
                lb, ub = state_bounds_obs[i]
                
            # Use -inf/inf from numpy if needed, converting json's "Infinity" string
            lb = -np.inf if lb == -float('inf') else lb
            ub = np.inf if ub == float('inf') else ub
            
            lbx.append(lb)
            ubx.append(ub)
            
    # Control bounds (U trajectory, shape 1 x N)
    num_controls = len(control_bounds)
    for _ in range(N): # Iterate over timesteps
        for i in range(num_controls):
            lb, ub = control_bounds[i]
            lbx.append(lb)
            ubx.append(ub)

    return lbx, ubx


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
        X_next = pendulum_dynamics(X[:, k], U[:, k], dt, params, integration_method='rk4')

        # Enforce that the predicted state equals the decision variable for the next timestep
        dynamics_constraints.append(X[:, k + 1] - X_next)

    # Concatenate all dynamics constraints into a single vector
    dynamics_constraints = ca.vertcat(initial_state_constraint, *dynamics_constraints)

    # === Build bounds on state and control vectors ===
    # Use the updated build_mpc_bounds which prioritizes physical joint limits
    lbx, ubx = build_mpc_bounds(
        params["state_bounds_obs"], # Pass obs bounds
        params["control_bounds"], 
        N, 
        params.get("joint_bounds") # Pass physical joint bounds if they exist
    )

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
        "ipopt.print_level": 0,  # Reduced verbosity
        "ipopt.tol": 1e-6,  # Tighter tolerance
        "ipopt.max_iter": 200,  # More iterations allowed
        "ipopt.linear_solver": "mumps",  # Use MUMPS which comes with CasADi
        "ipopt.hessian_approximation": "limited-memory",  # Better for large problems
        "print_time": True,
        "ipopt.warm_start_init_point": "yes",  # Enable warm start
        "ipopt.mu_strategy": "adaptive",  # Better convergence
        "ipopt.ma57_automatic_scaling": "yes",  # Better numerical stability
        "ipopt.sb": "yes",  # Suppress banner
        # "ipopt.max_cpu_time": 1.0  # Remove or increase time limit for now
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
                 N: int = 30,
                 dt: float = 0.02,
                 param_path: str = "src/environments/inverted_pendulum_params.json"):
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
        # Reduced state weights, increased velocity weights slightly
        self.Q = ca.diag([1.0, 20.0, 5.0, 10.0])  
        # Significantly increased control weight
        self.R = ca.DM([50.0]) 
        # Keep terminal cost proportional to Q
        self.Q_terminal = 5.0 * self.Q  

        # Define reference state (upright position)
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
        initial_x_vec = X_guess.T.flatten()
        initial_u_vec = U_guess.T.flatten()
        initial_vec = np.concatenate([initial_x_vec, initial_u_vec])

        # === Call CasADi solver ===
        try:
            solution = self.solver(
                x0=initial_vec, 
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=0, 
                ubg=0,
                p=x0.ravel() 
            )
            
            # Simplified diagnostics after successful solve
            cost = float(solution["f"]) if "f" in solution else np.nan
            constraint_violation = float(np.max(np.abs(solution['g']))) if "g" in solution else np.nan
            print(f"  MPC Solve: Cost={cost:.2f}, ConstrViol={constraint_violation:.2e}", end='') # Print on one line

        except Exception as e:
            print(f"\nMPC Solver failed: {e}")
            # Return a default safe action (e.g., zero) or re-raise
            # For now, return a dictionary indicating failure
            return {
                "X_solution": None, # Indicate failure
                "U_solution": None,
                "u_next": 0.0, # Default safe action
                "cost": np.inf,
                "constraint_violation": np.inf,
                "solver_status": "failed"
            }

        # === Extract U from solver output ===
        optimal_vars = solution["x"].full().flatten()
        len_x = len(x0)
        width_x = self.N + 1
        width_u = self.N
        idx_last_x = len_x * width_x
        X_solution = optimal_vars[:idx_last_x].reshape((width_x, len_x)).T
        U_solution = optimal_vars[idx_last_x:].reshape((width_u, 1)).T

        # Separate out the immediate timestep's control signal, for convenience
        u_next = float(U_solution[0, 0])

        # Print concise diagnostics about the solution
        print(f", u_next={u_next:.3f}, Max|U|={np.max(np.abs(U_solution)):.2f}") 

        # Collate all useful outputs into a dict
        solver_outputs = {
            "X_solution": X_solution,
            "U_solution": U_solution,
            "u_next": u_next,
            "cost": cost,
            "constraint_violation": constraint_violation,
            "solver_status": self.solver.stats().get('return_status', 'unknown')
        }

        # === Update self.X_prev and self.U_prev attributes ===
        self.X_prev = X_solution
        self.U_prev = U_solution

        return solver_outputs

    def step(self, x0: np.ndarray) -> float:
        """
        Return the first control input from the MPC solution. Handles solver failures.

        Args:
            x0 (np.ndarray): Current state, shape (4,)

        Returns:
            float: First control input u₀
        """
        solver_outputs = self.solve(x0)
        # Check if solve failed and return safe action if necessary
        if solver_outputs.get("solver_status") == "failed" or solver_outputs.get("X_solution") is None:
            print("Warning: MPC step using default safe action due to solver failure.")
            # Reset previous solution to avoid bad warm-start
            self.X_prev = None
            self.U_prev = None
            return 0.0 # Return zero action
        
        return solver_outputs["u_next"]
