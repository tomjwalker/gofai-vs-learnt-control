"""

"""

import casadi as ca
import numpy as np
from typing import Tuple, Dict
from pathlib import Path

from src.utils.parameters import load_inverted_pendulum_params
from src.utils.mpc_helpers import build_mpc_bounds
from src.environments.casadi_dynamics import pendulum_dynamics
# Import the cost functions
from src.algorithms.classic.mpc_costs import COST_FUNCTION_MAP, MPCCostBase 
# Import the guess strategies
from src.algorithms.classic.mpc_guesses import GUESS_STRATEGY_MAP, MPCGuessBase

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
    X_ref: ca.DM,
    cost_calculator: 'MPCCostBase'
) -> Tuple[ca.Function, ca.SX, ca.SX, ca.SX, list, list]:
    """
    Build an MPC solver for the inverted pendulum environment.
    Uses a provided cost calculator object for cost definition.

    Args:
        params: Dict of physical/environment parameters, including bounds.
        N: Prediction horizon.
        dt: Timestep.
        Q: State weighting matrix.
        R: Control weighting matrix.
        Q_terminal: Terminal cost matrix.
        X_ref: Reference state to track.
        cost_calculator: An object with calculate_stage_cost and 
                         calculate_terminal_cost methods.

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

    # === Define MPC cost function (Using cost_calculator) ===
    cost = 0
    for k in range(N):
        # Calculate stage cost using the provided calculator
        stage_cost = cost_calculator.calculate_stage_cost(
            Xk=X[:, k], 
            Uk=U[:, k], 
            Xref=X_ref, 
            Q=Q, 
            R=R
        )
        cost += stage_cost
    
    # Add terminal cost using the provided calculator
    terminal_cost = cost_calculator.calculate_terminal_cost(
        XN=X[:, N], 
        Xref=X_ref, 
        Q_terminal=Q_terminal
    )
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
                 param_path: str = "src/environments/inverted_pendulum_params.json",
                 cost_type: str = 'quadratic',
                 guess_type: str = 'warmstart'):
        """
        Initialise MPC with problem horizon, timestep, physical parameters, cost type, and guess type.
        - Loads environment parameters
        - Defines cost matrices Q, R, Q_terminal
        - Defines reference state
        - Builds solver using build_mpc_solver
        - Stores symbolic variables and bounds
        Args:
            N (int): Prediction horizon.
            dt (float): Timestep.
            param_path (str): Path to the environment parameters JSON file.
            cost_type (str): Type of cost function to use ('quadratic' or 'pendulum_swingup').
            guess_type (str): Initial guess strategy ('basic', 'warmstart', 'pendulum_heuristic').
        """

        self.dt = dt
        self.cost_type = cost_type
        self.guess_type = guess_type
        self.param_path = param_path
        print(f"MPC Initializing with param_path: {self.param_path}") # Added for debugging

        # Load environment parameters
        if not Path(self.param_path).exists():
            raise FileNotFoundError(f"Parameter file not found at: {self.param_path}")
        self.params = load_inverted_pendulum_params(self.param_path)

        # Set instance variables
        self.N = N
        self.cost_type = cost_type
        self.guess_type = guess_type

        # --- Instantiate the appropriate cost calculator --- 
        CostCalculatorClass = COST_FUNCTION_MAP.get(cost_type)
        if CostCalculatorClass is None:
            raise ValueError(f"Unknown cost_type: '{cost_type}'. Available types: {list(COST_FUNCTION_MAP.keys())}")
        self.cost_calculator: MPCCostBase = CostCalculatorClass()
        print(f"MPC using cost type: {cost_type}")

        # --- Instantiate the appropriate guess strategy --- 
        GuessStrategyClass = GUESS_STRATEGY_MAP.get(guess_type)
        if GuessStrategyClass is None:
             raise ValueError(f"Unknown guess_type: '{guess_type}'. Available types: {list(GUESS_STRATEGY_MAP.keys())}")
        self.guess_strategy: MPCGuessBase = GuessStrategyClass()
        print(f"MPC using guess type: {guess_type}")

        # --- Define cost matrices (Q, R) --- 
        # These are now weights used BY the cost calculator
        # Default weights (can be overridden by evaluate_mpc_controller.py)
        self.Q = ca.diag([1.0, 20.0, 5.0, 10.0])  
        self.R = ca.DM([50.0]) 
        self.Q_terminal = 5.0 * self.Q  

        # Define reference state (upright position)
        self.X_ref = ca.DM.zeros(4, 1)

        # --- Build solver (Pass the cost calculator) --- 
        self.solver, self.X, self.U, self.X_init, self.lbx, self.ubx = build_mpc_solver(
            self.params, self.N, self.dt, self.Q, self.R, self.Q_terminal, self.X_ref, 
            self.cost_calculator
        )

        # Attributes for warm-starting
        self.X_prev = None 
        self.U_prev = None

    def solve(self, x0: np.ndarray) -> dict:
        """
        Solve the MPC problem given initial state x0.
        Uses the configured guess_strategy.

        Args:
            x0 (np.ndarray): Current state, shape (4,)

        Returns:
            dict: Solver output dictionary including solution, cost, status etc.
        """

        # === Construct initial guess using the strategy ===
        X_guess, U_guess = self.guess_strategy.get_guess(x0=x0, N=self.N, controller=self)
        
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

            # Check if solution dictionary and essential keys exist
            if solution is not None and 'x' in solution and 'f' in solution:
                print(f"Solver stats: {self.solver.stats()}")
                print(f"Solver return_status: {self.solver.stats().get('return_status', 'N/A')}")
                print(f"Solver success: {self.solver.stats().get('success', 'N/A')}")
                
                print(f"Solver objective function value (cost): {solution['f']}")
                # Try extracting solution variables safely
                try:
                    sol_x_values = solution['x']
                    # Unpack solution
                    # Shape is (nx + nu) * N + nx = (4+1)*30+4 = 154 typically
                    # print(f"Raw solver solution vector (sol['x']) shape: {sol_x_values.shape}")
                    # Extract state trajectory (X_sol) and control trajectory (U_sol)
                    X_sol_flat = sol_x_values[0 : 4 * (self.N + 1)]
                    U_sol_flat = sol_x_values[4 * (self.N + 1) :]
                    X_sol = ca.reshape(X_sol_flat, 4, self.N + 1)
                    U_sol = ca.reshape(U_sol_flat, 1, self.N)
                    print(f"  X_sol shape: {X_sol.shape}, U_sol shape: {U_sol.shape}")
                    print(f"  X_sol (first 5):\n{X_sol[:, :min(5, self.N+1)]}")
                    print(f"  U_sol (first 5):\n{U_sol[:, :min(5, self.N)]}")
                except Exception as e:
                    print(f"  Error extracting X_sol/U_sol from solver solution: {e}")
                    X_sol = None # Indicate failure
                    U_sol = None
            else:
                print("Solver did not return a valid solution dictionary ('x' or 'f' key missing).")
                X_sol = None
                U_sol = None

            # Extract the first control action even if the full solution is dubious,
            # but only if U_sol was successfully extracted
            if U_sol is not None and U_sol.shape[1] > 0:
                u_next = U_sol[:, 0].full().flatten()[0] # Get the first control action
            else:
                print("  Could not extract U_sol, defaulting u_next to 0.")
                u_next = 0.0 # Default to zero if solution extraction failed
                
            # Calculate cost and constraint violation from solution if possible
            cost_val = solution['f'].full().item() if solution is not None and 'f' in solution else float('inf')
            constr_viol = np.linalg.norm(solution['g'].full()) if solution is not None and 'g' in solution else float('inf')
            print(f"  Calculated u_next: {u_next:.4f}, Cost: {cost_val:.4f}, ConstrViol: {constr_viol:.4e}")

        except Exception as e:
            print(f"\n!!! Exception during solver call !!!")
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
            "cost": cost_val,
            "constraint_violation": constr_viol,
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
