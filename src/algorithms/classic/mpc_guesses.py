# src/algorithms/classic/mpc_guesses.py
import numpy as np
import casadi as ca # May not be needed here, but good practice

class MPCGuessBase:
    """Base class for MPC initial guess strategies."""
    def get_guess(self, x0: np.ndarray, N: int, controller: 'MPCController') -> tuple[np.ndarray, np.ndarray]:
        """
        Generate initial guesses for the state (X) and control (U) trajectories.
        
        Args:
            x0 (np.ndarray): Current state observation.
            N (int): MPC prediction horizon.
            controller (MPCController): Reference to the controller instance 
                                        (needed for warm starts, params etc.).
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X_guess (state_dim x N+1), U_guess (control_dim x N)
        """
        raise NotImplementedError

class BasicGuess(MPCGuessBase):
    """Basic guess: repeat initial state, zero control."""
    def get_guess(self, x0: np.ndarray, N: int, controller: 'MPCController') -> tuple[np.ndarray, np.ndarray]:
        state_dim = len(x0)
        control_dim = controller.U.shape[0] # Get control dim from symbolic var
        
        x0_col = x0.reshape(-1, 1)
        X_guess = np.tile(x0_col, (1, N + 1))
        U_guess = np.zeros((control_dim, N))
        return X_guess, U_guess

class WarmStartGuess(MPCGuessBase):
    """Warm start: shift previous solution if available, otherwise use basic."""
    def get_guess(self, x0: np.ndarray, N: int, controller: 'MPCController') -> tuple[np.ndarray, np.ndarray]:
        state_dim = len(x0)
        control_dim = controller.U.shape[0]
        
        if controller.X_prev is None or controller.U_prev is None:
            # Fallback to basic guess if no previous solution exists
            basic_guesser = BasicGuess()
            return basic_guesser.get_guess(x0, N, controller)
        else:
            # Warm start logic (adapted from previous controller method)
            X_guess = np.zeros_like(controller.X_prev)
            X_guess[:, :-1] = controller.X_prev[:, 1:]
            X_guess[:, -1] = controller.X_prev[:, -1]
            X_guess[:, 0] = x0 # Override first state with current state
            
            U_guess = np.zeros_like(controller.U_prev)
            U_guess[:, :-1] = controller.U_prev[:, 1:]
            U_guess[:, -1] = controller.U_prev[:, -1]
            
            return X_guess, U_guess

class PendulumSwingupHeuristicGuess(MPCGuessBase):
    """Heuristic guess for pendulum swing-up: bang-bang control."""
    def get_guess(self, x0: np.ndarray, N: int, controller: 'MPCController') -> tuple[np.ndarray, np.ndarray]:
        state_dim = len(x0)
        control_dim = controller.U.shape[0]
        dt = controller.dt
        
        # Basic guess for state trajectory (can be improved with simulation)
        basic_guesser = BasicGuess()
        X_guess, _ = basic_guesser.get_guess(x0, N, controller)
        
        # Heuristic control: bang-bang
        # Parameters for heuristic (could be configurable)
        t_bang1 = 0.3 # Duration of first push (seconds)
        t_bang2 = 0.3 # Duration of second push (seconds)
        try:
            # Use control limits from params if available
            u_max = controller.params['control_bounds'][0][1] 
        except (KeyError, IndexError):
            print("Warning (HeuristicGuess): Control bounds not found in params. Using default +/- 3.")
            u_max = 3.0
            
        n_bang1 = int(t_bang1 / dt)
        n_bang2 = int(t_bang2 / dt)
        
        U_guess = np.zeros((control_dim, N))
        
        # Apply positive force initially
        force_sign = 1.0
        if n_bang1 > 0:
            U_guess[0, :min(n_bang1, N)] = force_sign * u_max
            
        # Apply negative force after first bang
        if n_bang1 < N and n_bang2 > 0:
            start_idx = n_bang1
            end_idx = min(n_bang1 + n_bang2, N)
            U_guess[0, start_idx:end_idx] = -force_sign * u_max
            
        # Remaining control is zero (already initialized)
        print(f"Generated heuristic swing-up guess (U starts with {U_guess[0, 0]:.1f})")
        return X_guess, U_guess

# Dictionary to map guess type strings to classes
GUESS_STRATEGY_MAP = {
    'basic': BasicGuess,
    'warmstart': WarmStartGuess,
    'pendulum_heuristic': PendulumSwingupHeuristicGuess
} 