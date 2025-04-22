# src/algorithms/classic/mpc_costs.py
import casadi as ca

class MPCCostBase:
    """Base class for MPC cost functions."""
    def calculate_stage_cost(self, Xk, Uk, Xref, Q, R):
        """
        Calculate the cost for a single stage (timestep k).
        
        Args:
            Xk: Symbolic state vector at step k.
            Uk: Symbolic control vector at step k.
            Xref: Reference state vector.
            Q: State weighting matrix.
            R: Control weighting matrix.
            
        Returns:
            Symbolic stage cost expression.
        """
        raise NotImplementedError

    def calculate_terminal_cost(self, XN, Xref, Q_terminal):
        """
        Calculate the terminal cost at the end of the horizon (step N).
        
        Args:
            XN: Symbolic state vector at step N.
            Xref: Reference state vector.
            Q_terminal: Terminal state weighting matrix.
            
        Returns:
            Symbolic terminal cost expression.
        """
        raise NotImplementedError

class QuadraticCost(MPCCostBase):
    """Standard quadratic MPC cost."""
    def calculate_stage_cost(self, Xk, Uk, Xref, Q, R):
        state_error = Xk - Xref
        # Assuming Uk is shape (control_dim, 1)
        control = Uk 
        # Ensure matrix dimensions match for mtimes
        # Cost = error^T * Q * error + control^T * R * control
        stage_cost = ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([control.T, R, control])
        return stage_cost

    def calculate_terminal_cost(self, XN, Xref, Q_terminal):
        terminal_error = XN - Xref
        terminal_cost = ca.mtimes([terminal_error.T, Q_terminal, terminal_error])
        return terminal_cost

class PendulumSwingupCost(MPCCostBase):
    """Cost function suitable for pendulum swing-up.
       Uses quadratic cost for x, x_dot, theta_dot and (1-cos(theta)) for angle.
    """
    def calculate_stage_cost(self, Xk, Uk, Xref, Q, R):
        # Xk = [x, theta, x_dot, theta_dot]
        # Xref is likely [0, 0, 0, 0]
        state_error = Xk - Xref 
        control = Uk
        
        # Quadratic cost for x, x_dot, theta_dot (indices 0, 2, 3)
        # Assumes Q is diagonal for easy access Q[i,i]
        cost_x = state_error[0]**2 * Q[0, 0]
        cost_x_dot = state_error[2]**2 * Q[2, 2]
        cost_theta_dot = state_error[3]**2 * Q[3, 3]
        
        # Cosine-based cost for theta (index 1)
        # Cost = Q_theta * (1 - cos(theta_k)) -> encourages theta=0
        cost_theta = (1 - ca.cos(Xk[1])) * Q[1, 1] 
        
        # Control cost
        cost_control = ca.mtimes([control.T, R, control])
        
        stage_cost = cost_x + cost_theta + cost_x_dot + cost_theta_dot + cost_control
        return stage_cost

    def calculate_terminal_cost(self, XN, Xref, Q_terminal):
        terminal_error = XN - Xref
        # Assumes Q_terminal is diagonal
        term_cost_x = terminal_error[0]**2 * Q_terminal[0, 0]
        term_cost_x_dot = terminal_error[2]**2 * Q_terminal[2, 2]
        term_cost_theta_dot = terminal_error[3]**2 * Q_terminal[3, 3]
        term_cost_theta = (1 - ca.cos(XN[1])) * Q_terminal[1, 1]
        
        terminal_cost = term_cost_x + term_cost_theta + term_cost_x_dot + term_cost_theta_dot
        return terminal_cost

# --- New Cost Function --- 
class PendulumSwingupAtan2Cost(MPCCostBase):
    """Cost function for swing-up using quadratic cost on atan2 angle error."""
    def calculate_stage_cost(self, Xk, Uk, Xref, Q, R):
        # Xk = [x, theta, x_dot, theta_dot]
        state_error = Xk - Xref 
        control = Uk
        
        # Quadratic cost for x, x_dot, theta_dot (indices 0, 2, 3)
        cost_x = state_error[0]**2 * Q[0, 0]
        cost_x_dot = state_error[2]**2 * Q[2, 2]
        cost_theta_dot = state_error[3]**2 * Q[3, 3]
        
        # atan2-based cost for theta (index 1)
        # error = atan2(sin(theta - theta_ref), cos(theta - theta_ref))
        # Since Xref[1] is 0, this simplifies to atan2(sin(theta), cos(theta))
        angle_error = ca.atan2(ca.sin(Xk[1]), ca.cos(Xk[1])) 
        cost_theta = angle_error**2 * Q[1, 1] 
        
        # Control cost
        cost_control = ca.mtimes([control.T, R, control])
        
        stage_cost = cost_x + cost_theta + cost_x_dot + cost_theta_dot + cost_control
        return stage_cost

    def calculate_terminal_cost(self, XN, Xref, Q_terminal):
        terminal_error = XN - Xref
        
        term_cost_x = terminal_error[0]**2 * Q_terminal[0, 0]
        term_cost_x_dot = terminal_error[2]**2 * Q_terminal[2, 2]
        term_cost_theta_dot = terminal_error[3]**2 * Q_terminal[3, 3]
        
        # Use atan2 for terminal angle cost as well
        terminal_angle_error = ca.atan2(ca.sin(XN[1]), ca.cos(XN[1]))
        term_cost_theta = terminal_angle_error**2 * Q_terminal[1, 1]
        
        terminal_cost = term_cost_x + term_cost_theta + term_cost_x_dot + term_cost_theta_dot
        return terminal_cost
# -------------------------

# Dictionary to map cost type strings to classes
COST_FUNCTION_MAP = {
    'quadratic': QuadraticCost,
    'pendulum_swingup': PendulumSwingupCost,
    'pendulum_atan2': PendulumSwingupAtan2Cost
} 