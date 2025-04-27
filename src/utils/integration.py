# src/utils/integration.py
import casadi as ca
from typing import Callable, Any # Use Any for params flexibility

def rk4_step(ode_func: Callable[[ca.SX, ca.SX, Any], ca.SX], 
             x: ca.SX, 
             u: ca.SX, 
             dt: float, 
             params: Any) -> ca.SX:
    """
    Performs one RK4 integration step for a given CasADi ODE function.

    Args:
        ode_func: A CasADi function with signature f(x, u, params) -> x_dot.
        x: Current state vector (CasADi SX or DM).
        u: Current control input vector (CasADi SX or DM).
        dt: Time step duration.
        params: Parameter vector/structure expected by ode_func (CasADi SX or DM).

    Returns:
        The next state vector (CasADi SX or DM).
    """
    k1 = ode_func(x, u, params)
    k2 = ode_func(x + dt/2 * k1, u, params)
    k3 = ode_func(x + dt/2 * k2, u, params)
    k4 = ode_func(x + dt * k3, u, params)
    x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

# Potential future addition: Euler step function
# def euler_step(ode_func: Callable[[ca.SX, ca.SX, Any], ca.SX], 
#                x: ca.SX, 
#                u: ca.SX, 
#                dt: float, 
#                params: Any) -> ca.SX:
#     """Performs one Euler integration step."""
#     x_dot = ode_func(x, u, params)
#     x_next = x + dt * x_dot
#     return x_next 