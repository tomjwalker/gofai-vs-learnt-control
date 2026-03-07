from __future__ import annotations

from collections.abc import Callable
from typing import Any

import casadi as ca

CasadiVector = ca.SX | ca.DM


def rk4_step(
    ode_func: Callable[[CasadiVector, CasadiVector, Any], CasadiVector],
    x: CasadiVector,
    u: CasadiVector,
    dt: float,
    params: Any,
) -> CasadiVector:
    """Perform one RK4 integration step for an ODE of the form `f(x, u, params)`."""
    k1 = ode_func(x, u, params)
    k2 = ode_func(x + dt / 2 * k1, u, params)
    k3 = ode_func(x + dt / 2 * k2, u, params)
    k4 = ode_func(x + dt * k3, u, params)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
