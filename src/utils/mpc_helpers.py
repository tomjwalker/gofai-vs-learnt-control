from __future__ import annotations


def build_mpc_bounds(
    state_bounds: list[tuple[float, float]],
    control_bounds: list[tuple[float, float]],
    N: int,
) -> tuple[list[float], list[float]]:
    """Build lower/upper bound vectors for stacked MPC state and control variables."""
    lbx: list[float] = []
    ubx: list[float] = []

    for _ in range(N + 1):
        for lower_bound, upper_bound in state_bounds:
            lbx.append(lower_bound)
            ubx.append(upper_bound)

    for _ in range(N):
        for lower_bound, upper_bound in control_bounds:
            lbx.append(lower_bound)
            ubx.append(upper_bound)

    return lbx, ubx
