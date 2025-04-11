"""
mpc_helpers.py

Contains utility functions for building standard MPC components such as bound vectors
based on state and control constraints over a fixed prediction horizon.
"""

from typing import List, Tuple
import numpy as np

def build_mpc_bounds(
    state_bounds: List[Tuple[float, float]],
    control_bounds: List[Tuple[float, float]],
    N: int
) -> Tuple[List[float], List[float]]:
    """
    Build the lbx and ubx bound vectors for the full set of MPC decision variables
    (states and controls) over the prediction horizon.

    Args:
        state_bounds: List of (min, max) tuples for each state dimension.
        control_bounds: List of (min, max) tuples for each control dimension.
        N: Prediction horizon.

    Returns:
        lbx: Flattened list of lower bounds for all decision variables.
        ubx: Flattened list of upper bounds for all decision variables.
    """
    lbx = []
    ubx = []

    # States: N+1 time steps
    for _ in range(N + 1):
        for lo, hi in state_bounds:
            lbx.append(lo)
            ubx.append(hi)

    # Controls: N time steps
    for _ in range(N):
        for lo, hi in control_bounds:
            lbx.append(lo)
            ubx.append(hi)

    return lbx, ubx
