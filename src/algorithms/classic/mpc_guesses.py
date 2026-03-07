from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.algorithms.classic.mpc_controller import MPCController


class MPCGuessBase:
    """Base class for MPC initial guess strategies."""

    def get_guess(
        self,
        x0: np.ndarray,
        N: int,
        controller: MPCController,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class BasicGuess(MPCGuessBase):
    """Basic guess: repeat the initial state and use zero control."""

    def get_guess(
        self,
        x0: np.ndarray,
        N: int,
        controller: MPCController,
    ) -> tuple[np.ndarray, np.ndarray]:
        control_dim = int(controller.U.shape[0])
        x0_column = x0.reshape(-1, 1)
        x_guess = np.tile(x0_column, (1, N + 1))
        u_guess = np.zeros((control_dim, N))
        return x_guess, u_guess


class WarmStartGuess(MPCGuessBase):
    """Warm start by shifting the previous solution, or fall back to the basic guess."""

    def get_guess(
        self,
        x0: np.ndarray,
        N: int,
        controller: MPCController,
    ) -> tuple[np.ndarray, np.ndarray]:
        if controller.X_prev is None or controller.U_prev is None:
            return BasicGuess().get_guess(x0, N, controller)

        x_guess = np.zeros_like(controller.X_prev)
        x_guess[:, :-1] = controller.X_prev[:, 1:]
        x_guess[:, -1] = controller.X_prev[:, -1]
        x_guess[:, 0] = x0

        u_guess = np.zeros_like(controller.U_prev)
        u_guess[:, :-1] = controller.U_prev[:, 1:]
        u_guess[:, -1] = controller.U_prev[:, -1]
        return x_guess, u_guess


class PendulumSwingupHeuristicGuess(MPCGuessBase):
    """Bang-bang heuristic to help the pendulum start swinging up."""

    def get_guess(
        self,
        x0: np.ndarray,
        N: int,
        controller: MPCController,
    ) -> tuple[np.ndarray, np.ndarray]:
        control_dim = int(controller.U.shape[0])
        x_guess, _ = BasicGuess().get_guess(x0, N, controller)

        t_bang1 = 0.3
        t_bang2 = 0.3
        try:
            u_max = float(controller.params["control_bounds"][0][1])
        except (KeyError, IndexError, TypeError, ValueError):
            u_max = 3.0

        n_bang1 = int(t_bang1 / controller.dt_controller)
        n_bang2 = int(t_bang2 / controller.dt_controller)
        u_guess = np.zeros((control_dim, N))
        if n_bang1 > 0:
            u_guess[0, : min(n_bang1, N)] = u_max
        if n_bang1 < N and n_bang2 > 0:
            start_idx = n_bang1
            end_idx = min(n_bang1 + n_bang2, N)
            u_guess[0, start_idx:end_idx] = -u_max
        return x_guess, u_guess


class HybridGuess(MPCGuessBase):
    """Heuristic when the pendulum is down, warm-start otherwise."""

    def get_guess(
        self,
        x0: np.ndarray,
        N: int,
        controller: MPCController,
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.cos(x0[1]) <= -0.7:
            return PendulumSwingupHeuristicGuess().get_guess(x0, N, controller)
        return WarmStartGuess().get_guess(x0, N, controller)


GUESS_STRATEGY_MAP = {
    "basic": BasicGuess,
    "warmstart": WarmStartGuess,
    "pendulum_heuristic": PendulumSwingupHeuristicGuess,
    "hybrid": HybridGuess,
}
