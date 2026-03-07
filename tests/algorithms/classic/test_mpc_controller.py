from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from src.algorithms.classic.mpc_controller import MPCController
from src.algorithms.classic.mpc_guesses import BasicGuess, WarmStartGuess

DEFAULT_HORIZON = 5
DEFAULT_DT = 0.02
DEFAULT_PARAM_PATH = "src/environments/pendulum_params.json"


@pytest.fixture
def sample_state() -> np.ndarray:
    return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)


@pytest.fixture
def mpc_controller() -> MPCController:
    return MPCController(
        N=DEFAULT_HORIZON,
        dt_controller=DEFAULT_DT,
        param_path=DEFAULT_PARAM_PATH,
        guess_type="basic",
    )


def test_mpc_controller_init(mpc_controller: MPCController) -> None:
    assert mpc_controller.N == DEFAULT_HORIZON
    assert mpc_controller.dt_controller == DEFAULT_DT
    assert mpc_controller.params is not None
    assert isinstance(mpc_controller.Q, ca.DM)
    assert mpc_controller.Q.shape == (4, 4)
    assert isinstance(mpc_controller.R, ca.DM)
    assert mpc_controller.R.shape == (1, 1)
    assert isinstance(mpc_controller.Q_terminal, ca.DM)
    assert mpc_controller.Q_terminal.shape == (4, 4)
    assert mpc_controller.X_prev is None
    assert mpc_controller.U_prev is None


def test_basic_guess_strategy_returns_tiled_state(
    mpc_controller: MPCController,
    sample_state: np.ndarray,
) -> None:
    guesser = BasicGuess()
    x_guess, u_guess = guesser.get_guess(sample_state, mpc_controller.N, mpc_controller)
    np.testing.assert_allclose(
        x_guess,
        np.tile(sample_state.reshape(-1, 1), (1, mpc_controller.N + 1)),
    )
    np.testing.assert_allclose(u_guess, np.zeros((1, mpc_controller.N)))


def test_warmstart_guess_strategy_shifts_previous_solution(
    mpc_controller: MPCController,
    sample_state: np.ndarray,
) -> None:
    mpc_controller.X_prev = np.arange(4 * (mpc_controller.N + 1), dtype=np.float64).reshape(
        4, mpc_controller.N + 1
    )
    mpc_controller.U_prev = np.arange(mpc_controller.N, dtype=np.float64).reshape(
        1, mpc_controller.N
    )

    guesser = WarmStartGuess()
    x_guess, u_guess = guesser.get_guess(sample_state, mpc_controller.N, mpc_controller)

    np.testing.assert_allclose(x_guess[:, 0], sample_state)
    np.testing.assert_allclose(x_guess[:, 1:-1], mpc_controller.X_prev[:, 2:])
    np.testing.assert_allclose(x_guess[:, -1], mpc_controller.X_prev[:, -1])
    np.testing.assert_allclose(u_guess[:, :-1], mpc_controller.U_prev[:, 1:])
    np.testing.assert_allclose(u_guess[:, -1], mpc_controller.U_prev[:, -1])


class _FakeSolver:
    def __init__(self, solution_vector: np.ndarray) -> None:
        self._solution_vector = solution_vector

    def __call__(self, **_: object) -> dict[str, ca.DM]:
        return {
            "x": ca.DM(self._solution_vector),
            "f": ca.DM([12.5]),
            "g": ca.DM.zeros(4 * (DEFAULT_HORIZON + 1), 1),
        }

    def stats(self) -> dict[str, str]:
        return {"return_status": "Solve_Succeeded"}


def test_solve_unpacks_solution_and_updates_warm_start_state(
    mpc_controller: MPCController,
    sample_state: np.ndarray,
) -> None:
    x_solution = np.tile(sample_state.reshape(-1, 1), (1, mpc_controller.N + 1))
    u_solution = np.linspace(0.0, 0.4, mpc_controller.N, dtype=np.float64).reshape(
        1, mpc_controller.N
    )
    solution_vector = np.concatenate([x_solution.T.flatten(), u_solution.T.flatten()])
    mpc_controller.solver = _FakeSolver(solution_vector)

    outputs = mpc_controller.solve(sample_state)

    np.testing.assert_allclose(outputs["X_solution"], x_solution)
    np.testing.assert_allclose(outputs["U_solution"], u_solution)
    assert outputs["u_next"] == pytest.approx(float(u_solution[0, 0]))
    assert outputs["cost"] == pytest.approx(12.5)
    assert outputs["constraint_violation"] == pytest.approx(0.0)
    assert outputs["solver_status"] == "Solve_Succeeded"
    np.testing.assert_allclose(mpc_controller.X_prev, x_solution)
    np.testing.assert_allclose(mpc_controller.U_prev, u_solution)


def test_step_returns_zero_on_solver_failure(
    mpc_controller: MPCController,
    sample_state: np.ndarray,
) -> None:
    def _raise_solver_error(**_: object) -> dict[str, ca.DM]:
        raise RuntimeError("solver failure")

    mpc_controller.solver = _raise_solver_error  # type: ignore[assignment]
    mpc_controller.X_prev = np.ones((4, mpc_controller.N + 1), dtype=np.float64)
    mpc_controller.U_prev = np.ones((1, mpc_controller.N), dtype=np.float64)

    action = mpc_controller.step(sample_state)

    assert action == 0.0
    assert mpc_controller.X_prev is None
    assert mpc_controller.U_prev is None
