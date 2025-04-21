import pytest
import numpy as np
import casadi as ca
import os
import sys # Import sys

# Ensure the working directory is the project root when running tests
# This allows consistent relative path access (e.g., to param files)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
# Add project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) 

os.chdir(PROJECT_ROOT)

from src.algorithms.classic.mpc_controller import MPCController

# Default parameters for testing
DEFAULT_N = 10 # Shorter horizon for faster tests
DEFAULT_DT = 0.02
# Use the actual parameter file - ensure it exists and is valid
DEFAULT_PARAM_PATH = "src/environments/inverted_pendulum_params.json"

@pytest.fixture
def mpc_controller():
    """Fixture to create an MPCController instance for testing."""
    # Check if param file exists before creating controller
    if not os.path.exists(DEFAULT_PARAM_PATH):
        pytest.skip(f"Parameter file not found at {DEFAULT_PARAM_PATH}, skipping MPC tests.")
    return MPCController(N=DEFAULT_N, dt=DEFAULT_DT, param_path=DEFAULT_PARAM_PATH)

@pytest.fixture
def sample_state():
    """Provides a sample initial state."""
    return np.array([0.1, 0.2, 0.3, 0.4]) # Example non-zero state

def test_mpc_controller_init(mpc_controller):
    """Test basic initialization of MPCController."""
    assert mpc_controller.N == DEFAULT_N
    assert mpc_controller.dt == DEFAULT_DT
    assert mpc_controller.params is not None # Check params loaded
    assert isinstance(mpc_controller.Q, ca.DM)
    assert mpc_controller.Q.shape == (4, 4)
    assert isinstance(mpc_controller.R, ca.DM)
    assert mpc_controller.R.shape == (1, 1)
    assert isinstance(mpc_controller.Q_terminal, ca.DM)
    assert mpc_controller.Q_terminal.shape == (4, 4)
    assert mpc_controller.X_prev is None # Initial state
    assert mpc_controller.U_prev is None # Initial state

def test_get_guesses_basic(mpc_controller, sample_state):
    """Test the basic guessing method."""
    X_guess, U_guess = mpc_controller.get_guesses_basic(sample_state)

    # Check shapes
    assert X_guess.shape == (4, DEFAULT_N + 1)
    assert U_guess.shape == (1, DEFAULT_N)

    # Check content
    # X_guess should be tiled sample_state
    expected_X_guess = np.tile(sample_state.reshape(-1, 1), (1, DEFAULT_N + 1))
    np.testing.assert_array_almost_equal(X_guess, expected_X_guess)
    # U_guess should be zeros
    np.testing.assert_array_almost_equal(U_guess, np.zeros((1, DEFAULT_N)))

def test_get_guesses_warmstart(mpc_controller, sample_state):
    """Test the warm-start guessing method."""
    # Manually set previous solutions for testing warm start
    mpc_controller.X_prev = np.random.rand(4, DEFAULT_N + 1) * 0.1 # Small random values
    mpc_controller.U_prev = np.random.rand(1, DEFAULT_N) * 0.5

    X_guess, U_guess = mpc_controller.get_guesses_warmstart(sample_state)

    # Check shapes
    assert X_guess.shape == (4, DEFAULT_N + 1)
    assert U_guess.shape == (1, DEFAULT_N)

    # Check content (specific warm-start logic)
    # 1. X_guess[:, 0] should be the current sample_state
    np.testing.assert_array_almost_equal(X_guess[:, 0], sample_state)
    # 2. X_guess[:, 1:] should be shifted from X_prev[:, 1:] (excluding last column of X_prev)
    np.testing.assert_array_almost_equal(X_guess[:, 1:-1], mpc_controller.X_prev[:, 2:])
    # 3. Last column of X_guess should replicate the second-to-last original column of X_prev
    np.testing.assert_array_almost_equal(X_guess[:, -1], mpc_controller.X_prev[:, -1])

    # 4. U_guess should be shifted from U_prev
    np.testing.assert_array_almost_equal(U_guess[:, :-1], mpc_controller.U_prev[:, 1:])
    # 5. Last element of U_guess should replicate the last element of U_prev
    np.testing.assert_array_almost_equal(U_guess[:, -1], mpc_controller.U_prev[:, -1])


def test_get_guesses_dispatch(mpc_controller, sample_state):
    """Test that get_guesses correctly dispatches to basic or warmstart."""
    # 1. Test initial call (should use basic)
    X_guess_basic, U_guess_basic = mpc_controller.get_guesses(sample_state)
    expected_X_basic, expected_U_basic = mpc_controller.get_guesses_basic(sample_state)
    np.testing.assert_array_almost_equal(X_guess_basic, expected_X_basic)
    np.testing.assert_array_almost_equal(U_guess_basic, expected_U_basic)

    # Manually set previous solutions to enable warm start
    mpc_controller.X_prev = np.random.rand(4, DEFAULT_N + 1)
    mpc_controller.U_prev = np.random.rand(1, DEFAULT_N)

    # 2. Test subsequent call (should use warmstart)
    X_guess_warm, U_guess_warm = mpc_controller.get_guesses(sample_state)
    expected_X_warm, expected_U_warm = mpc_controller.get_guesses_warmstart(sample_state)
    np.testing.assert_array_almost_equal(X_guess_warm, expected_X_warm)
    np.testing.assert_array_almost_equal(U_guess_warm, expected_U_warm)

def test_solve_structure(mpc_controller, sample_state):
    """Test the basic structure and types of the solve() output."""
    # We don't mock the solver here, just run it once.
    # This implicitly tests build_mpc_solver structure as well.
    # It might be slow and depends on IPOPT being available.
    # If IPOPT isn't easily available in test env, this test might need mocking.
    try:
        solver_outputs = mpc_controller.solve(sample_state)
    except Exception as e:
        pytest.fail(f"mpc_controller.solve() raised an exception: {e}")

    # Check required keys are present
    assert "X_solution" in solver_outputs
    assert "U_solution" in solver_outputs
    assert "u_next" in solver_outputs
    assert "cost" in solver_outputs
    assert "constraint_violation" in solver_outputs
    assert "solver_status" in solver_outputs

    # Check basic types and shapes if solve succeeded
    if solver_outputs["solver_status"] != "failed" and solver_outputs["X_solution"] is not None:
        assert isinstance(solver_outputs["X_solution"], np.ndarray)
        assert solver_outputs["X_solution"].shape == (4, mpc_controller.N + 1)
        assert isinstance(solver_outputs["U_solution"], np.ndarray)
        assert solver_outputs["U_solution"].shape == (1, mpc_controller.N)
        assert isinstance(solver_outputs["u_next"], float)
        assert isinstance(solver_outputs["cost"], float)
        assert isinstance(solver_outputs["constraint_violation"], float)
        assert isinstance(solver_outputs["solver_status"], str)
    elif solver_outputs["solver_status"] == "failed":
        # Check fallback values for failure case
        assert solver_outputs["X_solution"] is None
        assert solver_outputs["U_solution"] is None
        assert solver_outputs["u_next"] == 0.0
        assert solver_outputs["cost"] == np.inf
        assert solver_outputs["constraint_violation"] == np.inf

def test_step_basic(mpc_controller, sample_state):
    """Test that step() returns a float control value."""
    try:
        control_value = mpc_controller.step(sample_state)
        assert isinstance(control_value, float)
    except Exception as e:
        pytest.fail(f"mpc_controller.step() raised an exception: {e}")

def test_step_solver_failure(mpc_controller, sample_state, mocker):
    """Test that step() returns 0.0 if solve() indicates failure."""
    # Mock the internal solve method to simulate failure
    failed_output = {
        "X_solution": None, 
        "U_solution": None,
        "u_next": 0.0, 
        "cost": np.inf,
        "constraint_violation": np.inf,
        "solver_status": "failed" # Simulate failure status
    }
    mocker.patch.object(mpc_controller, 'solve', return_value=failed_output)
    
    control_value = mpc_controller.step(sample_state)
    assert control_value == 0.0
    # Check that previous solution was reset on failure
    assert mpc_controller.X_prev is None 
    assert mpc_controller.U_prev is None

# TODO: Add tests for edge cases / potential failures in solve/step
# TODO: Add tests for edge cases / potential failures 