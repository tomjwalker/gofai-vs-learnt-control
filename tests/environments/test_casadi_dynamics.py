from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from src.environments.wrappers import (
    InvertedDoublePendulumComparisonWrapper,
    casadi_double_pendulum_to_gym_raw,
    gym_raw_double_pendulum_to_casadi,
)
from src.utils.trajectory_comparison import (
    compute_trajectory_metrics,
    dump_comparison_artifacts,
    format_metrics_summary,
    rollout_double_pendulum_casadi,
    rollout_double_pendulum_gym,
    rollout_single_pendulum_casadi,
    rollout_single_pendulum_gym,
)


def _assert_metrics_below_threshold(
    *,
    tmp_path: Path,
    stem: str,
    reference,
    candidate,
    max_threshold: float,
    rmse_threshold: float,
) -> None:
    metrics = compute_trajectory_metrics(reference.states, candidate.states)
    if metrics.overall_max_abs_error <= max_threshold and metrics.overall_rmse <= rmse_threshold:
        return

    artifacts = dump_comparison_artifacts(tmp_path, stem, reference, candidate, metrics)
    summary = format_metrics_summary(metrics, reference.state_labels)
    pytest.fail(
        f"{stem} exceeded thresholds max<={max_threshold} and rmse<={rmse_threshold}. "
        f"Observed {summary}. Artifacts: {artifacts.csv_path}, {artifacts.json_path}"
    )


def test_double_pendulum_state_conversion_round_trip() -> None:
    casadi_state = np.array([0.0, 0.1, -0.1, 0.0, 0.2, -0.3], dtype=np.float64)
    raw_state = casadi_double_pendulum_to_gym_raw(casadi_state)
    round_trip = gym_raw_double_pendulum_to_casadi(raw_state)
    np.testing.assert_allclose(raw_state, np.array([0.0, 0.1, -0.2, 0.0, 0.2, -0.5]))
    np.testing.assert_allclose(round_trip, casadi_state)


def test_double_pendulum_wrapper_reset_matches_raw_mujoco_state() -> None:
    initial_state = np.array([0.0, 0.1, -0.1, 0.0, 0.2, -0.3], dtype=np.float64)
    expected_raw_state = casadi_double_pendulum_to_gym_raw(initial_state)
    env = InvertedDoublePendulumComparisonWrapper(
        gym.make("InvertedDoublePendulum-v5"),
        return_state_convention="gym_raw",
    )
    try:
        observation, _ = env.reset(
            initial_state=initial_state,
            initial_state_convention="casadi_absolute",
        )
        raw_state = np.concatenate([env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()])
        native_observation = env.unwrapped._get_obs()
    finally:
        env.close()

    np.testing.assert_allclose(observation, expected_raw_state)
    np.testing.assert_allclose(raw_state, expected_raw_state)
    np.testing.assert_allclose(native_observation[0], expected_raw_state[0])
    np.testing.assert_allclose(native_observation[1], np.sin(expected_raw_state[1]))
    np.testing.assert_allclose(native_observation[2], np.sin(expected_raw_state[2]))
    np.testing.assert_allclose(native_observation[3], np.cos(expected_raw_state[1]))
    np.testing.assert_allclose(native_observation[4], np.cos(expected_raw_state[2]))
    np.testing.assert_allclose(native_observation[5:8], expected_raw_state[3:6])


def test_double_pendulum_environment_reports_actual_dt() -> None:
    env = InvertedDoublePendulumComparisonWrapper(gym.make("InvertedDoublePendulum-v5"))
    try:
        expected_dt = float(env.unwrapped.model.opt.timestep * env.unwrapped.frame_skip)
        assert env.unwrapped.dt == pytest.approx(expected_dt)
        assert env.unwrapped.frame_skip == 5
    finally:
        env.close()


def test_single_pendulum_rollout_matches_gym_baseline(tmp_path: Path) -> None:
    initial_state = np.array([0.0, 0.1, 0.0, 0.0], dtype=np.float64)
    control_sequence = np.zeros(25, dtype=np.float64)
    gym_rollout = rollout_single_pendulum_gym(
        initial_state=initial_state,
        control_sequence=control_sequence,
    )
    casadi_rollout = rollout_single_pendulum_casadi(
        initial_state=initial_state,
        control_sequence=control_sequence,
        dt=gym_rollout.dt,
    )
    _assert_metrics_below_threshold(
        tmp_path=tmp_path,
        stem="single_pendulum_free",
        reference=gym_rollout,
        candidate=casadi_rollout,
        max_threshold=0.08,
        rmse_threshold=0.03,
    )


def test_double_pendulum_free_rollout_matches_gym_on_short_horizon(tmp_path: Path) -> None:
    initial_state = np.array([0.0, 0.1, -0.1, 0.0, 0.0, 0.0], dtype=np.float64)
    control_sequence = np.zeros(10, dtype=np.float64)
    gym_rollout = rollout_double_pendulum_gym(
        initial_state=initial_state,
        control_sequence=control_sequence,
    )
    casadi_rollout = rollout_double_pendulum_casadi(
        initial_state=initial_state,
        control_sequence=control_sequence,
        dt=gym_rollout.dt,
    )
    _assert_metrics_below_threshold(
        tmp_path=tmp_path,
        stem="double_pendulum_free_short_horizon",
        reference=gym_rollout,
        candidate=casadi_rollout,
        max_threshold=0.09,
        rmse_threshold=0.035,
    )


def test_double_pendulum_controlled_rollout_matches_gym_on_short_horizon(tmp_path: Path) -> None:
    initial_state = np.array([0.0, 0.1, -0.1, 0.0, 0.0, 0.0], dtype=np.float64)
    control_sequence = np.concatenate(
        [np.full(5, -0.1, dtype=np.float64), np.full(5, 0.1, dtype=np.float64)]
    )
    gym_rollout = rollout_double_pendulum_gym(
        initial_state=initial_state,
        control_sequence=control_sequence,
    )
    casadi_rollout = rollout_double_pendulum_casadi(
        initial_state=initial_state,
        control_sequence=control_sequence,
        dt=gym_rollout.dt,
    )
    _assert_metrics_below_threshold(
        tmp_path=tmp_path,
        stem="double_pendulum_controlled_short_horizon",
        reference=gym_rollout,
        candidate=casadi_rollout,
        max_threshold=0.18,
        rmse_threshold=0.05,
    )
