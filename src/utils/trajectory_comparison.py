from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

import casadi as ca
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from src.environments.double_pendulum_dynamics import double_pendulum_dynamics_step
from src.environments.pendulum_dynamics import pendulum_dynamics
from src.environments.wrappers import (
    DoublePendulumStateConvention,
    InvertedDoublePendulumComparisonWrapper,
    InvertedPendulumComparisonWrapper,
)
from src.utils.parameters import load_double_pendulum_params, load_pendulum_params

FloatArray: TypeAlias = npt.NDArray[np.float64]
_SINGLE_STATE_LABELS = ("x", "theta", "x_dot", "theta_dot")
_DOUBLE_STATE_LABELS = ("x", "theta1", "theta2", "x_dot", "theta1_dot", "theta2_dot")


@dataclass(frozen=True)
class RolloutResult:
    states: FloatArray
    dt: float
    state_labels: tuple[str, ...]
    state_convention: str
    frame_skip: int
    truncated: bool


@dataclass(frozen=True)
class TrajectoryMetrics:
    max_abs_error: FloatArray
    mean_abs_error: FloatArray
    rmse: FloatArray
    overall_max_abs_error: float
    overall_rmse: float


@dataclass(frozen=True)
class ComparisonArtifacts:
    csv_path: Path
    json_path: Path


def _coerce_vector(state: npt.ArrayLike, *, name: str) -> FloatArray:
    vector = np.asarray(state, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape {vector.shape}.")
    return vector


def _coerce_control_sequence(control_sequence: npt.ArrayLike) -> FloatArray:
    controls = np.asarray(control_sequence, dtype=np.float64)
    if controls.ndim != 1:
        raise ValueError(f"control_sequence must be 1D, got shape {controls.shape}.")
    return controls


def align_rollouts(
    reference: RolloutResult,
    candidate: RolloutResult,
) -> tuple[FloatArray, FloatArray]:
    aligned_length = min(reference.states.shape[0], candidate.states.shape[0])
    return reference.states[:aligned_length], candidate.states[:aligned_length]


def compute_trajectory_metrics(
    reference_states: npt.ArrayLike,
    candidate_states: npt.ArrayLike,
) -> TrajectoryMetrics:
    reference = np.asarray(reference_states, dtype=np.float64)
    candidate = np.asarray(candidate_states, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Trajectory shapes must match exactly, got {reference.shape} and {candidate.shape}."
        )

    error = candidate - reference
    abs_error = np.abs(error)
    squared_error = error**2
    max_abs_error = np.max(abs_error, axis=0)
    mean_abs_error = np.mean(abs_error, axis=0)
    rmse = np.sqrt(np.mean(squared_error, axis=0))
    return TrajectoryMetrics(
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        rmse=rmse,
        overall_max_abs_error=float(np.max(max_abs_error)),
        overall_rmse=float(np.sqrt(np.mean(squared_error))),
    )


def format_metrics_summary(
    metrics: TrajectoryMetrics,
    state_labels: tuple[str, ...],
) -> str:
    per_state_summary = ", ".join(
        f"{label}: max={max_error:.6f}, rmse={rmse_error:.6f}"
        for label, max_error, rmse_error in zip(
            state_labels,
            metrics.max_abs_error,
            metrics.rmse,
            strict=True,
        )
    )
    return (
        f"overall_max_abs_error={metrics.overall_max_abs_error:.6f}, "
        f"overall_rmse={metrics.overall_rmse:.6f}, per_state=[{per_state_summary}]"
    )


def dump_comparison_artifacts(
    output_dir: str | Path,
    stem: str,
    reference: RolloutResult,
    candidate: RolloutResult,
    metrics: TrajectoryMetrics,
) -> ComparisonArtifacts:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"{stem}.csv"
    json_path = destination / f"{stem}.json"

    aligned_reference, aligned_candidate = align_rollouts(reference, candidate)
    header = [
        "step",
        *[f"ref_{label}" for label in reference.state_labels],
        *[f"cand_{label}" for label in candidate.state_labels],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, (ref_row, cand_row) in enumerate(
            zip(aligned_reference, aligned_candidate, strict=True)
        ):
            writer.writerow([index, *ref_row.tolist(), *cand_row.tolist()])

    payload = {
        "reference": {
            "dt": reference.dt,
            "state_labels": list(reference.state_labels),
            "state_convention": reference.state_convention,
            "frame_skip": reference.frame_skip,
            "truncated": reference.truncated,
        },
        "candidate": {
            "dt": candidate.dt,
            "state_labels": list(candidate.state_labels),
            "state_convention": candidate.state_convention,
            "frame_skip": candidate.frame_skip,
            "truncated": candidate.truncated,
        },
        "metrics": {
            "max_abs_error": metrics.max_abs_error.tolist(),
            "mean_abs_error": metrics.mean_abs_error.tolist(),
            "rmse": metrics.rmse.tolist(),
            "overall_max_abs_error": metrics.overall_max_abs_error,
            "overall_rmse": metrics.overall_rmse,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ComparisonArtifacts(csv_path=csv_path, json_path=json_path)


def rollout_single_pendulum_gym(
    *,
    initial_state: npt.ArrayLike,
    control_sequence: npt.ArrayLike,
    env_id: str = "InvertedPendulum-v5",
) -> RolloutResult:
    initial_state_vector = _coerce_vector(initial_state, name="initial_state")
    controls = _coerce_control_sequence(control_sequence)
    env = InvertedPendulumComparisonWrapper(gym.make(env_id))
    try:
        observation, _ = env.reset(initial_state=initial_state_vector)
        states = np.zeros((controls.shape[0] + 1, observation.shape[0]), dtype=np.float64)
        states[0, :] = observation
        truncated = False
        for index, control in enumerate(controls, start=1):
            action = np.array([control], dtype=np.float64)
            observation, _reward, _terminated, truncated, _info = env.step(action)
            states[index, :] = observation
            if truncated:
                states = states[: index + 1]
                break
        return RolloutResult(
            states=states,
            dt=float(env.unwrapped.dt),
            state_labels=_SINGLE_STATE_LABELS,
            state_convention="gym_raw",
            frame_skip=int(env.unwrapped.frame_skip),
            truncated=truncated,
        )
    finally:
        env.close()


def rollout_single_pendulum_casadi(
    *,
    initial_state: npt.ArrayLike,
    control_sequence: npt.ArrayLike,
    dt: float,
    param_path: str = "src/environments/pendulum_params.json",
    integration_method: str = "rk4",
) -> RolloutResult:
    initial_state_vector = _coerce_vector(initial_state, name="initial_state")
    controls = _coerce_control_sequence(control_sequence)
    params = load_pendulum_params(param_path)
    current_state = initial_state_vector.copy()
    states = np.zeros((controls.shape[0] + 1, initial_state_vector.shape[0]), dtype=np.float64)
    states[0, :] = current_state

    x_symbol = ca.SX.sym("x", 4)
    u_symbol = ca.SX.sym("u", 1)
    x_next = pendulum_dynamics(
        x_symbol,
        u_symbol,
        dt,
        params,
        integration_method=integration_method,
    )
    dynamics = ca.Function("single_pendulum_step", [x_symbol, u_symbol], [x_next])

    for index, control in enumerate(controls, start=1):
        current_state = np.asarray(
            dynamics(ca.DM(current_state), ca.DM([control])).full().ravel(),
            dtype=np.float64,
        )
        states[index, :] = current_state

    return RolloutResult(
        states=states,
        dt=float(dt),
        state_labels=_SINGLE_STATE_LABELS,
        state_convention="casadi_state",
        frame_skip=1,
        truncated=False,
    )


def rollout_double_pendulum_gym(
    *,
    initial_state: npt.ArrayLike,
    control_sequence: npt.ArrayLike,
    env_id: str = "InvertedDoublePendulum-v5",
    initial_state_convention: DoublePendulumStateConvention = "casadi_absolute",
    return_state_convention: DoublePendulumStateConvention = "casadi_absolute",
) -> RolloutResult:
    initial_state_vector = _coerce_vector(initial_state, name="initial_state")
    controls = _coerce_control_sequence(control_sequence)
    env = InvertedDoublePendulumComparisonWrapper(
        gym.make(env_id),
        return_state_convention=return_state_convention,
    )
    try:
        observation, _ = env.reset(
            initial_state=initial_state_vector,
            initial_state_convention=initial_state_convention,
        )
        states = np.zeros((controls.shape[0] + 1, observation.shape[0]), dtype=np.float64)
        states[0, :] = observation
        truncated = False
        for index, control in enumerate(controls, start=1):
            action = np.array([control], dtype=np.float64)
            observation, _reward, _terminated, truncated, _info = env.step(action)
            states[index, :] = observation
            if truncated:
                states = states[: index + 1]
                break
        return RolloutResult(
            states=states,
            dt=float(env.unwrapped.dt),
            state_labels=_DOUBLE_STATE_LABELS,
            state_convention=return_state_convention,
            frame_skip=int(env.unwrapped.frame_skip),
            truncated=truncated,
        )
    finally:
        env.close()


def rollout_double_pendulum_casadi(
    *,
    initial_state: npt.ArrayLike,
    control_sequence: npt.ArrayLike,
    dt: float,
    param_path: str = "src/environments/double_pendulum_params.json",
) -> RolloutResult:
    initial_state_vector = _coerce_vector(initial_state, name="initial_state")
    controls = _coerce_control_sequence(control_sequence)
    states = np.zeros((controls.shape[0] + 1, initial_state_vector.shape[0]), dtype=np.float64)
    states[0, :] = initial_state_vector

    current_state = initial_state_vector.copy()
    frame_skip = 1
    for index, control in enumerate(controls, start=1):
        current_state, frame_skip = step_double_pendulum_casadi(
            state=current_state,
            control=control,
            dt=dt,
            param_path=param_path,
        )
        states[index, :] = current_state

    return RolloutResult(
        states=states,
        dt=float(dt),
        state_labels=_DOUBLE_STATE_LABELS,
        state_convention="casadi_absolute",
        frame_skip=frame_skip,
        truncated=False,
    )


def step_double_pendulum_casadi(
    *,
    state: npt.ArrayLike,
    control: float,
    dt: float,
    param_path: str = "src/environments/double_pendulum_params.json",
    integration_timestep: float | None = None,
) -> tuple[FloatArray, int]:
    state_vector = _coerce_vector(state, name="state")
    params, model_timestep, _joint_limits = load_double_pendulum_params(param_path)
    if params is None or model_timestep is None:
        raise ValueError(f"Could not load double-pendulum parameters from {param_path}.")
    substep_dt = model_timestep if integration_timestep is None else float(integration_timestep)
    if substep_dt <= 0.0:
        raise ValueError(f"Model timestep must be positive, got {substep_dt}.")

    substeps_per_step = int(round(dt / substep_dt))
    if substeps_per_step <= 0:
        raise ValueError(
            f"Rollout dt={dt} must be at least one model timestep={substep_dt}."
        )

    reconstructed_dt = substeps_per_step * substep_dt
    if not np.isclose(reconstructed_dt, dt, atol=1e-12, rtol=1e-9):
        raise ValueError(
            "Double-pendulum rollout dt must be an integer multiple of the model timestep. "
            f"Got dt={dt} and model_timestep={substep_dt}."
        )

    parameter_vector = ca.DM(params)
    current_state = ca.DM(state_vector)
    control_dm = ca.DM([control])
    for _ in range(substeps_per_step):
        current_state = double_pendulum_dynamics_step(
            x=current_state,
            u=control_dm,
            dt=substep_dt,
            params=parameter_vector,
            integration_method="rk4",
        )
    return np.asarray(current_state.full().ravel(), dtype=np.float64), substeps_per_step


def metrics_to_dict(metrics: TrajectoryMetrics) -> dict[str, Any]:
    return asdict(metrics)
