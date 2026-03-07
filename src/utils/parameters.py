from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias

PendulumParams: TypeAlias = dict[str, Any]
JointLimits: TypeAlias = dict[str, list[float]]
DoublePendulumParams: TypeAlias = list[float]
DoublePendulumLoadResult: TypeAlias = tuple[
    DoublePendulumParams | None,
    float | None,
    JointLimits | None,
]
DoublePendulumLoadResultWithMapping: TypeAlias = tuple[
    DoublePendulumParams | None,
    float | None,
    JointLimits | None,
    PendulumParams | None,
]
_DOUBLE_PENDULUM_PARAM_ORDER = [
    "M",
    "m1",
    "m2",
    "l1",
    "l2",
    "d1",
    "d2",
    "Icm1",
    "Icm2",
    "g",
    "b_slide",
    "b_fric",
    "b_joint1",
    "b_joint2",
    "gear",
]


def _read_json_mapping(json_file: str | Path) -> PendulumParams:
    path = Path(json_file)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}.")
    return payload


def load_pendulum_params(
    json_file: str | Path = "pendulum_params.json",
    *,
    verbose: bool = False,
) -> PendulumParams:
    """Load scalar and bounds parameters for the single inverted pendulum."""
    params = _read_json_mapping(json_file)
    if params.get("pole_length") is None and params.get("pole_half_length") is not None:
        params["pole_length"] = 2.0 * float(params["pole_half_length"])

    if verbose:
        print(f"Loaded Inverted Pendulum parameters from '{json_file}':")
        for key, value in params.items():
            print(f"  {key} = {value}")

    return params


def load_double_pendulum_params(
    json_file: str | Path = "src/environments/double_pendulum_params.json",
    *,
    return_mapping: bool = False,
    verbose: bool = False,
) -> DoublePendulumLoadResult | DoublePendulumLoadResultWithMapping:
    """Load ordered model parameters, model timestep, and joint limits for the double pendulum."""
    path = Path(json_file)
    try:
        all_params = _read_json_mapping(path)
    except FileNotFoundError:
        if verbose:
            print(f"Error: Parameter file not found at {path}")
        return (None, None, None, None) if return_mapping else (None, None, None)
    except (json.JSONDecodeError, ValueError) as error:
        if verbose:
            print(f"Error loading JSON from {path}: {error}")
        return (None, None, None, None) if return_mapping else (None, None, None)

    symbolic_params = all_params.get("symbolic_params_mapping")
    if not isinstance(symbolic_params, dict):
        if verbose:
            print(f"Error: 'symbolic_params_mapping' missing from {path}.")
        return (None, None, None, None) if return_mapping else (None, None, None)

    ordered_param_list: list[float] = []
    missing_keys: list[str] = []
    for key in _DOUBLE_PENDULUM_PARAM_ORDER:
        value = symbolic_params.get(key)
        if value is None:
            missing_keys.append(key)
            continue
        try:
            ordered_param_list.append(float(value))
        except (TypeError, ValueError):
            missing_keys.append(key)

    if missing_keys:
        if verbose:
            print(f"Error: Missing or invalid double-pendulum parameters: {missing_keys}")
        return (None, None, None, None) if return_mapping else (None, None, None)

    dt = float(symbolic_params.get("dt", 0.01))
    raw_joint_limits = symbolic_params.get("joint_limits", {})
    joint_limits: JointLimits = {}
    if isinstance(raw_joint_limits, dict):
        for key, value in raw_joint_limits.items():
            if isinstance(value, list) and len(value) == 2:
                joint_limits[key] = [float(value[0]), float(value[1])]

    if verbose:
        print(
            "Successfully extracted "
            f"{len(ordered_param_list)} double-pendulum parameters, dt={dt}, and joint limits."
        )

    if return_mapping:
        return ordered_param_list, dt, joint_limits, symbolic_params
    return ordered_param_list, dt, joint_limits
