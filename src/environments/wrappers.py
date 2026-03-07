from __future__ import annotations

from typing import Any, Literal, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
DoublePendulumStateConvention: TypeAlias = Literal["gym_raw", "casadi_absolute"]


def _coerce_state_vector(
    state: npt.ArrayLike,
    *,
    expected_dim: int,
    name: str,
) -> FloatArray:
    vector = np.asarray(state, dtype=np.float64)
    if vector.shape != (expected_dim,):
        raise ValueError(f"{name} must have shape ({expected_dim},), got {vector.shape}.")
    return vector


def gym_raw_double_pendulum_to_casadi(state: npt.ArrayLike) -> FloatArray:
    raw_state = _coerce_state_vector(
        state,
        expected_dim=6,
        name="Gym raw double-pendulum state",
    )
    casadi_state = raw_state.copy()
    casadi_state[2] = raw_state[1] + raw_state[2]
    casadi_state[5] = raw_state[4] + raw_state[5]
    return casadi_state


def casadi_double_pendulum_to_gym_raw(state: npt.ArrayLike) -> FloatArray:
    casadi_state = _coerce_state_vector(
        state,
        expected_dim=6,
        name="CasADi double-pendulum state",
    )
    raw_state = casadi_state.copy()
    raw_state[2] = casadi_state[2] - casadi_state[1]
    raw_state[5] = casadi_state[5] - casadi_state[4]
    return raw_state


class DiscretizeActionWrapper(gym.ActionWrapper):
    """Map a 1D continuous Box action space to a discrete action space."""

    def __init__(self, env: gym.Env[Any, Any], n_bins: int) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("DiscretizeActionWrapper requires a Box action space.")
        if len(env.action_space.shape) != 1:
            raise ValueError(
                "DiscretizeActionWrapper only supports 1D Box action spaces, "
                f"got shape {env.action_space.shape}."
            )

        self.n_bins = n_bins
        self.original_action_space = env.action_space
        self.low = self.original_action_space.low.astype(np.float64)
        self.high = self.original_action_space.high.astype(np.float64)
        self.action_space = gym.spaces.Discrete(self.n_bins)

    def action(self, action: int) -> npt.NDArray[np.generic]:
        if self.n_bins == 1:
            continuous_action = self.low + (self.high - self.low) / 2.0
        else:
            continuous_action = self.low + (self.high - self.low) * action / (self.n_bins - 1)
        clipped = np.clip(continuous_action, self.low, self.high)
        return clipped.astype(self.original_action_space.dtype)


class InvertedPendulumComparisonWrapper(gym.Wrapper):
    """Expose a long-horizon comparison interface for `InvertedPendulum-v5`."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        super().__init__(env)
        self._remove_hinge_limits_if_available()

    def _remove_hinge_limits_if_available(self) -> None:
        try:
            model = self.env.unwrapped.model
            hinge_index = -1
            for index in range(model.njnt):
                if model.joint(index).name == "hinge":
                    hinge_index = index
                    break
            if hinge_index != -1:
                model.jnt_limited[hinge_index] = 0
                model.jnt_range[hinge_index] = [-1e10, 1e10]
        except Exception:
            return

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        initial_state: npt.ArrayLike | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        if initial_state is None:
            return np.asarray(observation, dtype=np.float64), info

        state_vector = _coerce_state_vector(
            initial_state,
            expected_dim=4,
            name="Inverted pendulum initial state",
        )
        qpos = state_vector[:2]
        qvel = state_vector[2:]
        self.env.unwrapped.set_state(qpos, qvel)
        observation = self.env.unwrapped._get_obs()
        return np.asarray(observation, dtype=np.float64), info

    def step(
        self,
        action: npt.ArrayLike,
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        observation, reward, _terminated, truncated, info = self.env.step(action)
        return np.asarray(observation, dtype=np.float64), float(reward), False, truncated, info


class InvertedDoublePendulumComparisonWrapper(gym.Wrapper):
    """Expose raw or converted double-pendulum state for reproducible comparisons."""

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        return_state_convention: DoublePendulumStateConvention = "gym_raw",
    ) -> None:
        super().__init__(env)
        self.return_state_convention = return_state_convention
        self.nq = int(self.env.unwrapped.model.nq)
        self.nv = int(self.env.unwrapped.model.nv)
        if (self.nq, self.nv) != (3, 3):
            raise ValueError(f"Expected nq=3 and nv=3, got {(self.nq, self.nv)}.")
        self.observation_space = gym.spaces.Box(
            low=np.full(6, -np.inf, dtype=np.float64),
            high=np.full(6, np.inf, dtype=np.float64),
            dtype=np.float64,
        )

    def _get_raw_state(self) -> FloatArray:
        qpos = np.asarray(self.env.unwrapped.data.qpos, dtype=np.float64)
        qvel = np.asarray(self.env.unwrapped.data.qvel, dtype=np.float64)
        return np.concatenate([qpos, qvel]).astype(np.float64)

    def _convert_for_return(self, raw_state: FloatArray) -> FloatArray:
        if self.return_state_convention == "gym_raw":
            return raw_state
        return gym_raw_double_pendulum_to_casadi(raw_state)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        initial_state: npt.ArrayLike | None = None,
        initial_state_convention: DoublePendulumStateConvention = "gym_raw",
    ) -> tuple[FloatArray, dict[str, Any]]:
        _, info = super().reset(seed=seed, options=options)

        if initial_state is not None:
            if initial_state_convention == "gym_raw":
                raw_state = _coerce_state_vector(
                    initial_state,
                    expected_dim=6,
                    name="Double-pendulum raw initial state",
                )
            elif initial_state_convention == "casadi_absolute":
                raw_state = casadi_double_pendulum_to_gym_raw(initial_state)
            else:
                raise ValueError(f"Unknown initial state convention: {initial_state_convention}.")
            self.env.unwrapped.set_state(raw_state[: self.nq], raw_state[self.nq :])

        return self._convert_for_return(self._get_raw_state()), info

    def step(
        self,
        action: npt.ArrayLike,
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        _, reward, _terminated, truncated, info = self.env.step(action)
        return (
            self._convert_for_return(self._get_raw_state()),
            float(reward),
            False,
            truncated,
            info,
        )


class SwingUpResetWrapper(gym.Wrapper):
    """Reset the pendulum to the hanging-down start state."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        super().__init__(env)
        self.unwrapped_env = env.unwrapped
        self.initial_qpos = np.array([0.0, -np.pi], dtype=np.float64)
        self.initial_qvel = np.array([0.0, 0.0], dtype=np.float64)

    def reset(self, **kwargs: Any) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.unwrapped_env.set_state(self.initial_qpos, self.initial_qvel)
        observation = self.unwrapped_env._get_obs()
        return np.asarray(observation, dtype=np.float64), info
