from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.trajectory_comparison import (
    align_rollouts,
    compute_trajectory_metrics,
    dump_comparison_artifacts,
    format_metrics_summary,
    rollout_double_pendulum_casadi,
    rollout_double_pendulum_gym,
)

DEFAULT_INITIAL_STATE = np.array([0.0, 0.1, -0.1, 0.0, 0.0, 0.0], dtype=np.float64)
STATE_LABELS = ("x", "theta1", "theta2", "x_dot", "theta1_dot", "theta2_dot")


def _build_control_sequence(mode: str, steps: int, amplitude: float) -> np.ndarray:
    if mode == "free":
        return np.zeros(steps, dtype=np.float64)
    half_steps = steps // 2
    return np.concatenate(
        [
            np.full(half_steps, -amplitude, dtype=np.float64),
            np.full(steps - half_steps, amplitude, dtype=np.float64),
        ]
    )


def _plot_rollouts(
    time_axis: np.ndarray,
    gym_states: np.ndarray,
    casadi_states: np.ndarray,
    *,
    title: str,
    output_path: Path | None,
) -> None:
    figure, axes = plt.subplots(len(STATE_LABELS), 1, figsize=(10, 12), sharex=True)
    figure.suptitle(title)
    for index, label in enumerate(STATE_LABELS):
        axes[index].plot(time_axis, casadi_states[:, index], label="CasADi", linewidth=2.0)
        axes[index].plot(time_axis, gym_states[:, index], label="Gymnasium", linestyle="--")
        axes[index].set_ylabel(label)
        axes[index].grid(True)
        axes[index].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gymnasium and CasADi trajectories for InvertedDoublePendulum-v5.",
    )
    parser.add_argument("--mode", choices=("free", "control"), default="free")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--control-amplitude", type=float, default=0.1)
    parser.add_argument("--artifact-dir", type=Path, default=Path("analysis") / "artifacts")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    control_sequence = _build_control_sequence(args.mode, args.steps, args.control_amplitude)

    gym_rollout = rollout_double_pendulum_gym(
        initial_state=DEFAULT_INITIAL_STATE,
        control_sequence=control_sequence,
    )
    casadi_rollout = rollout_double_pendulum_casadi(
        initial_state=DEFAULT_INITIAL_STATE,
        control_sequence=control_sequence,
        dt=gym_rollout.dt,
    )

    aligned_gym_states, aligned_casadi_states = align_rollouts(gym_rollout, casadi_rollout)
    metrics = compute_trajectory_metrics(aligned_gym_states, aligned_casadi_states)
    summary = format_metrics_summary(metrics, gym_rollout.state_labels)

    print(f"Mode: {args.mode}")
    print(f"Gymnasium dt: {gym_rollout.dt:.5f} s")
    print(f"Gymnasium frame_skip: {gym_rollout.frame_skip}")
    print(f"Gymnasium state convention: {gym_rollout.state_convention}")
    print(f"CasADi state convention: {casadi_rollout.state_convention}")
    print(summary)

    artifacts = dump_comparison_artifacts(
        args.artifact_dir,
        f"double_pendulum_{args.mode}_{args.steps}_steps",
        gym_rollout,
        casadi_rollout,
        metrics,
    )
    print(f"Saved comparison CSV: {artifacts.csv_path}")
    print(f"Saved comparison JSON: {artifacts.json_path}")

    if not args.no_plot:
        time_axis = np.arange(aligned_gym_states.shape[0], dtype=np.float64) * gym_rollout.dt
        plot_title = (
            "InvertedDoublePendulum-v5: Gymnasium vs CasADi "
            f"({args.mode}, dt={gym_rollout.dt:.3f}, frame_skip={gym_rollout.frame_skip})"
        )
        _plot_rollouts(
            time_axis,
            aligned_gym_states,
            aligned_casadi_states,
            title=plot_title,
            output_path=args.plot_path,
        )


if __name__ == "__main__":
    main()
