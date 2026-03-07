from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from src.utils.parameters import load_double_pendulum_params
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
PARAM_PATH = "src/environments/double_pendulum_params.json"


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


def _load_link_lengths(param_path: str) -> tuple[float, float]:
    params, _dt, _joint_limits, mapping = load_double_pendulum_params(
        param_path,
        return_mapping=True,
    )
    if params is None or mapping is None:
        raise ValueError(f"Could not load double-pendulum parameters from {param_path}.")
    return float(mapping["l1"]), float(mapping["l2"])


def _first_limit_hit_step(
    states: np.ndarray,
    *,
    lower_limit: float,
    upper_limit: float,
) -> int | None:
    for index, position in enumerate(states[:, 0]):
        if position <= lower_limit or position >= upper_limit:
            return index
    return None


def _compute_link_positions(
    state: np.ndarray,
    *,
    link1_length: float,
    link2_length: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    cart_x = float(state[0])
    theta1 = float(state[1])
    theta2 = float(state[2])
    pivot = (cart_x, 0.0)
    joint1 = (
        cart_x + link1_length * np.sin(theta1),
        link1_length * np.cos(theta1),
    )
    tip = (
        joint1[0] + link2_length * np.sin(theta2),
        joint1[1] + link2_length * np.cos(theta2),
    )
    return pivot, joint1, tip


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


def _animation_writer(output_path: Path, fps: int) -> animation.AbstractMovieWriter:
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Saving MP4 animations requires ffmpeg to be available.")
        return animation.FFMpegWriter(fps=fps)
    raise ValueError(
        f"Unsupported animation format '{suffix}'. Use .gif or .mp4 for --animation-path."
    )


def _animate_rollouts(
    time_axis: np.ndarray,
    gym_states: np.ndarray,
    casadi_states: np.ndarray,
    *,
    title: str,
    link1_length: float,
    link2_length: float,
    animation_path: Path | None,
    fps: int,
    show_animation: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    figure.suptitle(title)
    axis.set_aspect("equal")
    axis.set_xlabel("Cart Position (m)")
    axis.set_ylabel("Vertical Position (m)")
    axis.grid(True)

    cart_width = 0.18
    cart_height = 0.09
    max_abs_x = float(np.max(np.abs(np.concatenate([gym_states[:, 0], casadi_states[:, 0]]))))
    x_margin = link1_length + link2_length + 0.3
    y_margin = 0.25
    axis.set_xlim(-max_abs_x - x_margin, max_abs_x + x_margin)
    axis.set_ylim(-link1_length - link2_length - y_margin, link1_length + link2_length + y_margin)

    gym_cart = plt.Rectangle(
        (0.0, -cart_height / 2.0),
        cart_width,
        cart_height,
        color="#ff7f0e",
        alpha=0.45,
    )
    casadi_cart = plt.Rectangle(
        (0.0, -cart_height / 2.0),
        cart_width,
        cart_height,
        color="#1f77b4",
        alpha=0.45,
    )
    axis.add_patch(casadi_cart)
    axis.add_patch(gym_cart)

    casadi_line1, = axis.plot([], [], color="#1f77b4", linewidth=3.0, label="CasADi")
    casadi_line2, = axis.plot([], [], color="#1f77b4", linewidth=3.0)
    casadi_tip, = axis.plot([], [], "o", color="#1f77b4")

    gym_line1, = axis.plot([], [], "--", color="#ff7f0e", linewidth=2.5, label="Gymnasium")
    gym_line2, = axis.plot([], [], "--", color="#ff7f0e", linewidth=2.5)
    gym_tip, = axis.plot([], [], "o", color="#ff7f0e")

    time_text = axis.text(0.02, 0.96, "", transform=axis.transAxes, va="top")
    axis.legend(loc="upper right")

    def update(frame_index: int) -> list[object]:
        casadi_state = casadi_states[frame_index]
        gym_state = gym_states[frame_index]

        casadi_pivot, casadi_joint1, casadi_tip_xy = _compute_link_positions(
            casadi_state,
            link1_length=link1_length,
            link2_length=link2_length,
        )
        gym_pivot, gym_joint1, gym_tip_xy = _compute_link_positions(
            gym_state,
            link1_length=link1_length,
            link2_length=link2_length,
        )

        casadi_cart.set_xy((casadi_pivot[0] - cart_width / 2.0, -cart_height / 2.0))
        gym_cart.set_xy((gym_pivot[0] - cart_width / 2.0, -cart_height / 2.0))

        casadi_line1.set_data(
            [casadi_pivot[0], casadi_joint1[0]],
            [casadi_pivot[1], casadi_joint1[1]],
        )
        casadi_line2.set_data(
            [casadi_joint1[0], casadi_tip_xy[0]],
            [casadi_joint1[1], casadi_tip_xy[1]],
        )
        casadi_tip.set_data([casadi_tip_xy[0]], [casadi_tip_xy[1]])

        gym_line1.set_data([gym_pivot[0], gym_joint1[0]], [gym_pivot[1], gym_joint1[1]])
        gym_line2.set_data([gym_joint1[0], gym_tip_xy[0]], [gym_joint1[1], gym_tip_xy[1]])
        gym_tip.set_data([gym_tip_xy[0]], [gym_tip_xy[1]])

        time_text.set_text(f"Time: {time_axis[frame_index]:.2f} s")
        return [
            casadi_cart,
            gym_cart,
            casadi_line1,
            casadi_line2,
            casadi_tip,
            gym_line1,
            gym_line2,
            gym_tip,
            time_text,
        ]

    animator = animation.FuncAnimation(
        figure,
        update,
        frames=time_axis.shape[0],
        interval=1000.0 / fps,
        blit=True,
        repeat=False,
    )

    if animation_path is not None:
        animation_path.parent.mkdir(parents=True, exist_ok=True)
        animator.save(animation_path, writer=_animation_writer(animation_path, fps))
        print(f"Saved animation: {animation_path}")

    if show_animation:
        plt.show()
    else:
        plt.close(figure)


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
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--animation-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=20)
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

    _params, _dt, joint_limits, _mapping = load_double_pendulum_params(
        PARAM_PATH,
        return_mapping=True,
    )
    if joint_limits is not None and "x" in joint_limits:
        lower_limit, upper_limit = joint_limits["x"]
        first_hit_step = _first_limit_hit_step(
            aligned_gym_states,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        if first_hit_step is not None:
            hit_time = first_hit_step * gym_rollout.dt
            hit_position = aligned_gym_states[first_hit_step, 0]
            print(
                "Gym cart first reaches the slider limit at "
                f"t={hit_time:.2f} s (x={hit_position:.3f})."
            )
            print(
                "Beyond that point MuJoCo includes joint-limit constraint forces "
                "that are not modeled in the current symbolic dynamics."
            )

    artifacts = dump_comparison_artifacts(
        args.artifact_dir,
        f"double_pendulum_{args.mode}_{args.steps}_steps",
        gym_rollout,
        casadi_rollout,
        metrics,
    )
    print(f"Saved comparison CSV: {artifacts.csv_path}")
    print(f"Saved comparison JSON: {artifacts.json_path}")

    time_axis = np.arange(aligned_gym_states.shape[0], dtype=np.float64) * gym_rollout.dt
    title = (
        "InvertedDoublePendulum-v5: Gymnasium vs CasADi "
        f"({args.mode}, dt={gym_rollout.dt:.3f}, frame_skip={gym_rollout.frame_skip})"
    )

    if not args.no_plot:
        _plot_rollouts(
            time_axis,
            aligned_gym_states,
            aligned_casadi_states,
            title=title,
            output_path=args.plot_path,
        )

    if args.animate or args.animation_path is not None:
        link1_length, link2_length = _load_link_lengths(PARAM_PATH)
        _animate_rollouts(
            time_axis,
            aligned_gym_states,
            aligned_casadi_states,
            title=title,
            link1_length=link1_length,
            link2_length=link2_length,
            animation_path=args.animation_path,
            fps=args.fps,
            show_animation=args.animate,
        )


if __name__ == "__main__":
    main()
