# manim/phase_plane_mpc.py
"""Phase-plane visualisation of cart-pole swing-up with MPC.

Layers
------
1. Grey streamlines = open-loop pendulum dynamics (no control)
2. Blue streamlines = closed-loop vector field when MPC is applied
3. Animated dotted horizon + red dot = current state and MPC forecast

The scene re-uses the data generated for the cost-surface animation:
`runs/MPC/<run_id>/manim_mpc_data.pkl`

If you have a proper MPC solver class available, plug it into
`first_control_from_mpc`.  Otherwise (e.g. for rendering without solving
100s of MPC QPs) we fall back to a *nearest-neighbour* look-up: the first
control stored in the `.pkl` file for the closest recorded state.

Run with:
    manim -pql manim/phase_plane_mpc.py PhasePlaneMPC
"""

from __future__ import annotations

import sys
import pickle
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import casadi as ca
from manim import *

# -----------------------------------------------------------------------------
# Project paths and run ID (keep in sync with cost-surface script)
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

RUN_ID = "20250430_184109_36e06939"
data_dir = project_root / "runs" / "MPC" / RUN_ID
pkl_path = data_dir / "manim_mpc_data.pkl"
config_path = data_dir / "config.json"
ctrl_cache_path = data_dir / "ctrl_cache.pkl" # Path for cached control grid

# -----------------------------------------------------------------------------
# Try importing the real MPC solver if available
# -----------------------------------------------------------------------------
try:
    from src.algorithms.classic.mpc_controller import MPCController  # type: ignore
    from src.environments.pendulum_dynamics import _calculate_state_derivatives # Import dynamics function
    from src.utils.parameters import load_pendulum_params # Needed for full params dict
except Exception as e:
    print(f"Warning: Failed to import required modules ({e}). NN fallback only.")
    MPCController = None  # noqa: N816 – fallback placeholder
    _calculate_state_derivatives = None
    load_pendulum_params = None


# -----------------------------------------------------------------------------
# Helper – load recorded trajectory once for fallback control lookup
# -----------------------------------------------------------------------------
with open(pkl_path, "rb") as f:
    recorded_steps: list[dict] = pickle.load(f)


@lru_cache(maxsize=None)
def _nn_control(theta: float, theta_dot: float) -> float:
    """Nearest-neighbour control from recorded MPC data (fallback)."""
    valid_steps = [step for step in recorded_steps if step.get("U_solution") is not None and step["U_solution"].shape[0] > 0]
    if not valid_steps:
        print("Warning: No valid U_solution found in recorded data for NN fallback. Returning 0.")
        return 0.0
    arr = np.array([[step["obs"][1], step["obs"][3], step["U_solution"][0, 0]] for step in valid_steps])
    d = np.square(arr[:, 0] - theta) + np.square(arr[:, 1] - theta_dot)
    idx = int(np.argmin(d))
    return float(arr[idx, 2])


# -----------------------------------------------------------------------------
# Function giving first control for an arbitrary state (θ, θ̇)
# -----------------------------------------------------------------------------
mpc_solver_instance = None # Global instance
if MPCController is not None:

    with open(config_path) as f:
        run_cfg = json.load(f)

    try:
        # --- CORRECTED Instantiation: Extract args from config --- 
        controller_args = {
            "N": int(run_cfg.get("N", 30)),
            "dt_controller": float(run_cfg.get("dt_controller", 0.02)),
            "param_path": str(run_cfg.get("param_path", "src/environments/pendulum_params.json")),
            "cost_type": str(run_cfg.get("cost_type", "quadratic")),
            "guess_type": str(run_cfg.get("guess_type", "warmstart")),
            "q_diag": [float(x) for x in run_cfg.get("q_diag", ["1.0", "20.0", "5.0", "10.0"])], # Convert Q from string list
            "r_val": float(run_cfg.get("r_val", "50.0")),
            "q_terminal_multiplier": float(run_cfg.get("q_terminal_multiplier", 5.0))
        }
        mpc_solver_instance = MPCController(**controller_args)  # Use keyword arguments
        print("Successfully instantiated MPCController.")
    except Exception as err:  # pragma: no cover – fallback to NN
        print(f"⚠️  Could not instantiate MPCController – {err}")
        MPCController = None  # type: ignore[misc]
        mpc_solver_instance = None # Ensure it's None on failure


def first_control_from_mpc(theta: float, theta_dot: float) -> float:  # noqa: N802 – keep snake-case
    """Return first MPC torque for pendulum state (θ, θ̇)."""
    if mpc_solver_instance is None:
        # print("Using NN fallback control") # Optional: uncomment for debug
        return _nn_control(theta, theta_dot)

    state = np.array([0.0, theta, 0.0, theta_dot])  # cart x=0, θ, ẋ=0, θ̇
    try: 
        # Use the step method which should exist and return the control directly
        if hasattr(mpc_solver_instance, 'step') and callable(mpc_solver_instance.step):
            u_val = mpc_solver_instance.step(state)
            return float(u_val)
        else:
            # Fallback if step() method is missing (shouldn't happen based on MPCController code)
            print("Warning: Imported MPCController has no 'step' method. Using fallback NN control.")
            return _nn_control(theta, theta_dot)
            
    except Exception as e:
        print(f"Error calling MPC solver step ({e}). Using fallback NN control.")
        return _nn_control(theta, theta_dot)


# -----------------------------------------------------------------------------
# Full Parameters Dictionary (for dynamics function)
# -----------------------------------------------------------------------------
params_dict = {}
if load_pendulum_params is not None:
    try: 
        # Determine param path from config or default
        param_file_path = "src/environments/pendulum_params.json" # Default
        if 'run_cfg' in locals() and "param_path" in run_cfg:
            param_file_path = run_cfg["param_path"]
        elif Path(config_path).exists(): # Try loading config just for path
            with open(config_path) as f:
                fallback_cfg = json.load(f)
                param_file_path = fallback_cfg.get("param_path", param_file_path)
        
        params_dict = load_pendulum_params(param_file_path)
        print(f"Loaded full params dict from {param_file_path} for dynamics functions.")
    except Exception as e:
        print(f"Warning: Failed to load full params dict ({e}). Dynamics might be incorrect.")
else:
     print("Warning: load_pendulum_params not available. Dynamics might be incorrect.")


# -----------------------------------------------------------------------------
# Main scene
# -----------------------------------------------------------------------------
class PhasePlaneMPC(ThreeDScene):
    def construct(self):
        # --------------------------------------------------------------
        # Axes (θ vs θ̇)
        # --------------------------------------------------------------
        axes = Axes(
            x_range=[-2 * PI, 2 * PI, PI],
            y_range=[-12, 12, 4],
            x_length=8,
            y_length=5,
            tips=False,
            axis_config=dict(include_numbers=True, stroke_width=1),
            # Use decimal_number_config for rounding
            x_axis_config={
                "include_numbers": False, # Disable default numbers for X-axis
            },
            y_axis_config={
                "decimal_number_config": {"num_decimal_places": 1},
                "include_numbers": True,
            },
        )
        # REMOVE post-creation formatting attempts
        # axes.x_axis.number_to_text_func = ...
        # axes.y_axis.number_to_text_func = ...

        # --- Manually add Pi labels for X-axis --- 
        pi_labels = {
            -2 * PI: MathTex("-2\pi"),
            -PI: MathTex("-\pi"),
            0: MathTex("0"), # Manually add 0 if needed, or let default handle?
            PI: MathTex("\pi"),
            2 * PI: MathTex("2\pi"),
        }
        axes.x_axis.add_labels(pi_labels)
        
        # Standard Y axis label
        y_label = axes.get_y_axis_label(MathTex(r"\dot{\theta}"))
        # Create X axis label separately as add_labels might overwrite it
        x_label = axes.get_x_axis_label(MathTex(r"\theta"))
        
        labels = VGroup(x_label, y_label)
        
        self.add(axes, labels)

        # --- Create LaTeX labels ---
        # Using \boldsymbol for state vector x
        open_loop_eq = MathTex(r"\dot{\boldsymbol x} = f(\boldsymbol x)", font_size=36)
        closed_loop_eq = MathTex(r"\dot{\boldsymbol x} = f(\boldsymbol x) + g(\boldsymbol x)\,\boldsymbol{\pi}(\boldsymbol x)", font_size=36)
        state_assumption_eq = MathTex(r"\text{Assuming } x = \dot{x} = 0", font_size=28)
        
        # Position equations using .to_edge(UL)
        open_loop_eq.to_edge(UL)
        closed_loop_eq.align_to(open_loop_eq, UL) # Align to same corner edge
        # Position assumption below the main equations
        state_assumption_eq.next_to(open_loop_eq, DOWN, buff=0.1).align_to(open_loop_eq, LEFT)

        # Store the current equation label for transformations
        current_eq_label = open_loop_eq
        # Add assumption label separately (doesn't transform)
        self.add(state_assumption_eq)

        # --------------------------------------------------------------
        # Open-loop vector field (grey)
        # USE EXTERNAL DYNAMICS FUNCTION
        # --------------------------------------------------------------
        # Removed g, L loading here, use full params_dict
        # g = 9.81 ...
        # L = 1.0 ...

        def open_vec(pt):
            if _calculate_state_derivatives is None or not params_dict:
                return np.array([0,0,0]) # Cannot calculate dynamics
            
            coords = axes.p2c(pt)
            if coords.shape[0] < 2:
                return np.array([0,0,0])
            θ, ω = coords[:2]
            
            # Construct 4D state (assume x=0, x_dot=0 for phase plot)
            x_state = ca.vertcat(0.0, θ, 0.0, ω)
            u_zero = ca.vertcat(0.0) # Zero control input
            
            try:
                state_deriv = _calculate_state_derivatives(x_state, u_zero, params_dict)
                dθ = state_deriv[1].full().item() # theta_dot from derivative
                dω = state_deriv[3].full().item() # theta_ddot from derivative
            except Exception as e:
                 print(f"Error in open_vec dynamics calc: {e}")
                 dθ, dω = 0, 0
            
            # Convert dynamics vector back to screen coordinates delta
            delta_coords = axes.c2p(θ + dθ, ω + dω) - pt
            # Limit magnitude for visual clarity
            norm = np.linalg.norm(delta_coords)
            max_norm = 0.15
            if norm > max_norm and norm != 0:
                 delta_coords = delta_coords * (max_norm / norm)
            return delta_coords

        open_stream = StreamLines(
            open_vec,
            color=GREY_A,
            stroke_width=0.5,
            x_range=[-2 * PI, 2 * PI],
            y_range=[-12, 12],
            padding=1,
        )
        # REMOVED self.add(open_stream)
        # Animate creation immediately after axes, along with its label
        self.play(Create(open_stream), Write(current_eq_label), run_time=2)
        
        # CHANGE wait time after showing open-loop
        self.wait(10)

        # --------------------------------------------------------------
        # Closed-loop field (blue) – sample on coarse grid then interpolate
        # USE EXTERNAL DYNAMICS FUNCTION & CACHING
        # --------------------------------------------------------------
        grid_theta = np.linspace(-2 * PI, 2 * PI, 25)
        grid_omega = np.linspace(-12, 12, 25)
        ctrl_cache: dict[tuple[float, float], np.ndarray] = {}
        
        # --- Caching Logic ---
        if ctrl_cache_path.exists():
            print(f"Loading control grid from cache: {ctrl_cache_path}")
            try:
                with open(ctrl_cache_path, "rb") as f_cache:
                    ctrl_cache = pickle.load(f_cache)
                # Basic validation (check if keys match expected grid? Optional)
                if not isinstance(ctrl_cache, dict):
                    print("Warning: Cache file invalid format. Recalculating.")
                    ctrl_cache = {}
            except Exception as e:
                print(f"Warning: Failed to load cache file ({e}). Recalculating.")
                ctrl_cache = {}
        else:
             print("Control grid cache not found. Calculating...")

        if not ctrl_cache: # If cache was empty or failed to load
            if _calculate_state_derivatives is not None and params_dict:
                print("Calculating Control Grid...") # Simple print message
                # Use rich.progress context manager
                for θ_grid in grid_theta:
                    for ω_grid in grid_omega:
                        u = first_control_from_mpc(θ_grid, ω_grid) # Get MPC control
                        x_state_grid = ca.vertcat(0.0, θ_grid, 0.0, ω_grid)
                        u_input = ca.vertcat(u)
                        try:
                            state_deriv_grid = _calculate_state_derivatives(x_state_grid, u_input, params_dict)
                            dθ_grid = state_deriv_grid[1].full().item()
                            dω_grid = state_deriv_grid[3].full().item()
                            ctrl_cache[(θ_grid, ω_grid)] = np.array([dθ_grid, dω_grid])
                        except Exception as e:
                            ctrl_cache[(θ_grid, ω_grid)] = np.array([0.0, 0.0]) # Default on error
                
                # Save the calculated cache
                try:
                    with open(ctrl_cache_path, "wb") as f_cache:
                        pickle.dump(ctrl_cache, f_cache)
                    print(f"Control grid saved to cache: {ctrl_cache_path}")
                except Exception as e:
                    print(f"Warning: Failed to save control grid cache ({e}).")
            else:
                print("Warning: Cannot calculate closed-loop field, dynamics function unavailable.")

        # --- Define ctrl_vec function (uses ctrl_cache) ---
        def ctrl_vec(pt):
            if not ctrl_cache:
                 return np.array([0,0,0]) # Cannot calculate
                 
            coords = axes.p2c(pt)
            if coords.shape[0] < 2:
                return np.array([0,0,0])
            θ, ω = coords[:2]
            
            # Find nearest grid point for cached control
            θg = grid_theta[np.argmin(np.abs(grid_theta - θ))]
            ωg = grid_omega[np.argmin(np.abs(grid_omega - ω))]
            dθ, dω = ctrl_cache.get((θg, ωg), np.array([0.0, 0.0])) # Use .get for safety
            
            # Convert dynamics vector back to screen coordinates delta
            delta_coords = axes.c2p(θ + dθ, ω + dω) - pt
            # Limit magnitude
            norm = np.linalg.norm(delta_coords)
            max_norm = 0.15
            if norm > max_norm and norm != 0:
                 delta_coords = delta_coords * (max_norm / norm)
            return delta_coords

        closed_stream = StreamLines(
            ctrl_vec,
            color=BLUE_D,
            stroke_width=1.5,
            x_range=[-2 * PI, 2 * PI],
            y_range=[-12, 12],
            padding=1,
        )
        # Animate closed_stream creation AND transform the equation label
        # We transform current_eq_label which starts as open_loop_eq
        self.play(Create(closed_stream), Transform(current_eq_label, closed_loop_eq), run_time=2)

        # Fade open-loop slightly to emphasise control
        self.play(open_stream.animate.set_opacity(0.3), run_time=1)

        # --------------------------------------------------------------
        # Animated MPC horizon (dotted) + current state (red dot)
        # --------------------------------------------------------------
        red_dot = Dot(color=RED, radius=0.06).move_to(
            axes.c2p(recorded_steps[0]["obs"][1], recorded_steps[0]["obs"][3])
        )
        self.add(red_dot)

        # INITIALIZE step_idx BEFORE using it in always_redraw
        self.step_idx = 0  # type: ignore[attr-defined]

        # always_redraw horizon polyline so it updates each loop index self.step_idx
        def horizon_vmobject():
            # Access self.step_idx safely now
            step = recorded_steps[self.step_idx] 
            if step["X_solution"] is None:
                return VMobject() # Return empty if no solution
            pts = [axes.c2p(x[1], x[3]) for x in step["X_solution"].T]
            # Create the dashed line
            line = VMobject(stroke_color=ORANGE, stroke_width=2, fill_opacity=0, z_index=4) # Explicitly set fill_opacity=0
            line.set_points_as_corners(pts)
            line.set_opacity(0.8)
            line.set_dash_array([0.1, 0.1])
            # Create dots at each forecast point, ensure no fill
            dots = VGroup(*[Dot(point=p, radius=0.03, color=ORANGE, fill_opacity=0.8) for p in pts])
            dots.set_fill(opacity=0) # Ensure dots group has no fill
            # Return group containing line and dots, ensure no fill
            final_group = VGroup(line, dots)
            final_group.set_fill(opacity=0)
            return final_group

        horizon = always_redraw(horizon_vmobject)
        self.add(horizon)

        # main loop – animate through first N recorded closed-loop steps
        # self.step_idx = 0 # Already initialized
        # REDUCED upper limit for faster debugging
        for k in range(1, min(3, len(recorded_steps))): 
            self.step_idx = k  # type: ignore[attr-defined]
            start = red_dot.get_center()
            end   = axes.c2p(recorded_steps[k]["obs"][1], recorded_steps[k]["obs"][3])
            self.play(MoveAlongPath(red_dot, Line(start, end), rate_func=linear), run_time=1.2)

        self.wait(1)