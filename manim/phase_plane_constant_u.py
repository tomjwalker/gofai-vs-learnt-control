# manim/phase_plane_constant_u.py
"""Phase-plane visualisation of cart-pole with CONSTANT control inputs.

Sweeps through different constant values for the control input `u` and shows
the resulting vector field (blue streamlines) overlaid on the open-loop
dynamics (grey streamlines).

Uses the dynamics function from `pendulum_dynamics.py` and parameters
from the config file associated with the specified RUN_ID.

Run with:
    manim -pql manim/phase_plane_constant_u.py PhasePlaneConstantU
"""

from __future__ import annotations

import sys
import pickle
import json
from pathlib import Path

import numpy as np
import casadi as ca
from manim import *

# -----------------------------------------------------------------------------
# Project paths and run ID
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

RUN_ID = "20250430_184109_36e06939" # Used to find config/params file
data_dir = project_root / "runs" / "MPC" / RUN_ID
config_path = data_dir / "config.json"

# -----------------------------------------------------------------------------
# Try importing dynamics and parameter loading functions
# -----------------------------------------------------------------------------
try:
    from src.environments.pendulum_dynamics import _calculate_state_derivatives
    from src.utils.parameters import load_pendulum_params
except Exception as e:
    print(f"ERROR: Failed to import required modules ({e}). Cannot run.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Load Parameters Dictionary (Needed for dynamics function)
# -----------------------------------------------------------------------------
params_dict = {}
try:
    param_file_path = "src/environments/pendulum_params.json" # Default
    if Path(config_path).exists():
        with open(config_path) as f:
            run_cfg = json.load(f)
            param_file_path = run_cfg.get("param_path", param_file_path)
    
    params_dict = load_pendulum_params(param_file_path)
    print(f"Loaded full params dict from {param_file_path} for dynamics functions.")
except Exception as e:
    print(f"ERROR: Failed to load pendulum params dict ({e}). Cannot run.")
    sys.exit(1)

if not params_dict:
    print("ERROR: Parameters dictionary is empty. Cannot run.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Main scene
# -----------------------------------------------------------------------------
class PhasePlaneConstantU(ThreeDScene): # Use 3D for consistency if needed
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
            x_axis_config={"include_numbers": False}, # Only Pi labels
            y_axis_config={"decimal_number_config": {"num_decimal_places": 1}},
        )
        pi_labels = {
            -2 * PI: MathTex("-2\pi"), -PI: MathTex("-\pi"),
             0: MathTex("0"), PI: MathTex("\pi"), 2 * PI: MathTex("2\pi"),
        }
        axes.x_axis.add_labels(pi_labels)
        x_label = axes.get_x_axis_label(MathTex(r"\theta"))
        y_label = axes.get_y_axis_label(MathTex(r"\dot{\theta}"))
        labels = VGroup(x_label, y_label)
        self.add(axes, labels)

        # --------------------------------------------------------------
        # Open-loop vector field (grey)
        # --------------------------------------------------------------
        def open_vec(pt):
            coords = axes.p2c(pt)
            if coords.shape[0] < 2: return np.array([0,0,0])
            θ, ω = coords[:2]
            x_state = ca.vertcat(0.0, θ, 0.0, ω)
            u_zero = ca.vertcat(0.0)
            try:
                state_deriv = _calculate_state_derivatives(x_state, u_zero, params_dict)
                dθ, dω = state_deriv[1].full().item(), state_deriv[3].full().item()
            except Exception: dθ, dω = 0, 0
            delta_coords = axes.c2p(θ + dθ, ω + dω) - pt
            norm = np.linalg.norm(delta_coords)
            max_norm = 0.15
            if norm > max_norm and norm != 0: delta_coords *= (max_norm / norm)
            return delta_coords

        open_stream = StreamLines(
            open_vec, color=GREY_A, stroke_width=0.5,
            x_range=[-2 * PI, 2 * PI], y_range=[-12, 12], padding=1,
        )
        self.play(Create(open_stream), run_time=2)
        self.wait(2) # Hold the open-loop view

        # --------------------------------------------------------------
        # Constant Control Sweep
        # --------------------------------------------------------------
        u_values = np.arange(-3, 4, 1) # Control values from -3 to 3
        
        # Placeholder Mobjects for transformation
        current_streamlines = VMobject() # Start empty
        current_u_label = VMobject()

        for i, u_val in enumerate(u_values):
            print(f"Visualizing constant u = {u_val}")
            # Define vector field function for this SPECIFIC u_val
            def constant_u_vec(pt, current_u=u_val): # Capture u_val
                coords = axes.p2c(pt)
                if coords.shape[0] < 2: return np.array([0,0,0])
                θ, ω = coords[:2]
                x_state = ca.vertcat(0.0, θ, 0.0, ω)
                u_input = ca.vertcat(current_u)
                try:
                    state_deriv = _calculate_state_derivatives(x_state, u_input, params_dict)
                    dθ, dω = state_deriv[1].full().item(), state_deriv[3].full().item()
                except Exception: dθ, dω = 0, 0
                delta_coords = axes.c2p(θ + dθ, ω + dω) - pt
                norm = np.linalg.norm(delta_coords)
                max_norm = 0.15
                if norm > max_norm and norm != 0: delta_coords *= (max_norm / norm)
                return delta_coords

            # Create new streamlines and label for this u
            new_streamlines = StreamLines(
                constant_u_vec, color=BLUE_D, 
                stroke_width=1.8,
                x_range=[-2 * PI, 2 * PI], y_range=[-12, 12], padding=1,
            )
            new_u_label = MathTex(f"u = {u_val}", font_size=36).to_edge(UL)

            # Animation: Transform or Create
            if i == 0:
                # First iteration: Create the blue streamlines and label
                self.play(Create(new_streamlines), Write(new_u_label), run_time=1.5)
                # Initialize placeholders AFTER first creation
                current_streamlines = new_streamlines
                current_u_label = new_u_label
            else:
                # Subsequent iterations: Use ReplacementTransform
                self.play(
                    ReplacementTransform(current_streamlines, new_streamlines),
                    ReplacementTransform(current_u_label, new_u_label),
                    run_time=1.5
                )
                # Update placeholders AFTER the transform animation completes
                current_streamlines = new_streamlines
                current_u_label = new_u_label
            
            self.wait(1.5) # Hold view for this u_val

        self.wait(5) # Keep final frame visible 