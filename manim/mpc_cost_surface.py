# manim/mpc_cost_surface.py
import sys
import pickle
from pathlib import Path
import numpy as np
import casadi as ca
from manim import * # Import Manim library
import json # Added for loading config

# Add project root to sys.path to allow importing src modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import necessary components from the project
from src.algorithms.classic.mpc_costs import PendulumSwingupAtan2Cost # Assuming this cost was used
from src.utils.parameters import load_pendulum_params

# REDUCED number of steps for testing completion
MAX_STEPS_TO_ANIMATE = 5 

class MPCCostSurfaceAnimation(ThreeDScene):
    def construct(self):
        # --- Configuration ---
        run_id = "20250430_184109_36e06939" 
        data_dir = project_root / "runs" / "MPC" / run_id
        data_path = data_dir / "manim_mpc_data.pkl"
        config_path = data_dir / "config.json" # Path to config file
        # param_path = project_root / "src" / "environments" / "pendulum_params_swingup.json" # TODO: Get from config - Removed

        # --- Load Run Configuration ---
        print(f"Loading run configuration from: {config_path}")
        if not config_path.exists():
            error_text = Text(f"Error: Config file not found at\n{config_path}", color=RED, font_size=24)
            self.add(error_text)
            self.wait(5)
            return
        try:
            with open(config_path, 'r') as f:
                run_config = json.load(f)
            print("Run configuration loaded.")
        except Exception as e:
            error_text = Text(f"Error loading config: {e}", color=RED, font_size=24)
            self.add(error_text)
            self.wait(5)
            return
            
        # Extract relevant config values (provide defaults?)
        param_path_str = run_config.get("param_path", "src/environments/pendulum_params_swingup.json")
        param_path = project_root / param_path_str # Convert to Path object relative to project root
        q_diag_str = run_config.get("q_diag", ["0.1", "5.0", "0.1", "0.1"]) # Get as strings (or default strings)
        r_val_str = run_config.get("r_val", "0.01") # Get R as string too for consistency
        
        # Convert Q and R to floats
        try:
            q_diag = [float(x) for x in q_diag_str]
            r_val = float(r_val_str)
        except ValueError as e:
            error_text = Text(f"Error converting Q/R from config to float: {e}", color=RED, font_size=24)
            self.add(error_text)
            self.wait(5)
            return
            
        # --- ADJUST COST WEIGHTS --- 
        # Modify weights AFTER loading from config
        if len(q_diag) > 3:
            print(f"Original q_diag[1] (theta): {q_diag[1]}")
            print(f"Original q_diag[3] (theta_dot): {q_diag[3]}")
            # Reduce theta weight
            q_diag[1] *= 0.8 
            # Set theta_dot weight (original * 2.0 * 0.75 = original * 1.5)
            # Assuming the loaded value is the 'original'
            # Let's recalculate based on loaded value to avoid compounding if script runs weirdly
            original_theta_dot_weight_from_config = float(run_config.get("q_diag", ["0.1", "5.0", "0.1", "0.1"])[3]) # Reload original string and convert
            q_diag[3] = original_theta_dot_weight_from_config * 1.5 # Apply desired factor (1.5 = 2.0 * 0.75)
            print(f"Modified q_diag[1] (theta): {q_diag[1]}")
            print(f"Modified q_diag[3] (theta_dot): {q_diag[3]}")
        else:
            print("Warning: q_diag does not have enough elements to adjust weights.")
            
        # TODO: Check cost_type from config matches PendulumSwingupAtan2Cost?
        # cost_type_from_config = run_config.get("cost_type")

        # State indices to plot on X and Y axes of the surface
        # 1: theta (pole angle), 3: theta_dot (pole angular velocity)
        x_axis_state_idx = 1 
        y_axis_state_idx = 3
        
        # Fixed state values for dimensions not plotted on X/Y
        # Use reference state (0) for cart pos and velocity for now
        fixed_state_values = {0: 0.0, 2: 0.0} 
        
        # Fixed control value for cost surface calculation
        fixed_control_value = np.array([0.0])

        # --- Load Data ---
        print(f"Loading Manim data from: {data_path}")
        if not data_path.exists():
            error_text = Text(f"Error: Data file not found at{data_path}", color=RED, font_size=24)
            self.add(error_text)
            self.wait(5)
            return
            
        try:
            with open(data_path, 'rb') as f:
                manim_data = pickle.load(f)
            print(f"Loaded {len(manim_data)} MPC steps.")
        except Exception as e:
            error_text = Text(f"Error loading data: {e}", color=RED, font_size=24)
            self.add(error_text)
            self.wait(5)
            return
            
        if not manim_data:
            error_text = Text("Error: Loaded data is empty.", color=RED, font_size=24)
            self.add(error_text)
            self.wait(5)
            return

        # --- Setup Cost Calculation ---
        # Use loaded Q/R values
        # Q_diag = [0.1, 5.0, 0.1, 0.1] # Example values used in run - Removed
        # R_val = 0.01 - Removed
        Q = ca.diag(q_diag)
        R = ca.DM([r_val])
        X_ref = ca.DM.zeros(4, 1)
        
        cost_calculator = PendulumSwingupAtan2Cost()
        
        # Create symbolic variables for the cost function
        Xk_sym = ca.SX.sym('Xk', 4)
        Uk_sym = ca.SX.sym('Uk', 1)
        
        # Create a CasADi function for the stage cost
        # We only need the stage cost for the surface plot
        stage_cost_func = ca.Function(
            'stage_cost', 
            [Xk_sym, Uk_sym], 
            [cost_calculator.calculate_stage_cost(Xk_sym, Uk_sym, X_ref, Q, R)]
        )

        # --- Define Cost Surface Function (for Manim) ---
        def cost_surface_func(u, v):
            # u corresponds to the state on the x-axis (theta)
            # v corresponds to the state on the y-axis (theta_dot)
            state_vec = np.zeros(4)
            state_vec[x_axis_state_idx] = u
            state_vec[y_axis_state_idx] = v
            # Fill in fixed values for other states
            for idx, val in fixed_state_values.items():
                 if idx != x_axis_state_idx and idx != y_axis_state_idx:
                     state_vec[idx] = val
                     
            # Calculate cost using the CasADi function
            cost_val = stage_cost_func(state_vec, fixed_control_value).full().item()
            # Clamp extremely high costs for visualization? Maybe later.
            return cost_val

        # --- Create 3D Axes ---
        axes = ThreeDAxes(
            # ADJUSTED axis ranges
            x_range=[-2 * PI, 2 * PI, PI],       # Theta range approx (-2pi to 2pi)
            y_range=[-12, 12, 4],                # Theta_dot range approx (-12 to 12)
            z_range=[0, 50, 10],                 # Cost range (adjust based on data)
            x_length=7,
            y_length=7,
            z_length=5,
            axis_config={"include_numbers": True, "include_tip": True},
            # Remove labels from constructor
        )
        
        # --- Add axis labels manually ---
        # ESCAPED backslashes for LaTeX
        x_label = axes.get_x_axis_label(Tex("$\\theta$ (rad)")) 
        y_label = axes.get_y_axis_label(Tex("$\\dot{\\theta}$ (rad/s)"))
        z_label = axes.get_z_axis_label(Tex("Cost")) 
        
        axis_labels = VGroup(x_label, y_label, z_label) # Group labels

        # --- Create Cost Surface ---
        surface = Surface(
            lambda u, v: axes.c2p(u, v, cost_surface_func(u, v)),
            # UPDATED surface ranges to match axes
            u_range=[-2 * PI, 2 * PI],
            v_range=[-12, 12],
            resolution=(48, 48), # Adjust resolution for quality/performance
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_color=LIGHT_GREY,
            stroke_width=0.5,
        )

        # --- Animation Setup ---
        # Adjust camera angle (lower, rotated 90deg), zoom further
        self.set_camera_orientation(phi=50 * DEGREES, theta=135 * DEGREES, zoom=0.45) 
        self.add(axes)
        self.play(Create(surface), Create(axis_labels)) # Create labels alongside surface

        # Add title BEFORE the loop - Use MathTex for better LaTeX rendering
        title = MathTex("\text{MPC Cost Surface (}\\theta\text{ vs }\\dot{\\theta}\text{)}", font_size=36).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title) # Keep title fixed on screen

        # --- Animation Loop Initialization ---
        # Get data for the very first step to initialize elements
        first_step_data = manim_data[0]
        initial_obs = first_step_data["obs"]
        initial_cost = cost_surface_func(initial_obs[x_axis_state_idx], initial_obs[y_axis_state_idx])
        initial_point_coords = axes.c2p(
            initial_obs[x_axis_state_idx], 
            initial_obs[y_axis_state_idx], 
            initial_cost
        )
        
        # Create persistent Mobjects to update
        current_state_dot = Dot3D(point=initial_point_coords, color=RED, radius=0.08)
        step_label = Text(f"Step: {first_step_data['step']}", font_size=24).to_corner(UL)
        self.add_fixed_in_frame_mobjects(step_label) # Add label fixed on screen
        self.add(current_state_dot) # Add the initial dot
        
        # Variables to manage animations
        last_prediction_group = VGroup() # Initialize prediction group (will be populated in first loop iteration)
        # REMOVED combined step_anim_run_time
        # step_anim_run_time = 0.5 
        # NEW run times for different phases
        forecast_fadeout_run_time = 0.25
        forecast_anim_run_time = 3.0 
        state_update_run_time = 1.5
        # REMOVED wait_time_per_step as we are controlling timing with distinct play calls
        # wait_time_per_step = 0.1

        for i, step_data in enumerate(manim_data):
            # Limit the number of steps animated
            if i >= MAX_STEPS_TO_ANIMATE:
                break
                
            print(f"Animating Manim step {i} (Sim step {step_data['step']})...")
            
            # --- Update Current State Dot ---
            current_obs = step_data["obs"]
            # Calculate cost at the actual current state (might differ slightly from surface)
            current_cost = cost_surface_func(current_obs[x_axis_state_idx], current_obs[y_axis_state_idx])
            current_point_coords = axes.c2p(
                current_obs[x_axis_state_idx], 
                current_obs[y_axis_state_idx], 
                current_cost # Use calculated cost for Z
            )
            # CORRECTED: Animate the existing dot's movement
            move_dot_anim = current_state_dot.animate.move_to(current_point_coords)

            # --- Update Predicted Trajectory Dots --- (Calculation Only)
            X_solution = step_data["X_solution"]
            new_prediction_group = VGroup()
            if X_solution is not None and X_solution.shape[1] > 0:
                for k in range(1, X_solution.shape[1]): 
                    pred_state = X_solution[:, k]
                    pred_cost = cost_surface_func(pred_state[x_axis_state_idx], pred_state[y_axis_state_idx])
                    pred_point_coords = axes.c2p(
                        pred_state[x_axis_state_idx], 
                        pred_state[y_axis_state_idx], 
                        pred_cost
                    )
                    alpha = 1.0 - (k / X_solution.shape[1]) * 0.7 
                    color = interpolate_color(YELLOW, ORANGE, k / X_solution.shape[1])
                    pred_dot = Dot3D(point=pred_point_coords, color=color, radius=0.04, fill_opacity=alpha)
                    new_prediction_group.add(pred_dot)
                
            # --- Update Step Label --- (Calculation Only)
            target_step_label = Text(f"Step: {step_data['step']}", font_size=24).move_to(step_label)
            
            # --- Play Animations in Sequence ---
            
            # 1. Fade out old predictions (Quickly)
            if len(last_prediction_group) > 0:
                self.play(FadeOut(last_prediction_group), run_time=forecast_fadeout_run_time)
            
            # 2. Animate new forecast appearing sequentially (Step 5)
            if len(new_prediction_group) > 0:
                forecast_anim = AnimationGroup(
                    *[FadeIn(dot) for dot in new_prediction_group],
                    lag_ratio=0.1 # Adjust lag for desired sequential effect
                )
                self.play(forecast_anim, run_time=forecast_anim_run_time)
            else:
                # If no new predictions, maybe wait briefly or do nothing?
                self.wait(0.1) # Small wait if there was no forecast animation

            # 3. Animate state dot movement and label update (Step 6)
            update_label_anim = Transform(step_label, target_step_label)
            self.play(
                move_dot_anim, 
                update_label_anim,
                run_time=state_update_run_time 
            )

            # Update for next iteration (Step 7 implied by loop)
            last_prediction_group = new_prediction_group # Keep track of current prediction dots

        # Keep the final frame for a moment
        self.wait(2)

        # --- Clean up and final wait ---
        # (Optional: add any final cleanup or text)
        print("Animation loop finished.")
        self.wait(5) # Keep final state visible 