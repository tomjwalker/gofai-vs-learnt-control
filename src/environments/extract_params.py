import gymnasium as gym
import json
import numpy as np
import mujoco
import argparse
import math # For infinity checks

# Import the registration function
# from src.environments.custom_swingup_env import register_unlimited_swingup_env # Old import
from src.environments.swing_up_envs.pendulum_su import register_pendulum_swing_up # New import

# Placeholder for double pendulum registration (if needed later)
# from src.environments.swing_up_envs.double_pendulum_su import register_double_pendulum_swing_up

# --- Helper function to convert numpy types to python types for JSON --- 
def convert_to_python_types(data):
    if isinstance(data, dict):
        return {k: convert_to_python_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_python_types(item) for item in data)
    elif isinstance(data, np.ndarray):
        # Convert whole array to list, then process elements
        return convert_to_python_types(data.tolist())
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, (np.float16, np.float32, np.float64)):
        # Handle infinity specifically for JSON compatibility
        if math.isinf(data):
            return "Infinity" if data > 0 else "-Infinity"
        elif math.isnan(data):
            return None # Represent NaN as null in JSON
        else:
            return float(data)
    elif isinstance(data, (np.bool_)):
        return bool(data)
    elif isinstance(data, (np.void)):
        return None # Cannot serialize void type
    # Handle Python's infinity/NaN before they reach JSON dump
    elif isinstance(data, float):
        if math.isinf(data):
            return "Infinity" if data > 0 else "-Infinity"
        elif math.isnan(data):
            return None
        else:
            return data
    else:
        return data # Assume it's already JSON serializable

def extract_inverted_pendulum_params(env_id="InvertedPendulum-v5",
                                     render_mode=None,
                                     output_file_base="src/environments/inverted_pendulum_params", # Base name
                                     output_suffix="", # Suffix like _swingup
                                     unlimit_pole=False): # Flag to modify bounds
    """
    Extracts physical and control-related parameters from a Gymnasium MuJoCo environment
    (e.g., InvertedPendulum-v4), and saves them into a JSON file for later use in MPC or RL pipelines.

    What this does:
    ----------------
    1. Loads the environment.
    2. Extracts:
        - Cart mass
        - Pole mass
        - Pole inertia (around y-axis, for hinge)
        - Pole half-length (for geometry / COM)
        - Gravity
        - Action (control) limits
        - State (observation) limits
    3. Saves these as a structured JSON file for use in downstream modules.

    If unlimit_pole is True, modifies pole angle joint bounds before saving.
    Saves to output_file_base + output_suffix + .json
    """
    output_file = f"{output_file_base}{output_suffix}.json"
    # -------------------- Load environment --------------------
    env = gym.make(env_id, render_mode=render_mode)
    env.reset()
    model = env.unwrapped.model

    # --- DEBUGGING --- 
    # Moved outside the conditional block to always run
    print("--- DEBUG: Inspecting model loaded INITIALLY ---")
    print(f"Initial env_id used: {env_id}")
    print(f"Model njnt: {model.njnt}")
    jnt_names = [model.joint(i).name for i in range(model.njnt)]
    print(f"Model joint names: {jnt_names}")
    hinge_idx = jnt_names.index('hinge') if 'hinge' in jnt_names else -1
    print(f"Hinge index: {hinge_idx}")
    if hinge_idx != -1:
            print(f"Model hinge joint range (direct access): {model.jnt_range[hinge_idx]}")
    print("--- END DEBUG ---")
    # --- END DEBUGGING ---

    # -------------------- Extract body properties --------------------
    cart_mass = None
    pole_mass = None
    pole_inertia_y = None
    pole_half_length = None

    for i in range(model.nbody):
        name = model.body(i).name
        mass = float(model.body_mass[i])
        inertia = model.body_inertia[i]  # [Ix, Iy, Iz]

        if name == "cart":
            cart_mass = mass
        elif name == "pole":
            pole_mass = mass
            pole_inertia_y = float(inertia[1])  # Assuming hinge around y-axis

    # -------------------- Extract geom info --------------------
    for i in range(model.ngeom):
        geom_name = model.geom(i).name
        if geom_name == "cpole":
            pole_half_length = float(model.geom_size[i][1])  # Capsule half-length

    pole_length = 2.0 * pole_half_length if pole_half_length is not None else None

    # -------------------- Extract control and state limits --------------------
    # Control (action) limits
    control_lows = env.action_space.low.tolist()
    control_highs = env.action_space.high.tolist()
    control_bounds = list(zip(control_lows, control_highs))  # List of (min, max) tuples

    # State (observation) limits (informational, might be [-inf, inf])
    state_lows = env.observation_space.low.tolist()
    state_highs = env.observation_space.high.tolist()
    state_bounds_obs = list(zip(state_lows, state_highs))

    # --- Extract Actual Joint Limits from MuJoCo model --- 
    joint_bounds = {} # Store as {joint_name: [low, high]}
    if hasattr(model, 'jnt_range') and hasattr(model, 'jnt_qposadr'):
        for i in range(model.njnt):
            jnt_name = model.joint(i).name
            qpos_adr = model.jnt_qposadr[i] # Starting address in qpos
            # Assuming 1 DoF per joint for IP
            low, high = model.jnt_range[i]
            # Map common joint names to state indices/names if possible
            # This mapping might need adjustment for other envs
            state_name = None
            if jnt_name == 'slider': state_name = 'cart_pos' # Corresponds to qpos[0]
            elif jnt_name == 'hinge': state_name = 'pole_angle' # Corresponds to qpos[1]
            
            if state_name:
                 joint_bounds[state_name] = [float(low), float(high)]
            else:
                 joint_bounds[f"{jnt_name}_qpos{qpos_adr}"] = [float(low), float(high)] # Generic name
    else:
        print("Warning: Could not extract joint ranges from model.")

    # --- Optionally Modify Pole Angle Bounds --- 
    if unlimit_pole:
        # If unlimiting, ensure the custom env is registered and use it
        # register_unlimited_swingup_env() # Old function call
        register_pendulum_swing_up() # Call new function to register BOTH IDs
        env_id = 'Pendulum-SwingUpUnlimited-v0' # Use the correct unlimited ID
        print(f"--unlimit-pole specified. Using env_id: {env_id} for extraction.")
        # Re-create env with the correct ID (if not already done)
        # Note: This assumes the initial env_id wasn't already the unlimited one
        env.close() # Close the old env
        env = gym.make(env_id, render_mode=render_mode)
        env.reset()
        model = env.unwrapped.model
        # --- DEBUGGING --- 
        # The initial debug block can remain here if useful
        # ... (keep initial debug block) ...
        # --- END DEBUGGING ---
        # Re-extract bounds from the NEW model - NO LONGER NEEDED HERE
        # joint_bounds = {} # Reset dict
        # if hasattr(model, 'jnt_range') and hasattr(model, 'jnt_qposadr'):
        #     ...
        # else:
        #     print("Warning: Could not re-extract joint ranges from unlimited model.")
        
        # --- Get pre-extracted bounds from the env instance --- 
        if hasattr(env.unwrapped, 'extracted_joint_bounds'):
             joint_bounds = env.unwrapped.extracted_joint_bounds
             print(f"Read pre-extracted joint bounds from env object: {joint_bounds}")
        else:
             print("Error: Unlimited env instance missing 'extracted_joint_bounds' attribute.")
             # Fallback or raise error?
             joint_bounds = {} # Fallback to empty
        # ------------------------------------------------------

        # We no longer need to manually modify bounds here, 
        # as they come directly from the custom XML via the new env_id.
            
    # --- Extract Actuator Info (Gear Ratio) ---
    actuator_gear = {} # Store as {actuator_name: gear_ratio}
    if hasattr(model, 'actuator_gear') and hasattr(model, 'nu'): # nu is number of actuators
        for i in range(model.nu):
            act_name = model.actuator(i).name
            # Gear can be complex (matrix), assume scalar or take first element for simple motors
            gear_val = model.actuator_gear[i][0] if model.actuator_gear[i].size > 0 else 1.0
            actuator_gear[act_name] = float(gear_val)
    else:
        print("Warning: Could not extract actuator gear ratios from model.")

    # Extract body masses and inertias
    params = {
        "cart_mass": cart_mass,
        "pole_mass": pole_mass,
        "pole_inertia_about_y": pole_inertia_y,
        "pole_half_length": pole_half_length,
        "pole_length": pole_length,
        "gravity": 9.81,  # Assumed constant in most Gym environments
        "state_bounds_obs": state_bounds_obs,
        "joint_bounds": joint_bounds,
        "control_bounds": control_bounds,
        "actuator_gear": actuator_gear
    }

    # Extract body masses and inertias
    params['body_mass'] = model.body_mass.tolist()
    params['body_inertia'] = model.body_inertia.tolist()

    # Extract joint damping (assuming uniform damping for the hinge joint)
    # Find the hinge joint (assuming only one)
    # Corrected loop: Iterate using range(model.njnt)
    hinge_joint_indices = [i for i in range(model.njnt) 
                           if model.joint(i).type == mujoco.mjtJoint.mjJNT_HINGE]
    
    if hinge_joint_indices:
        # Use the first hinge joint found
        hinge_idx = hinge_joint_indices[0]
        # Damping is associated with degrees of freedom (DoF), find the DoF address
        dof_adr = model.jnt_dofadr[hinge_idx]
        params['joint_damping'] = model.dof_damping[dof_adr].item() # Use dof_adr
        print(f"Extracted hinge damping: {params['joint_damping']}")
    else:
        params['joint_damping'] = 1.0 # Default if no hinge found
        print("Warning: No hinge joint found for damping extraction. Using default.")

    # Extract slider joint damping
    slider_joint_indices = [i for i in range(model.njnt) 
                            if model.joint(i).type == mujoco.mjtJoint.mjJNT_SLIDE]
    if slider_joint_indices:
        slider_idx = slider_joint_indices[0]
        dof_adr = model.jnt_dofadr[slider_idx]
        params['slider_damping'] = model.dof_damping[dof_adr].item()
        print(f"Extracted slider damping: {params['slider_damping']}")
    else:
        params['slider_damping'] = 0.0 # Default to 0 if no slider found
        print("Warning: No slider joint found for damping extraction. Using default 0.")

    # Extract cart sliding friction
    # Find the 'cart' geom or the 'plane' geom it interacts with
    cart_geom_id = -1
    plane_geom_id = -1
    for i in range(model.ngeom):
        geom_name = model.geom(i).name
        if 'cart' in geom_name:
            cart_geom_id = i
        if 'plane' in geom_name or 'track' in geom_name or 'floor' in geom_name: # Common names for the surface
            plane_geom_id = i
            break # Assume first plane found is the relevant one
            
    # Prefer friction from the plane if found, otherwise use cart's
    # MuJoCo friction is [sliding, torsional, rolling]
    if plane_geom_id != -1:
        params['cart_friction'] = model.geom_friction[plane_geom_id][0].item()
        print(f"Extracted friction from geom '{model.geom(plane_geom_id).name}'")
    elif cart_geom_id != -1:
        params['cart_friction'] = model.geom_friction[cart_geom_id][0].item()
        print(f"Warning: Plane/track geom not found. Using friction from geom '{model.geom(cart_geom_id).name}'")
    else:
        params['cart_friction'] = 0.1 # Fallback default if neither found
        print("Warning: Cart and plane/track geom not found for friction extraction. Using default.")

    # -------------------- Save as JSON --------------------
    params_serializable = convert_to_python_types(params)
    with open(output_file, "w") as f:
        json.dump(params_serializable, f, indent=2)

    print(f"✅ Saved inverted pendulum parameters to: {output_file}")

    env.close()


# --- Function for Double Pendulum ---
def extract_inverted_double_pendulum_params(
    env_id="InvertedDoublePendulum-v5",
    render_mode=None,
    output_file="src/environments/double_pendulum_params.json"
):
    """
    Extracts physical and control parameters from InvertedDoublePendulum-v5.
    Saves parameters relevant for the symbolic/CasADi model.
    """
    print(f"--- Extracting DOUBLE Pendulum Params ({env_id}) ---")

    try:
        env = gym.make(env_id, render_mode=render_mode)
        env.reset()
        model = env.unwrapped.model
    except Exception as e:
        print(f"Error creating environment '{env_id}': {e}")
        return

    params_out = {
        "env_id": env_id,
    }

    # Expected names (based on XML)
    CART_BODY = "cart"
    POLE1_BODY = "pole"
    POLE2_BODY = "pole2"
    SLIDER_JOINT = "slider"
    HINGE1_JOINT = "hinge"
    HINGE2_JOINT = "hinge2"
    POLE1_GEOM = "cpole"
    POLE2_GEOM = "cpole2"
    PLANE_GEOM = "floor"
    SLIDER_ACTUATOR_JOINT = "slider" # The joint controlled by the actuator

    # --- Extract Body Properties --- 
    body_params = {}
    body_names = [model.body(i).name for i in range(model.nbody)]
    print(f"Model Bodies: {body_names}")
    for i in range(model.nbody):
        name = model.body(i).name
        body_params[name] = {
            "mass": model.body_mass[i], # Keep as numpy type initially
            "inertia": model.body_inertia[i] # Keep as numpy type initially
        }
    params_out["bodies"] = body_params

    # --- Extract Geom Properties --- 
    geom_params = {}
    geom_names = [model.geom(i).name for i in range(model.ngeom)]
    print(f"Model Geoms: {geom_names}")
    for i in range(model.ngeom):
        name = model.geom(i).name
        geom_params[name] = {
            "size": model.geom_size[i],
            "type": str(mujoco.mjtGeom(model.geom_type[i])),
            "friction": model.geom_friction[i]
        }
    params_out["geoms"] = geom_params

    # --- Extract Joint Properties --- 
    joint_params = {}
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    print(f"Model Joints: {joint_names}")
    if hasattr(model, 'jnt_range') and hasattr(model, 'jnt_qposadr') and hasattr(model, 'jnt_dofadr'):
        for i in range(model.njnt):
            name = model.joint(i).name
            joint_params[name] = {
                "range": model.jnt_range[i],
                "damping": model.dof_damping[model.jnt_dofadr[i]],
                "qpos_address": model.jnt_qposadr[i],
                "dof_address": model.jnt_dofadr[i],
                "type": str(mujoco.mjtJoint(model.jnt_type[i]))
            }
    else: print("Warning: Could not extract detailed joint ranges/damping.")
    params_out["joints"] = joint_params

    # --- Extract Actuator Properties --- 
    actuator_params = {}
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    actuator_joint_ids = [model.actuator_trnid[i, 0] for i in range(model.nu)] # Get joint ID for each actuator
    actuator_joint_names = [model.joint(jid).name for jid in actuator_joint_ids]
    print(f"Model Actuators: {list(zip(actuator_names, actuator_joint_names))}") # Show actuator and its joint

    if hasattr(model, 'actuator_gear') and hasattr(model, 'actuator_ctrlrange'):
        for i in range(model.nu):
            name = model.actuator(i).name
            gear_val = model.actuator_gear[i][0] if model.actuator_gear[i].size > 0 else 1.0
            actuator_params[name] = {
                "gear": gear_val,
                "ctrl_range": model.actuator_ctrlrange[i],
                "joint_controlled": model.joint(model.actuator_trnid[i, 0]).name # Store name of controlled joint
            }
    else: print("Warning: Could not extract actuator gear/ctrlrange.")
    params_out["actuators"] = actuator_params

    # --- Add General Properties --- 
    params_out["gravity"] = model.opt.gravity
    params_out["timestep"] = model.opt.timestep

    # --- Extract Control/Observation Limits from Gym Space --- 
    # Process these directly to handle infinity before general conversion
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    params_out["gym_action_space"] = {
        "low": env.action_space.low,
        "high": env.action_space.high
    }
    # Convert inf/-inf to strings for JSON compatibility
    params_out["gym_observation_space"] = {
        "low": ["-Infinity" if np.isneginf(x) else ("Infinity" if np.isposinf(x) else x) for x in obs_low],
        "high": ["-Infinity" if np.isneginf(x) else ("Infinity" if np.isposinf(x) else x) for x in obs_high]
    }

    # --- Map to Symbolic Model Parameters --- 
    symbolic_params = {}
    try:
        symbolic_params['M'] = body_params[CART_BODY]['mass']
        symbolic_params['m1'] = body_params[POLE1_BODY]['mass']
        symbolic_params['m2'] = body_params[POLE2_BODY]['mass']

        pole1_size = geom_params[POLE1_GEOM]['size']
        pole2_size = geom_params[POLE2_GEOM]['size']
        symbolic_params['l1'] = 2.0 * pole1_size[1]
        symbolic_params['l2'] = 2.0 * pole2_size[1]
        symbolic_params['d1'] = pole1_size[1]
        symbolic_params['d2'] = pole2_size[1]

        symbolic_params['Icm1'] = body_params[POLE1_BODY]['inertia'][1]
        symbolic_params['Icm2'] = body_params[POLE2_BODY]['inertia'][1]

        symbolic_params['g'] = -model.opt.gravity[2]

        symbolic_params['b_slide'] = joint_params[SLIDER_JOINT]['damping']
        symbolic_params['b_joint1'] = joint_params[HINGE1_JOINT]['damping']
        symbolic_params['b_joint2'] = joint_params[HINGE2_JOINT]['damping']

        plane_name = next((name for name in geom_names if PLANE_GEOM in name), None)
        if plane_name:
            symbolic_params['b_fric'] = geom_params[plane_name]['friction'][0]
        else:
            print(f"Warning: Geom '{PLANE_GEOM}' not found for friction. Defaulting b_fric to 0.")
            symbolic_params['b_fric'] = 0.0

        # Find actuator controlling the specified slider JOINT
        slider_actuator = None
        for act_name, act_data in actuator_params.items():
            if act_data.get("joint_controlled") == SLIDER_ACTUATOR_JOINT:
                slider_actuator = act_data
                print(f"Found actuator '{act_name}' controlling joint '{SLIDER_ACTUATOR_JOINT}'")
                break
        
        if slider_actuator:
            symbolic_params['gear'] = slider_actuator['gear']
            symbolic_params['control_limits_actuator'] = slider_actuator['ctrl_range']
        else:
            print(f"Warning: Actuator controlling joint '{SLIDER_ACTUATOR_JOINT}' not found. Defaulting gear to 1.0.")
            symbolic_params['gear'] = 1.0

        # Use Gym action space limits as primary control limits for symbolic model
        # Make sure this is extracted correctly before converting types
        gym_action_low = env.action_space.low
        gym_action_high = env.action_space.high
        symbolic_params['control_limits_gym'] = [gym_action_low[0], gym_action_high[0]]

        symbolic_params['joint_limits'] = {
            'x': joint_params[SLIDER_JOINT]['range'],
            'theta1': joint_params[HINGE1_JOINT]['range'],
            'theta2': joint_params[HINGE2_JOINT]['range']
        }
        
        # Add timestep directly into symbolic mapping for convenience
        symbolic_params['dt'] = model.opt.timestep

    except KeyError as e:
        print(f"\nError mapping parameter: Key '{e}' not found in extracted MuJoCo params.")
        print(" -> Check expected names (CART_BODY, POLE1_BODY, etc.) against printed names above.")
        print(" -> Aborting symbolic mapping.")
        symbolic_params = {} # Clear potentially incomplete mapping
    except Exception as e:
        print(f"\nError during symbolic parameter mapping: {e}")
        import traceback
        traceback.print_exc()
        symbolic_params = {} # Clear potentially incomplete mapping

    params_out["symbolic_params_mapping"] = symbolic_params

    # --- Convert all collected data to Python types for JSON --- 
    print("Converting extracted parameters to JSON-serializable types...")
    params_out_serializable = convert_to_python_types(params_out)
    print("Conversion complete.")

    # --- Save as JSON --- 
    try:
        with open(output_file, "w") as f:
            json.dump(params_out_serializable, f, indent=2)
        print(f"✅ Saved DOUBLE pendulum parameters to: {output_file}")
    except IOError as e:
        print(f"Error writing JSON file '{output_file}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred saving JSON: {e}")
        import traceback 
        traceback.print_exc() # Print traceback for JSON errors

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract parameters from MuJoCo environments.")
    parser.add_argument("--env-type", type=str, default="single", choices=["single", "double"], help="Type of pendulum environment.")
    parser.add_argument("--env-id", type=str, default=None, help="Specific Environment ID (overrides default for type).")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON file path (overrides default for type).")
    parser.add_argument("--unlimit-pole", action="store_true", help="(Single pendulum only) Widen pole angle joint limits.")
    args = parser.parse_args()

    if args.env_type == "single":
        env_id = args.env_id if args.env_id else "InvertedPendulum-v5"
        output_file = args.output_file if args.output_file else "src/environments/pendulum_params.json"
        # Determine suffix based on unlimit_pole
        output_suffix = "_swingup" if args.unlimit_pole else ""
        output_file_base = output_file.replace(".json", "").replace("_swingup", "") # Get base name
        
        extract_inverted_pendulum_params(
            env_id=env_id,
            output_file_base=output_file_base,
            output_suffix=output_suffix,
            unlimit_pole=args.unlimit_pole
        )
    elif args.env_type == "double":
        env_id = args.env_id if args.env_id else "InvertedDoublePendulum-v5"
        output_file = args.output_file if args.output_file else "src/environments/double_pendulum_params.json"
        if args.unlimit_pole:
            print("Warning: --unlimit-pole currently only affects single pendulum extraction.")
        extract_inverted_double_pendulum_params(
            env_id=env_id,
            output_file=output_file
        )
