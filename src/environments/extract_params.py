import gymnasium as gym
import json
import numpy as np
import mujoco
import argparse

# Import the registration function
# from src.environments.custom_swingup_env import register_unlimited_swingup_env # Old import
from src.environments.swing_up_envs.pendulum_su import register_pendulum_swing_up # New import


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
    with open(output_file, "w") as f:
        json.dump(params, f, indent=2)

    print(f"✅ Saved inverted pendulum parameters to: {output_file}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract parameters from InvertedPendulum env.")
    parser.add_argument("--env-id", type=str, default="InvertedPendulum-v5", help="Environment ID")
    parser.add_argument("--output-file-base", type=str, default="src/environments/inverted_pendulum_params", help="Base path for output JSON")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix to add to output filename (e.g., '_swingup')")
    parser.add_argument("--unlimit-pole", action="store_true", help="Widen pole angle joint limits in output")
    args = parser.parse_args()

    extract_inverted_pendulum_params(
        env_id=args.env_id,
        output_file_base=args.output_file_base,
        output_suffix=args.output_suffix,
        unlimit_pole=args.unlimit_pole
    )
