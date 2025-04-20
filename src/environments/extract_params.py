import gymnasium as gym
import json
import numpy as np


def extract_inverted_pendulum_params(env_id="InvertedPendulum-v5",
                                     render_mode=None,
                                     output_file="src/environments/inverted_pendulum_params.json"):
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
    """
    # -------------------- Load environment --------------------
    env = gym.make(env_id, render_mode=render_mode)
    env.reset()
    model = env.unwrapped.model

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

    # -------------------- Package all into dict --------------------
    params = {
        "cart_mass": cart_mass,
        "pole_mass": pole_mass,
        "pole_inertia_about_y": pole_inertia_y,
        "pole_half_length": pole_half_length,
        "pole_length": pole_length,
        "gravity": 9.81,  # Assumed constant in most Gym environments
        "state_bounds_obs": state_bounds_obs,
        "joint_bounds": joint_bounds,
        "control_bounds": control_bounds
    }

    # -------------------- Save as JSON --------------------
    with open(output_file, "w") as f:
        json.dump(params, f, indent=2)

    print(f"✅ Saved inverted pendulum parameters (including bounds) to: {output_file}")

    env.close()


if __name__ == "__main__":
    # Example usage
    extract_inverted_pendulum_params(
        env_id="InvertedPendulum-v5",
        render_mode=None,
        output_file="src/environments/inverted_pendulum_params.json"
    )
