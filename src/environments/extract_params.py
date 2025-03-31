import gymnasium as gym
import json


def extract_inverted_pendulum_params(env_id="InvertedPendulum-v4",
                                     render_mode=None,
                                     output_file="inverted_pendulum_params.json"):
    """
    1) Loads the MuJoCo environment (e.g. InvertedPendulum-v4).
    2) Extracts relevant physical parameters for a 2D cart-pole model.
    3) Saves these parameters to a JSON file for later use.
    """
    # Create the environment
    env = gym.make(env_id, render_mode=render_mode)
    env.reset()

    # Access the MuJoCo model
    model = env.unwrapped.model

    # Identify cart / pole masses and inertia
    cart_mass = None
    pole_mass = None
    pole_inertia_y = None
    pole_half_length = None

    # Find cart & pole by body name
    for i in range(model.nbody):
        name = model.body(i).name
        mass = float(model.body_mass[i])  # convert from numpy to float
        inertia = model.body_inertia[i]   # [Ix, Iy, Iz]

        if name == "cart":
            cart_mass = mass
        elif name == "pole":
            pole_mass = mass
            # For a single hinge around y-axis, the relevant inertia might be inertia[1]
            pole_inertia_y = float(inertia[1])

    # Find the 'cpole' geom to get half-length
    for i in range(model.ngeom):
        geom_name = model.geom(i).name
        if geom_name == "cpole":
            # Typically the second entry is half-length (capsule shape)
            pole_half_length = float(model.geom_size[i][1])

    env.close()

    # Derive simpler 2D parameters
    if pole_half_length is not None:
        pole_length = 2.0 * pole_half_length
    else:
        pole_length = None

    # Build a dict of parameters
    params = {
        "cart_mass": cart_mass,
        "pole_mass": pole_mass,
        "pole_inertia_about_y": pole_inertia_y,
        "pole_half_length": pole_half_length,
        "pole_length": pole_length,
        "gravity": 9.81  # might as well store gravity here too
    }

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Saved Inverted Pendulum parameters to '{output_file}'")


if __name__ == "__main__":
    # Example usage
    extract_inverted_pendulum_params(
        env_id="InvertedPendulum-v4",
        render_mode="human",
        output_file="inverted_pendulum_params.json"
    )
