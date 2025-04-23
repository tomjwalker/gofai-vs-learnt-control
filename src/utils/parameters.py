import json
import numpy as np


def load_pendulum_params(json_file="pendulum_params.json"):
    """
    Loads the environment parameters for the InvertedPendulum
    that were previously saved to JSON.
    Returns them as a Python dictionary, possibly with
    additional interpretation or checks.

    :param json_file: Path to the JSON file
    :return: Dictionary containing {cart_mass, pole_mass, pole_length, ...}
    """
    with open(json_file, "r") as f:
        params = json.load(f)

    # Optional: Provide some fallback or computed fields
    # For instance, if 'pole_length' wasn't saved, we can compute from 'pole_half_length'
    if params.get("pole_length") is None and params.get("pole_half_length") is not None:
        params["pole_length"] = 2.0 * params["pole_half_length"]

    # Additional checks or logging
    print(f"Loaded Inverted Pendulum parameters from '{json_file}':")
    for k, v in params.items():
        print(f"  {k} = {v}")

    # Return the dictionary for further usage
    return params


if __name__ == "__main__":
    # Example usage / test
    # Assuming the script is run from the project root or parameters.py location
    # Try loading the default file first
    try:
        # Update path
        ip_params = load_pendulum_params("src/environments/pendulum_params.json")
        print("Default Pendulum Parameters:")
        for key, value in ip_params.items():
            print(f"  {key}: {value}")
    except FileNotFoundError:
        print("Default Pendulum Parameters file not found. Please check the path.")
