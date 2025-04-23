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


def load_double_pendulum_params(
    json_file="src/environments/double_pendulum_params.json",
    return_mapping=False # Flag to return the full mapping dict
):
    """
    Loads parameters for InvertedDoublePendulum from JSON.

    Returns:
        tuple: 
            - list[float] | None: Ordered list of numerical parameters.
            - float | None: Timestep dt.
            - dict | None: Joint limits dictionary (e.g., {'x': [-1, 1], ...}).
            - dict | None: The symbolic_params_mapping dict (only if return_mapping=True).
    """
    try:
        with open(json_file, "r") as f:
            all_params = json.load(f)
    except FileNotFoundError:
        print(f"Error: Parameter file not found at {json_file}")
        return (None,) * (4 if return_mapping else 3)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file}: {e}")
        return (None,) * (4 if return_mapping else 3)

    symbolic_params = all_params.get("symbolic_params_mapping")
    if symbolic_params is None:
        print(f"Error: 'symbolic_params_mapping' key not found in {json_file}")
        return (None,) * (4 if return_mapping else 3)
        
    joint_limits = symbolic_params.get("joint_limits")
    if joint_limits is None:
        print(f"Warning: 'joint_limits' key not found in symbolic_params_mapping.")
        # Return empty dict to avoid downstream errors, but signal issue
        joint_limits = {}
        
    # --- Define the exact order required by the CasADi function ---
    param_order = [
        'M', 'm1', 'm2', 'l1', 'l2', 'd1', 'd2', 'Icm1', 'Icm2', 'g',
        'b_slide', 'b_fric', 'b_joint1', 'b_joint2', 'gear'
    ]
    n_expected_params = len(param_order)
    ordered_param_list = []
    missing_keys = []
    for key in param_order:
        value = symbolic_params.get(key)
        if value is None: missing_keys.append(key)
        elif not isinstance(value, (int, float)):
            try: value = float(value)
            except (ValueError, TypeError): 
                print(f"Error converting param '{key}'='{value}'."); missing_keys.append(key); value = None
        if value is not None: ordered_param_list.append(value)
        
    if missing_keys: 
        print(f"Error: Missing params: {missing_keys}"); 
        return (None,) * (4 if return_mapping else 3)
    if len(ordered_param_list) != n_expected_params: 
        print(f"Error: Param count mismatch ({len(ordered_param_list)} vs {n_expected_params})."); 
        return (None,) * (4 if return_mapping else 3)
         
    dt = float(symbolic_params.get('dt', 0.01))

    print(f"Successfully extracted {len(ordered_param_list)} parameters, dt={dt}, and joint limits.")
    
    if return_mapping:
        return ordered_param_list, dt, joint_limits, symbolic_params
    else:
        return ordered_param_list, dt, joint_limits


if __name__ == "__main__":
    # Example usage / test
    print("--- Testing Parameter Loading --- ")
    # --- Single Pendulum --- 
    print("\nAttempting to load SINGLE pendulum params...")
    try:
        ip_params = load_pendulum_params("src/environments/pendulum_params.json")
    except FileNotFoundError:
        print(" -> Single Pendulum Parameters file not found.")
    except Exception as e:
        print(f" -> Error loading single pendulum params: {e}")

    # --- Double Pendulum --- 
    print("\nAttempting to load DOUBLE pendulum params...")
    try:
        # Test default return (list, dt, limits)
        dp_params_list, dp_dt, dp_limits = load_double_pendulum_params(
            "src/environments/double_pendulum_params.json", return_mapping=False
        )
        if dp_params_list is None:
            print(" -> Failed to load Double Pendulum parameters/dt/limits.")
        else:
            print(f" -> Loaded {len(dp_params_list)} params, dt={dp_dt}, limits={dp_limits}.")
        
        # Test returning mapping dict as well
        dp_params_list_2, dp_dt_2, dp_limits_2, dp_map = load_double_pendulum_params(
             "src/environments/double_pendulum_params.json", return_mapping=True
        )
        if dp_map is None:
             print(" -> Failed to load Double Pendulum mapping dict.")
        else:
             print(f" -> Successfully loaded mapping dict (contains {len(dp_map)} keys).")
             
    except Exception as e:
        print(f" -> Error loading double pendulum params: {e}")
        import traceback
        traceback.print_exc()
