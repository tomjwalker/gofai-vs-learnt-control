# sympy_models/utils/sympy_to_casadi.py

import sympy as sp
import casadi as ca
import numpy as np # Added for example usage if needed
from typing import List, Tuple, Dict, Any
import os # For path operations
import re # For replacing math functions
import sympy.printing.pycode as pycode # For python code generation

# Helper function for recursive SymPy to CasADi expression conversion
def _sympy_to_casadi_sx(expr: sp.Expr, mapping: Dict[sp.Symbol, ca.SX]) -> ca.SX:
    """
    Recursively converts a SymPy expression to a CasADi SX expression.

    Args:
        expr: The SymPy expression to convert.
        mapping: A dictionary mapping SymPy symbols to their CasADi SX equivalents.

    Returns:
        The equivalent CasADi SX expression.

    Raises:
        NotImplementedError: If an unsupported SymPy expression type is encountered.
        ValueError: If a SymPy symbol in the expression is not found in the mapping.
    """
    if isinstance(expr, sp.Symbol):
        # If it's a symbol we defined (state, control, param), look it up
        if expr in mapping:
            return mapping[expr]
        else:
            # This case should generally not be hit if the mapping is constructed correctly
            # from all input symbols, but raise an error if it does occur.
            raise ValueError(f"SymPy symbol '{expr}' not found in the mapping to CasADi symbols.")
            # Alternative (less safe): return ca.SX.sym(str(expr)) 
    elif isinstance(expr, (sp.Number, sp.Integer, sp.Float)):
        # If it's a numerical value
        return ca.SX(float(expr))
    elif isinstance(expr, sp.Add):
        # Addition: Recursively convert args and sum them using CasADi's '+'
        casadi_expr = ca.SX(0)
        for arg in expr.args:
            casadi_expr += _sympy_to_casadi_sx(arg, mapping)
        return casadi_expr
    elif isinstance(expr, sp.Mul):
        # Multiplication: Recursively convert args and multiply them using CasADi's '*'
        casadi_expr = ca.SX(1)
        for arg in expr.args:
            casadi_expr *= _sympy_to_casadi_sx(arg, mapping)
        return casadi_expr
    elif isinstance(expr, sp.Pow):
        # Power: Recursively convert base and exponent
        base, exp = expr.args
        casadi_base = _sympy_to_casadi_sx(base, mapping)
        # CasADi handles numeric exponents directly in __pow__
        if isinstance(exp, sp.Number):
            casadi_exp = float(exp)
        else:
            # Recursively convert symbolic exponent if needed
            casadi_exp = _sympy_to_casadi_sx(exp, mapping)
        return casadi_base ** casadi_exp
    elif isinstance(expr, sp.sin):
        # Sine function
        casadi_arg = _sympy_to_casadi_sx(expr.args[0], mapping)
        return ca.sin(casadi_arg)
    elif isinstance(expr, sp.cos):
        # Cosine function
        casadi_arg = _sympy_to_casadi_sx(expr.args[0], mapping)
        return ca.cos(casadi_arg)
    elif isinstance(expr, sp.tan):
        # Tangent function
        casadi_arg = _sympy_to_casadi_sx(expr.args[0], mapping)
        return ca.tan(casadi_arg)
    # Add other common functions as needed (e.g., asin, acos, atan, exp, log)
    # elif isinstance(expr, sp.exp):
    #     casadi_arg = _sympy_to_casadi_sx(expr.args[0], mapping)
    #     return ca.exp(casadi_arg)
    # elif isinstance(expr, sp.log):
    #     casadi_arg = _sympy_to_casadi_sx(expr.args[0], mapping)
    #     return ca.log(casadi_arg)
    else:
        # If we encounter an unknown type, raise an error
        raise NotImplementedError(f"SymPy expression type '{type(expr)}' (expr: {expr}) is not supported for CasADi SX conversion.")


def generate_casadi_function(
    q: sp.Matrix, 
    q_d: sp.Matrix, 
    q_dd_exprs: sp.Matrix, 
    params_sym: List[sp.Symbol],
    control_sym: List[sp.Symbol],
    func_name: str = "symbolic_dynamics_ode",
    cse: bool = False # Note: CSE is not applied in this direct conversion path yet
) -> Tuple[ca.Function, List[str], List[str], List[str]]:
    """
    Converts SymPy expressions for system dynamics into a CasADi function
    using direct expression translation.

    Args:
        q: SymPy Matrix of generalized coordinates (e.g., [x(t), theta1(t), theta2(t)]).
        q_d: SymPy Matrix of generalized velocities (e.g., [Derivative(x(t),t), ...]).
        q_dd_exprs: SymPy Matrix containing the symbolic expressions for accelerations.
        params_sym: List of SymPy symbols representing system parameters.
        control_sym: List of SymPy symbols representing control inputs (e.g., [U]).
        func_name: Name for the generated CasADi function.
        cse: If True, apply Common Subexpression Elimination (Currently *not* implemented
             in this direct conversion pathway).

    Returns:
        Tuple containing:
        - casadi_func: The generated CasADi function: f(state, control, params) -> state_dot.
        - state_var_names: List of state variable names (e.g., ['x', 'theta1', 'x_d', 'theta1_d']).
        - control_var_names: List of control variable names.
        - param_var_names: List of parameter variable names.
    """
    print(f"--- Generating CasADi function '{func_name}' via direct translation ---")

    n_q = q.shape[0]
    assert q_d.shape == (n_q, 1), "q_d shape mismatch"
    assert q_dd_exprs.shape == (n_q, 1), "q_dd_exprs shape mismatch"
    n_u = len(control_sym)
    n_p = len(params_sym)
    n_state = 2 * n_q

    # 1. Define plain SymPy symbols for state components to use for substitution
    state_var_names = [s.func.__name__ for s in q] + [f"{s.func.__name__}_d" for s in q]
    state_symbols_plain = sp.symbols(state_var_names)
    if not isinstance(state_symbols_plain, (list, tuple)): 
        state_symbols_plain = (state_symbols_plain,) # Ensure tuple/list
    
    q_sym_plain = sp.Matrix(state_symbols_plain[:n_q])
    q_d_sym_plain = sp.Matrix(state_symbols_plain[n_q:])

    # 2. Create substitution dictionary: {Function/Derivative: Plain Symbol}
    sub_dict = {}
    time_sym = sp.symbols('t') # Assume 't' is the time variable in functions/derivatives
    for i in range(n_q):
        sub_dict[q[i]] = q_sym_plain[i]
        sub_dict[q_d[i]] = q_d_sym_plain[i]
        # Add substitution for second derivatives if they appear (e.g., in friction terms)
        sub_dict[q_d[i].diff(time_sym)] = sp.symbols(f"{state_var_names[i]}_dd") # Placeholder name

    # Define the state derivative vector expression using original functions/derivatives
    state_dot_exprs_orig = q_d.col_join(q_dd_exprs)
    assert state_dot_exprs_orig.shape == (n_state, 1), "State derivative vector shape mismatch"

    # 3. Substitute plain symbols into the derivative expression
    print("Substituting plain symbols into expressions...")
    state_dot_exprs_subs = state_dot_exprs_orig.subs(sub_dict)

    # Check for remaining functions/derivatives after substitution (should be none ideally)
    remaining_funcs = state_dot_exprs_subs.atoms(sp.Function)
    remaining_derivs = state_dot_exprs_subs.atoms(sp.Derivative)
    if remaining_funcs or remaining_derivs:
        print(f"Warning: Substitution might be incomplete.")
        print(f"  Remaining functions: {remaining_funcs}")
        print(f"  Remaining derivatives: {remaining_derivs}")

    # 4. Define CasADi symbolic inputs
    state_casadi = ca.SX.sym('state', n_state)
    control_casadi = ca.SX.sym('control', n_u)
    params_casadi = ca.SX.sym('params', n_p)

    # 5. Create the mapping from ALL SymPy input symbols to CasADi SX symbols
    sym_to_casadi_map = {}
    # Map state symbols
    for i, sym in enumerate(state_symbols_plain):
        sym_to_casadi_map[sym] = state_casadi[i]
    # Map control symbols
    control_var_names = []
    for i, sym in enumerate(control_sym):
        sym_to_casadi_map[sym] = control_casadi[i]
        control_var_names.append(str(sym))
    # Map parameter symbols
    param_var_names = []
    for i, sym in enumerate(params_sym):
        sym_to_casadi_map[sym] = params_casadi[i]
        param_var_names.append(str(sym))

    # 6. Convert each SymPy expression in the state derivative vector to CasADi SX
    print("Translating SymPy expressions to CasADi SX...")
    state_dot_casadi_list = []
    for i in range(n_state):
        sympy_expr = state_dot_exprs_subs[i]
        try:
            casadi_expr = _sympy_to_casadi_sx(sympy_expr, sym_to_casadi_map)
            state_dot_casadi_list.append(casadi_expr)
        except (NotImplementedError, ValueError) as e:
            print(f"Error translating expression for state_dot[{i}]: {sympy_expr}")
            print(f"Error details: {e}")
            raise RuntimeError("Failed to translate SymPy expression to CasADi.") from e

    # Combine into a single CasADi vector
    state_dot_casadi = ca.vertcat(*state_dot_casadi_list)

    # 7. Create the final CasADi function
    casadi_func = ca.Function(
        func_name,
        [state_casadi, control_casadi, params_casadi],
        [state_dot_casadi],
        ['state', 'control', 'params'], # Input names
        ['state_dot'] # Output names
    )
    print(f"CasADi function '{func_name}' created successfully.")

    # State variable names were determined earlier
    return casadi_func, state_var_names, control_var_names, param_var_names


# --- New Function for Python File Generation ---
def generate_casadi_python_file(
    q: sp.Matrix,
    q_d: sp.Matrix,
    accel_eqs_vec: sp.Matrix,
    control_sym: List[sp.Symbol],
    params_sym: List[sp.Symbol],
    target_py_file: str,
    generation_info: str = "Generated by sympy_models/sympy_inverted_double_pendulum.py"
) -> None:
    """
    Generates a Python file containing a CasADi dynamics function.

    Uses sympy.printing.pycode to translate symbolic expressions into
    executable Python/CasADi code.

    Args:
        q: SymPy Matrix of generalized coordinates (e.g., [x(t), theta1(t)]).
        q_d: SymPy Matrix of generalized velocities (e.g., [Derivative(x(t),t)]).
        accel_eqs_vec: SymPy Matrix of the solved acceleration expressions.
        control_sym: List of SymPy symbols for control inputs.
        params_sym: List of SymPy symbols for system parameters.
        target_py_file: Full path to the output Python file.
        generation_info: A comment string to include at the top of the file.
    """
    print(f"--- Generating Python Dynamics File: {target_py_file} ---")
    os.makedirs(os.path.dirname(target_py_file), exist_ok=True)

    n_q = q.shape[0]

    # Define plain symbols for substitution
    state_var_names = [s.func.__name__ for s in q] + [f"{s.func.__name__}_d" for s in q]
    state_symbols_plain = sp.symbols(state_var_names)
    if not isinstance(state_symbols_plain, (list, tuple)):
        state_symbols_plain = (state_symbols_plain,)
    q_sym_plain = sp.Matrix(state_symbols_plain[:n_q])
    q_d_sym_plain = sp.Matrix(state_symbols_plain[n_q:])

    # Create substitution dictionary
    sub_dict = {}
    time_sym = sp.symbols('t')
    for i in range(n_q):
        sub_dict[q[i]] = q_sym_plain[i]
        sub_dict[q_d[i]] = q_d_sym_plain[i]
        sub_dict[q_d[i].diff(time_sym)] = sp.symbols(f"{state_var_names[i]}_dd_placeholder")

    # Substitute plain symbols into acceleration expressions
    accel_exprs_subs = accel_eqs_vec.subs(sub_dict)

    # Get variable names in the correct order
    state_names = [str(s) for s in state_symbols_plain]
    control_names = [str(s) for s in control_sym]
    param_names = [str(s) for s in params_sym]
    n_state = len(state_names)
    n_control = len(control_names)
    n_params = len(param_names)

    # --- Build the Python code string ---
    # File Header
    py_code_string = f"""# {os.path.basename(target_py_file)}
# {generation_info}

import casadi as ca
import numpy as np

def get_dynamics_function():
    \"\"\"Creates and returns a CasADi Function for the derived dynamics.\"\"\"

    # --- Define Symbolic Variables ---
    state = ca.SX.sym('state', {n_state})
    control = ca.SX.sym('control', {n_control})
    params = ca.SX.sym('params', {n_params})

    # --- Unpack Variables ---
    # State variables ({len(state_names)})
"""
    for i, name in enumerate(state_names):
        py_code_string += f"    {name} = state[{i}]\n"
    py_code_string += f"    # Control variables ({len(control_names)})\n"
    for i, name in enumerate(control_names):
        py_code_string += f"    {name} = control[{i}]\n"
    py_code_string += f"    # Parameters ({len(param_names)})\n"
    for i, name in enumerate(param_names):
        py_code_string += f"    {name} = params[{i}]\n"

    py_code_string += "\n    # --- Define State Derivatives ---\n"

    # Velocity derivatives
    vel_deriv_strs = []
    for i in range(n_q):
        vel_deriv_strs.append(state_names[n_q + i])
        py_code_string += f"    {state_names[i]}_dot = {state_names[n_q + i]}\n"

    # Acceleration derivatives
    accel_deriv_names = [f"{state_names[i]}_ddot" for i in range(n_q)]
    accel_pycode_strs = []
    for i in range(n_q):
        accel_expr = accel_exprs_subs[i]
        raw_pycode = pycode(accel_expr)
        # Replace math functions with ca.*
        casadi_pycode = re.sub(r'math\.(\w+)\(', r'ca.\1(', raw_pycode)
        # Ensure powers are CasADi compatible (usually fine with **)
        accel_pycode_strs.append(casadi_pycode)
        py_code_string += f"    {accel_deriv_names[i]} = {casadi_pycode}\n"

    # Assemble state derivative vector
    state_dot_elements = vel_deriv_strs + accel_deriv_names
    py_code_string += "\n    # Assemble state derivative vector\n"
    py_code_string += f"    state_dot = ca.vertcat({', '.join(state_dot_elements)})\n"

    # Create and return the CasADi function
    py_code_string += "\n    # --- Create CasADi Function ---\n"
    py_code_string += f"""    dynamics_func = ca.Function(
        '{os.path.splitext(os.path.basename(target_py_file))[0]}_ode',
        [state, control, params],
        [state_dot],
        {['state', 'control', 'params']},
        {['state_dot']}
    )

    return dynamics_func
"""

    # Add main block for testing
    py_code_string += f"""

if __name__ == '__main__':
    print("Testing the generated CasADi function...")
    dynamics = get_dynamics_function()
    print("Function created:", dynamics)

    n_state = {n_state}
    n_control = {n_control}
    n_params = {n_params}

    # Example state (adjust as needed)
    state_val = np.zeros(n_state)
    # Example control
    control_val = np.zeros(n_control)
    # Example parameters (MUST match order defined in generation script)
    param_name_list_str = "Parameter order: {param_names}"
    print(param_name_list_str)
    # Provide a default array based on expected length
    param_val = np.ones(n_params) 
    # You SHOULD replace this with actual default/test values
    print(f"Using placeholder parameters: {{param_val}}")

    if len(param_val) != n_params:
        raise ValueError(f"Incorrect number of parameters provided. Expected {n_params}, got {{len(param_val)}}.")

    try:
        state_dot_val = dynamics(state_val, control_val, param_val)
        print("\nState vector (example):", state_val)
        print("Control vector (example):", control_val)
        print("Calculated state_dot:", state_dot_val)
    except Exception as e:
        print("\nError during function evaluation:", e)
        import traceback
        traceback.print_exc()

"""

    # Write the generated code to the file
    try:
        with open(target_py_file, 'w') as f:
            f.write(py_code_string)
        print(f"Successfully generated Python dynamics file: {target_py_file}")
    except IOError as e:
        print(f"Error writing file {target_py_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during file writing: {e}")


# Example Usage (Placeholder) - Needs update if structure changes
if __name__ == '__main__':
    # Define SymPy symbols for example
    t = sp.symbols('t')
    x_t, th_t = sp.Function('x')(t), sp.Function('theta')(t)
    x_d_t, th_d_t = x_t.diff(t), th_t.diff(t)
    m, l, g, U_sym = sp.symbols('m l g U') # Match control name U

    q_test = sp.Matrix([x_t, th_t])
    q_d_test = sp.Matrix([x_d_t, th_d_t])
    
    # Dummy acceleration expressions (in terms of functions/derivatives)
    x_ddot_expr = U_sym / m + g * sp.sin(th_t)
    th_ddot_expr = -g / l * sp.cos(th_t) - U_sym / (m * l) * sp.sin(th_t)
    q_dd_test = sp.Matrix([x_ddot_expr, th_ddot_expr])
    
    params_test = [m, l, g]
    control_test = [U_sym]

    try:
        # Use the new function name
        casadi_f, s_names, u_names, p_names = generate_casadi_function(
            q_test, q_d_test, q_dd_test, params_test, control_test, 
            func_name="example_pendulum_ode"
        )
        print("\nGenerated CasADi function:")
        print(casadi_f)
        print("State Variables:", s_names)
        print("Control Variables:", u_names)
        print("Parameter Variables:", p_names)

        # Test evaluation (requires numerical values and numpy)
        state_val = np.array([0.1, 0.2, 0.3, 0.4]) # x, th, x_d, th_d
        control_val = np.array([1.0]) # U
        param_val = np.array([1.0, 0.5, 9.81]) # m, l, g

        state_dot_val = casadi_f(state_val, control_val, param_val)
        print("\nEvaluated state_dot:", state_dot_val)

        # Test the new python file generation
        target_test_file = "./test_generated_dynamics.py"
        generate_casadi_python_file(
            q_test, q_d_test, q_dd_test, control_test, params_test,
            target_test_file,
            generation_info="Generated by sympy_to_casadi.py example"
        )
        # Optional: Add code to import and run the test file
        if os.path.exists(target_test_file):
             print(f"\nTest file {target_test_file} generated. Run it manually to test.")
             # import importlib
             # spec = importlib.util.spec_from_file_location("test_module", target_test_file)
             # test_module = importlib.util.module_from_spec(spec)
             # spec.loader.exec_module(test_module)
             # print("Successfully imported and ran test file.")
             # os.remove(target_test_file) # Clean up test file

    except Exception as e:
        import traceback
        print(f"\nError in example usage: {e}")
        traceback.print_exc() 