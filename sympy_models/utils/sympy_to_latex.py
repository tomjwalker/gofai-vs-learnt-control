# sympy_models/utils/sympy_to_latex.py

import sympy as sp
from sympy import sin, cos, Matrix
from pylatex.utils import NoEscape
from pylatex.base_classes import Environment

def format_vector_name(name):
    """Helper to format vector names consistently"""
    return NoEscape(r'\mathbf{' + name + r'}')

def sympy_to_latex(expr, use_dot_notation=False):
    """ Convert SymPy expression to bigger, parenthesised LaTeX matrix. """
    # Configure SymPy's latex printer for better vector/matrix output
    latex_str = sp.latex(expr, mode='inline', mat_str='matrix', mat_delim='')
    # Insert \displaystyle:
    latex_str = r'\displaystyle ' + latex_str

    # Dot notation replacements
    if use_dot_notation:
       latex_str = latex_str.replace(r'\frac{d}{d t} x{\left(t \right)}', r'\dot{x}')
       latex_str = latex_str.replace(r'\frac{d}{d t} \theta_{1}{\left(t \right)}', r'\dot{\theta}_{1}')
       latex_str = latex_str.replace(r'\frac{d}{d t} \theta_{2}{\left(t \right)}', r'\dot{\theta}_{2}')
       latex_str = latex_str.replace(r'\frac{d^{2}}{d t^{2}} x{\left(t \right)}', r'\ddot{x}')
       latex_str = latex_str.replace(r'\frac{d^{2}}{d t^{2}} \theta_{1}{\left(t \right)}', r'\ddot{\theta}_{1}')
       latex_str = latex_str.replace(r'\frac{d^{2}}{d t^{2}} \theta_{2}{\left(t \right)}', r'\ddot{\theta}_{2}')

    # Shorthand replacements for cleaner equations
    replacements = [
        # Remove explicit time dependencies
        (r'x{\left(t \right)}', 'x'),
        (r'\theta_{1}{\left(t \right)}', r'\theta_1'),
        (r'\theta_{2}{\left(t \right)}', r'\theta_2'),
        # Shorthands for trig functions
        (r'\cos{\left(\theta_{1} \right)}', r'c_1'),
        (r'\cos{\left(\theta_{2} \right)}', r'c_2'),
        (r'\sin{\left(\theta_{1} \right)}', r's_1'),
        (r'\sin{\left(\theta_{2} \right)}', r's_2'),
        # Clean up any leftover {\left and \right} artifacts
        (r'{\left(', r'('),
        (r'\right)}', r')'),
    ]
    
    for old, new in replacements:
        latex_str = latex_str.replace(old, new)

    # Handle matrices and vectors more robustly
    if isinstance(expr, sp.Matrix):
        # Remove any \left[ and \right] that SymPy might have added
        latex_str = latex_str.replace(r'\left[', '')
        latex_str = latex_str.replace(r'\right]', '')
        
        if expr.shape[1] == 1:  # Column vector
            latex_str = latex_str.replace(r'\begin{matrix}', r'\begin{pmatrix}')
            latex_str = latex_str.replace(r'\end{matrix}', r'\end{pmatrix}')
        else:  # Regular matrix
            latex_str = latex_str.replace(r'\begin{matrix}', r'\begin{bmatrix}')
            latex_str = latex_str.replace(r'\end{matrix}', r'\end{bmatrix}')

    # Remove HTML tags
    for remove_tag in (r'<span class="math-inline">', r'</span>', r'<span class="math-block">'):
        latex_str = latex_str.replace(remove_tag, '')

    # Remove any stray dollar signs
    latex_str = latex_str.replace('$', '')

    return latex_str

def sympy_to_multiline_latex(expr, use_dot_notation=False, line_length=55):
    """
    Convert SymPy expression to LaTeX with automatic line breaks for long expressions.
    Uses aligned environment to ensure proper alignment.
    """
    # Get the basic LaTeX string first
    latex_str = sympy_to_latex(expr, use_dot_notation)
    
    # If it's short enough, return as is
    if len(latex_str) <= line_length:
        return latex_str
    
    # Find good break points (+ and - operators at the top level)
    # and additional break points at multiplication and other operations
    parts = []
    current_part = ""
    paren_level = 0
    
    # Define additional break characters
    primary_break_chars = ['+', '-']
    secondary_break_chars = ['\\cdot', '\\sin', '\\cos', '=']
    
    i = 0
    while i < len(latex_str):
        # Track parentheses level
        if latex_str[i] in ['(', '[', '{']:
            paren_level += 1
        elif latex_str[i] in [')', ']', '}']:
            paren_level -= 1
        
        # Check for primary break points (+ and - not in exponents or subscripts)
        if paren_level == 0 and i > 0 and latex_str[i] in primary_break_chars and latex_str[i-1] != '^' and latex_str[i-1] != '_':
            parts.append(current_part)
            current_part = latex_str[i]  # Start new part with the operator
            i += 1
            continue
            
        # Check for secondary break points (functions like sin, cos, etc.)
        found_secondary = False
        if paren_level == 0 and len(current_part) > line_length // 2:
            for break_str in secondary_break_chars:
                if i + len(break_str) <= len(latex_str) and latex_str[i:i+len(break_str)] == break_str:
                    parts.append(current_part)
                    current_part = ""
                    found_secondary = True
                    break
        
        if not found_secondary:
            current_part += latex_str[i]
            i += 1
            
        # Force break if current part gets too long
        if len(current_part) > line_length and paren_level == 0:
            parts.append(current_part)
            current_part = ""
            
    # Add the last part
    if current_part:
        parts.append(current_part)
    
    # Generate the multiline output with proper alignment
    result = r'\begin{aligned} '
    
    # First line doesn't need indentation
    result += parts[0] + r' \\' + '\n'
    
    # Remaining lines
    for part in parts[1:]:
        result += r'& ' + part + r' \\' + '\n'
    
    # Close the environment
    result = result.rstrip('\n').rstrip('\\') + r' '
    result += r'\end{aligned}'
    
    return result

def add_multiline_eq(doc_obj, lhs_str, sympy_expr, label_str, use_dot=False):
    """
    Add a potentially long equation with proper line breaking
    """
    env = Environment()
    env._latex_name = 'align*'
    lhs_latex = lhs_str
    rhs_latex = sympy_to_multiline_latex(sympy_expr, use_dot_notation=use_dot)
    env.append(NoEscape(lhs_latex + r' &= ' + rhs_latex + r' \\'))
    doc_obj.append(env)

def add_sympy_eq_rhs(doc_obj, lhs_str, sympy_rhs_expr, label_str, use_dot=False):
    """
    Add an equation with align* environment, no HTML.
    """
    lhs_latex = lhs_str
    rhs_latex = sympy_to_latex(sympy_rhs_expr, use_dot_notation=use_dot)

    # Use align* environment directly
    env = Environment()
    env._latex_name = 'align*'
    env.append(NoEscape(lhs_latex + r' &= ' + rhs_latex + r' \\'))
    doc_obj.append(env)

def add_sympy_matrix_eq_rhs(doc_obj, lhs_str, sympy_matrix, label_str):
    """Same as above but for matrix eq: lhs = matrix."""
    matrix_latex = sympy_to_latex(sympy_matrix)
    
    # Use align* environment directly
    env = Environment()
    env._latex_name = 'align*'
    env.append(NoEscape(lhs_str + r' &= ' + matrix_latex + r' \\'))
    doc_obj.append(env)

def add_shorthand_definitions(doc):
    """Add definitions of our shorthands at the start of the document"""
    doc.append(NoEscape(r'\begin{align*}'))
    doc.append(NoEscape(r'&\text{Where: } \\'))
    doc.append(NoEscape(r'&c_1 = \cos(\theta_1), \quad s_1 = \sin(\theta_1) \\'))
    doc.append(NoEscape(r'&c_2 = \cos(\theta_2), \quad s_2 = \sin(\theta_2)'))
    doc.append(NoEscape(r'\end{align*}'))
    doc.append(NoEscape(r'\vspace{1em}'))  # Add some vertical space

def add_mass_matrix_elements(doc_obj, M):
    """Add mass matrix elements one by one with clear labeling"""
    doc_obj.append(NoEscape(r'\begin{align*}'))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            element = sp.simplify(M[i,j])
            if element != 0:  # Only show non-zero elements
                doc_obj.append(NoEscape(f"M_{{{i+1}{j+1}}} &= {sp.latex(element)} \\\\"))
    doc_obj.append(NoEscape(r'\end{align*}'))
    doc_obj.append(NoEscape(r'\vspace{1em}'))  # Add some vertical space

def add_cse_definitions(doc_obj, cse_defs):
    """Add definitions for Common Subexpression Elimination (CSE) terms."""
    if not cse_defs:
        return
    doc_obj.append(NoEscape(r'\textbf{Common Subexpressions:}\\ \vspace{0.5em}'))
    doc_obj.append(NoEscape(r'\begin{align*}'))
    for symbol, expr in cse_defs:
        # Use multiline for potentially long CSE expressions
        rhs_latex = sympy_to_multiline_latex(expr, use_dot_notation=True)
        doc_obj.append(NoEscape(f"{sp.latex(symbol)} &= {rhs_latex} \\\\"))
    doc_obj.append(NoEscape(r'\end{align*}'))
    doc_obj.append(NoEscape(r'\vspace{1em}'))  # Add some vertical space 