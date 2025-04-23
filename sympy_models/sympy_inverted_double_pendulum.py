"""
Symbolic Derivation and LaTeX Report Generation for Double Pendulum on a Cart.

Purpose:
--------
This script automates the derivation of the equations of motion (EoM) for a
classic double inverted pendulum mounted on a horizontally moving cart. It uses
the Lagrangian method of analytical mechanics and performs the symbolic
computations using the SymPy library.

The primary output is a nicely formatted PDF report, generated via LaTeX (using
the PyLaTeX library), detailing each step of the derivation process, including:
- System setup (parameters, coordinates)
- Kinematic calculations (positions, velocities)
- Formulation of Kinetic (T) and Potential (V) energies
- Construction of the Lagrangian (L = T - V)
- Application of the Euler-Lagrange equations
- Extraction of the system's Mass Matrix (M), Coriolis/Centrifugal/Gravity
  vector (C*dq + G)
- Solution for the generalized accelerations (q_ddot = M^-1 * (Tau - (C*dq + G)))

Optionally, it can substitute parameters for uniform rods and display those
results.

Methodology:
------------
1.  **Symbolic Representation:** SymPy is used to define system parameters
    (masses, lengths, inertia, etc.) and generalized coordinates (x, theta1,
    theta2) as symbolic variables and functions of time.
2.  **Lagrangian Mechanics:** The script follows the standard Lagrangian workflow:
    - Define positions and calculate velocities of relevant points/bodies.
    - Compute total kinetic energy (T) and potential energy (V).
    - Form the Lagrangian L = T - V.
    - Systematically apply the Euler-Lagrange equations:
      d/dt(dL/d(q_dot)) - dL/dq = Q
      to obtain the dynamic equations.
    - Q represents the generalized forces (only the cart force 'U' in this case).
3.  **Equation Solving:** The resulting second-order differential equations are
    manipulated symbolically to isolate the second derivatives (accelerations).
    This is done by identifying the Mass Matrix (M) and the vector of remaining
    terms (C*dq + G) and solving the system M*q_ddot = Tau - (C*dq + G).
4.  **Report Generation:** PyLaTeX is used to programmatically build a LaTeX
    document (`.tex` file). SymPy's `latex()` function converts symbolic
    expressions into LaTeX math strings, which are embedded in the document.
5.  **PDF Compilation:** If a LaTeX distribution is found, the script calls
    `pdflatex` via a subprocess to compile the `.tex` file into a PDF report.

Key Libraries:
--------------
- `sympy`: For symbolic mathematics (calculus, algebra, solving).
- `pylatex`: For programmatic generation of `.tex` files.
- `subprocess`, `os`, `platform`, `shutil`: Standard libraries used for
  checking/calling the external LaTeX compiler and opening the PDF.

Prerequisites:
--------------
- Python 3.x
- SymPy library (`pip install sympy`)
- PyLaTeX library (`pip install pylatex`)
- **A full LaTeX distribution:** Must be installed separately (e.g., TeX Live
  for Linux/macOS/Windows, or MiKTeX for Windows). The `pdflatex` executable
  must be in the system's PATH for automatic PDF compilation.

Configuration:
--------------
Several constants near the top of the script control its behavior:
- `OUTPUT_DIR`: Directory to save output files.
- `FILENAME_BASE`: Base name for the output `.tex` and `.pdf` files.
- `INCLUDE_UNIFORM_CASE`: Set to `True` to add a section with results assuming
  uniform rods (substitutes d1=l1/2, d2=l2/2).
- `COMPILE_PDF`: Set to `True` to attempt PDF compilation using `pdflatex`.
  Requires LaTeX to be installed. If `False`, only the `.tex` file is generated.
- `CLEAN_TEX`: Set to `True` to remove auxiliary LaTeX files (`.aux`, `.log`,
  `.toc`) and the `.tex` file *after* successful PDF compilation. Set to `False`
  to keep the `.tex` source for inspection or manual compilation.

Usage:
------
Run the script from the project root directory using the `-m` flag:
```bash
python -m sympy_models.sympy_inverted_double_pendulum
```
This ensures that Python's module resolution works correctly for the imports
within the script and its utilities.

"""


import sympy as sp
from sympy import sin, cos, Matrix
sp.init_printing(use_unicode=True, use_latex='mathjax')

from pylatex import Document, Section, Subsection, Command, Package
from pylatex.utils import NoEscape, bold
from pylatex.base_classes import Environment
from pylatex.math import Math
from pylatex.section import Paragraph

# Import helpers from the new utils module
from sympy_models.utils.sympy_to_latex import (
    format_vector_name,
    sympy_to_latex,
    sympy_to_multiline_latex,
    add_multiline_eq,
    add_sympy_eq_rhs,
    add_sympy_matrix_eq_rhs,
    add_shorthand_definitions,
    add_mass_matrix_elements,
    add_cse_definitions
)

import subprocess
import os
import platform
import shutil

# --- Step 0: Configuration ---
OUTPUT_DIR = 'sympy_models'  # Save outputs in this directory
FILENAME_BASE = 'double_pendulum_lagrangian_derivation_v2'
INCLUDE_UNIFORM_CASE = True
COMPILE_PDF = True
CLEAN_TEX = True

# Construct full paths
output_path_base = os.path.join(OUTPUT_DIR, FILENAME_BASE)

print("--- Running SymPy Calculations (v2) ---")

# Define time symbol
t = sp.symbols('t')
# Define constants (U is control force)
M, m1, m2, l1, l2, d1, d2, Icm1, Icm2, g, U = sp.symbols(
    'M m1 m2 l1 l2 d1 d2 Icm1 Icm2 g U', real=True
)
# Add symbols for damping, friction, gear ratio
b_slide, b_fric, b_joint1, b_joint2, gear = sp.symbols(
    'b_slide b_fric b_joint1 b_joint2 gear', real=True, positive=True # Assume positive
)

params = [M, m1, m2, l1, l2, d1, d2, Icm1, Icm2, g, U, 
          b_slide, b_fric, b_joint1, b_joint2, gear] # Add new params

# Generalized coords
x = sp.Function('x')(t)
th1 = sp.Function('theta1')(t)
th2 = sp.Function('theta2')(t)
q = sp.Matrix([x, th1, th2])

# First derivatives
q_d = q.diff(t)
x_d, th1_d, th2_d = q_d

# Second derivatives
q_dd = q_d.diff(t)
x_dd, th1_dd, th2_dd = q_dd

print("Coordinates defined:", q)
print("Velocities defined:", q_d)
print("Accelerations defined:", q_dd)

# 2. Kinematics
r_cart = sp.Matrix([x, 0])
r_cm1 = sp.Matrix([x + d1*sin(th1), d1*cos(th1)])
r_cm2 = sp.Matrix([
    x + l1*sin(th1) + d2*sin(th2),
    l1*cos(th1) + d2*cos(th2)
])

v_cart = r_cart.diff(t)
v_cm1 = r_cm1.diff(t)
v_cm2 = r_cm2.diff(t)

vx_cart, vy_cart = v_cart
vx_cm1, vy_cm1 = v_cm1
vx_cm2, vy_cm2 = v_cm2

# 3. Kinetic Energy
TCart = 0.5 * M * v_cart.dot(v_cart)
TRod1 = 0.5 * m1 * (v_cm1.dot(v_cm1)) + 0.5 * Icm1 * th1_d**2
TRod2 = 0.5 * m2 * (v_cm2.dot(v_cm2)) + 0.5 * Icm2 * th2_d**2
T = TCart + TRod1 + TRod2

# 4. Potential Energy
V = m1*g*r_cm1[1] + m2*g*r_cm2[1]

# 5. Lagrangian
L = T - V

# 6. Euler-Lagrange
# Define generalized forces including control, damping, and friction
Q_gen = sp.Matrix([
    gear*U - b_slide*x_d - b_fric*x_d, # Force on cart
    -b_joint1*th1_d,                   # Torque on joint 1
    -b_joint2*th2_d                    # Torque on joint 2
])

Eqs_LHS_vec = sp.zeros(len(q), 1)

# Store intermediate calculations for each coordinate
dL_dqdot_list = []
dtdL_dqdot_list = []
dL_dq_list = []

for i in range(len(q)):
    dL_dqdot = sp.diff(L, q_d[i])
    dtdL_dqdot = sp.diff(dL_dqdot, t)
    dL_dq = sp.diff(L, q[i])
    Eqs_LHS_vec[i] = dtdL_dqdot - dL_dq
    
    # Store intermediate results
    dL_dqdot_list.append(dL_dqdot)
    dtdL_dqdot_list.append(dtdL_dqdot)
    dL_dq_list.append(dL_dq)


# 7. Solve for Accelerations (Mass Matrix)
print("\nExtracting Mass Matrix terms:")
M_matrix = sp.zeros(len(q), len(q))
for i in range(len(q)):
    for j in range(len(q)):
        # Collect all terms with the acceleration
        M_matrix[i, j] = sp.collect(Eqs_LHS_vec[i], q_dd[j]).coeff(q_dd[j], 1)
        print(f"M[{i},{j}] coefficient of {q_dd[j]} in equation {i}:")
        print(sp.simplify(M_matrix[i, j]))

print("\nFull Mass Matrix:")
print(sp.simplify(M_matrix))

subs_dict = {acc: 0 for acc in q_dd}
C_G_vec = Eqs_LHS_vec.subs(subs_dict)
rhs_vector = Q_gen - C_G_vec
C_G_vec = sp.simplify(C_G_vec)
rhs_vector = sp.simplify(rhs_vector)

# Apply CSE to the RHS vector before solving
rhs_cse_defs, rhs_reduced_vector = sp.cse(rhs_vector)

# Note: Solving with the CSE'd vector might be slightly different
# Let's solve first, then apply CSE to the result for display
accel_eqs_vec = sp.simplify(M_matrix.LUsolve(rhs_vector))

# Apply CSE to the final acceleration equations for cleaner LaTeX output
accel_cse_defs, accel_reduced_vec = sp.cse(accel_eqs_vec)

print("Symbolic calculations complete.")

# --- Step 8: Generate LaTeX Document ---
print(f"\n--- Generating LaTeX Document ({output_path_base}.tex) ---")
geometry_options = {"tmargin": "2cm", "lmargin": "2cm", "rmargin":"2cm", "bmargin":"2cm"}
doc = Document(output_path_base, geometry_options=geometry_options)

# We want amsmath + extra packages for nicer matrices:
doc.packages.append(Package('amsmath'))       # Already there, but re-assert
doc.packages.append(Package('amssymb'))       # NEW
doc.packages.append(Package('amsfonts'))      # NEW
doc.packages.append(Package('bm'))            # Possibly for bold math if wanted
doc.packages.append(Package('graphicx'))

# Title, etc.
doc.preamble.append(Command('title', 'Lagrangian Derivation of Double Inverted Pendulum Dynamics (v2)'))
doc.preamble.append(Command('author', 'Generated by SymPy and PyLaTeX'))
doc.preamble.append(Command('date', NoEscape(r'\today')))
doc.append(NoEscape(r'\maketitle'))
doc.append(NoEscape(r'\tableofcontents'))
doc.append(NoEscape(r'\newpage'))

with doc.create(Section('1. System Setup')):
    doc.append('This document details the derivation of equations of motion for a double inverted pendulum on a cart using the Lagrangian method.\n')
    doc.append(Paragraph("Constants defined (U is control force):"))
    doc.append(Math(data=[NoEscape(sympy_to_latex(params))]))
    doc.append(Paragraph("Generalized coordinates (functions of time t):"))
    add_sympy_eq_rhs(doc, r"q(t)", q, "gen_coords")
    doc.append(Paragraph("Generalized velocities:"))
    add_sympy_eq_rhs(doc, r"\dot{q}(t)", q_d, "gen_vels", use_dot=True)
    doc.append(Paragraph("Generalized accelerations:"))
    add_sympy_eq_rhs(doc, r"\ddot{q}(t)", q_dd, "gen_accels", use_dot=True)

with doc.create(Section('2. Kinematics')):
    with doc.create(Subsection('Positions')):
         doc.append('Position of Cart Pivot:')
         add_sympy_eq_rhs(doc, format_vector_name('r_{cart}'), r_cart, "r_cart")
         doc.append('Position of CM of Rod 1:')
         add_sympy_eq_rhs(doc, format_vector_name('r_{cm1}'), r_cm1, "r_cm1")
         doc.append('Position of CM of Rod 2:')
         add_sympy_eq_rhs(doc, format_vector_name('r_{cm2}'), r_cm2, "r_cm2")

    with doc.create(Subsection('Velocities')):
        doc.append('Velocity of Cart:')
        add_sympy_eq_rhs(doc, format_vector_name('v_{cart}'), v_cart, "v_cart", use_dot=True)
        doc.append(NoEscape(r"Components are $v_{x,cart} = \dot{x}$ and $v_{y,cart} = 0$."))

        # Rod 1 velocity in matrix form
        doc.append('Velocity of CM of Rod 1:')
        v_cm1_matrix = sp.Matrix([vx_cm1, vy_cm1])
        add_sympy_eq_rhs(doc, format_vector_name('v_{cm1}'), v_cm1_matrix, "v_cm1", use_dot=True)

        # Rod 2 velocity in matrix form
        doc.append('Velocity of CM of Rod 2:')
        v_cm2_matrix = sp.Matrix([vx_cm2, vy_cm2])
        add_sympy_eq_rhs(doc, format_vector_name('v_{cm2}'), v_cm2_matrix, "v_cm2", use_dot=True)

with doc.create(Section('3. Kinetic Energy (T)')):
    doc.append(NoEscape(r'Total kinetic energy $T = T_{cart} + T_{rod1} + T_{rod2}$.'))
    doc.append(NoEscape(r'$$ T_{cart} = \frac{1}{2} M v_{x,cart}^2 $$'))
    doc.append(NoEscape(r'$$ T_{rod1} = \frac{1}{2} m_1 (v_{x,cm1}^2 + v_{y,cm1}^2) + \frac{1}{2} I_{cm1} \dot{\theta}_1^2 $$'))
    doc.append(NoEscape(r'$$ T_{rod2} = \frac{1}{2} m_2 (v_{x,cm2}^2 + v_{y,cm2}^2) + \frac{1}{2} I_{cm2} \dot{\theta}_2^2 $$'))

with doc.create(Section('4. Potential Energy (V)')):
    doc.append(NoEscape(r'Zero potential at $y=0$. So $V = m_1 g\,y_{cm1} + m_2 g\,y_{cm2}$.'))
    add_sympy_eq_rhs(doc, "V", V, "potential_energy")

with doc.create(Section('5. Lagrangian (L)')):
    doc.append(NoEscape(r'The Lagrangian is $L = T - V$.'))

with doc.create(Section('6. Euler-Lagrange Equations')):
    doc.append(NoEscape(r'$\frac{d}{dt}\bigl(\frac{\partial L}{\partial \dot{q}_i}\bigr) - \frac{\partial L}{\partial q_i} = Q_i$. For this system, $Q = [U, 0, 0]^T$.'))
    
    doc.append(Paragraph(bold('Intermediate steps of the Euler-Lagrange equations:')))
    
    # Add shorthand definitions
    add_shorthand_definitions(doc)
    
    coord_names = ["x", "\\theta_1", "\\theta_2"]
    
    with doc.create(Subsection('Partial derivatives with respect to velocities')):
        for i in range(len(q)):
            doc.append(Paragraph(f"For coordinate ${coord_names[i]}$:"))
            add_multiline_eq(doc, r"\frac{\partial L}{\partial \dot{" + coord_names[i] + r"}}", dL_dqdot_list[i], f"dL_dqdot_{i}", use_dot=True)
    
    with doc.create(Subsection('Time derivatives of partial derivatives')):
        for i in range(len(q)):
            doc.append(Paragraph(f"For coordinate ${coord_names[i]}$:"))
            add_multiline_eq(doc, r"\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{" + coord_names[i] + r"}}\right)", dtdL_dqdot_list[i], f"dtdL_dqdot_{i}", use_dot=True)
    
    with doc.create(Subsection('Partial derivatives with respect to coordinates')):
        for i in range(len(q)):
            doc.append(Paragraph(f"For coordinate ${coord_names[i]}$:"))
            add_multiline_eq(doc, r"\frac{\partial L}{\partial " + coord_names[i] + r"}", dL_dq_list[i], f"dL_dq_{i}", use_dot=True)
    
    with doc.create(Subsection('Complete Euler-Lagrange equations (LHS)')):
        for i in range(len(q)):
            doc.append(Paragraph(f"For coordinate ${coord_names[i]}$:"))
            add_multiline_eq(doc, r"\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{" + coord_names[i] + r"}}\right) - \frac{\partial L}{\partial " + coord_names[i] + r"}", Eqs_LHS_vec[i], f"eq_lhs_{i}", use_dot=True)

with doc.create(Section('7. Solving for Accelerations')):
    doc.append(r'The equations can be written in the form $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau$.')
    
    doc.append(Paragraph(bold('Mass Matrix M(q) elements:')))
    add_mass_matrix_elements(doc, M_matrix)
    
    doc.append(Paragraph(bold(r'Calculated RHS Vector $\tau - (C\dot{q}+G)$: (Simplified using CSE)')))
    # Add CSE definitions for RHS vector
    add_cse_definitions(doc, rhs_cse_defs)
    # Add the simplified RHS vector using the reduced expression
    add_sympy_matrix_eq_rhs(doc, r"\tau - (C\dot{q}+G)", rhs_reduced_vector[0], "rhs_vector_cse") # CSE returns list

with doc.create(Section('8. Equations of Motion (Accelerations)')):
    doc.append('Final expressions for the accelerations are (Simplified using CSE):')
    # Add CSE definitions for acceleration equations
    add_cse_definitions(doc, accel_cse_defs)
    
    # Use the reduced expressions from CSE for the final equations
    with doc.create(Subsection('Acceleration of the cart')):
        add_sympy_eq_rhs(doc, r"\ddot{x}", accel_reduced_vec[0][0], "x_ddot_cse", use_dot=True)
    with doc.create(Subsection('Angular Acceleration of Rod 1')):
        add_sympy_eq_rhs(doc, r"\ddot{\theta}_1", accel_reduced_vec[0][1], "th1_ddot_cse", use_dot=True)
    with doc.create(Subsection('Angular Acceleration of Rod 2')):
        add_sympy_eq_rhs(doc, r"\ddot{\theta}_2", accel_reduced_vec[0][2], "th2_ddot_cse", use_dot=True)

if INCLUDE_UNIFORM_CASE:
    try:
        with doc.create(Section('9. Uniform Rod Case')):
            doc.append('Substituting uniform rods: $d_1 = l_1/2, d_2 = l_2/2$.')
            uniform_rod_subs = {
                d1: l1/sp.Rational(2),
                d2: l2/sp.Rational(2)
            }
            M_matrix_uniform = sp.simplify(M_matrix.subs(uniform_rod_subs))
            rhs_vector_uniform = sp.simplify(rhs_vector.subs(uniform_rod_subs))
            accel_eqs_vec_uniform = sp.simplify(M_matrix_uniform.LUsolve(rhs_vector_uniform))

            # Apply CSE to uniform case accelerations as well
            uni_accel_cse_defs, uni_accel_reduced_vec = sp.cse(accel_eqs_vec_uniform)

            doc.append(Paragraph(bold('Mass Matrix M(q) (Uniform Rods):')))
            add_sympy_matrix_eq_rhs(doc, "M_{uniform}(q)", M_matrix_uniform, "mass_matrix_uniform")
            doc.append(Paragraph(bold('Accelerations (Uniform Rods): (Simplified using CSE)')))
            # Add CSE definitions for uniform case
            add_cse_definitions(doc, uni_accel_cse_defs)
            # Use reduced expressions
            add_sympy_eq_rhs(doc, r"\ddot{x}", uni_accel_reduced_vec[0][0], "x_ddot_uniform_cse", use_dot=True)
            add_sympy_eq_rhs(doc, r"\ddot{\theta}_1", uni_accel_reduced_vec[0][1], "th1_ddot_uniform_cse", use_dot=True)
            add_sympy_eq_rhs(doc, r"\ddot{\theta}_2", uni_accel_reduced_vec[0][2], "th2_ddot_uniform_cse", use_dot=True)
    except Exception as e:
        doc.append(Paragraph(bold('Error during uniform rod substitution.')))
        print(e)

print(f"\n--- Generating PDF ({output_path_base}.pdf) ---")
compiler = 'pdflatex'
if not shutil.which(compiler):
    print(f"ERROR: '{compiler}' not found.")
    COMPILE_PDF = False

if COMPILE_PDF:
    try:
        doc.generate_pdf(output_path_base, clean_tex=CLEAN_TEX, compiler=compiler, compiler_args=['-interaction=nonstopmode'])
        pdf_path = f"{output_path_base}.pdf"
        if os.path.exists(pdf_path):
            doc.generate_pdf(output_path_base, clean_tex=CLEAN_TEX, compiler=compiler, compiler_args=['-interaction=nonstopmode'])
            print(f"Successfully generated {pdf_path}")
            if os.path.exists(pdf_path):
                try:
                    if platform.system() == "Windows":
                        os.startfile(pdf_path)
                    elif platform.system() == "Darwin":
                        subprocess.call(['open', pdf_path])
                    else:
                        subprocess.call(['xdg-open', pdf_path])
                except Exception as open_e:
                    print(f"Could not open PDF automatically: {open_e}")
        else:
            print(f"{pdf_path} not found. Check log.")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print(f"Check the .log file or {output_path_base}.tex")
else:
    doc.generate_tex(output_path_base)
    print(f"Generated {output_path_base}.tex only.")

print("\n--- Script Finished ---")
