import sympy as sp
from sympy import sin, cos, Matrix # Use SymPy's Matrix
# Configure printing for console and LaTeX output
# derivative_brackets=False helps avoid d/dt{} notation sometimes
# use_latex='mathjax' or similar can influence output style
sp.init_printing(use_unicode=True, use_latex='mathjax')

# PyLaTeX imports
from pylatex import Document, Section, Subsection, Command, Package
from pylatex.utils import NoEscape, bold
from pylatex.base_classes import Environment
from pylatex.math import Math, Matrix as PyLatexMatrix # PyLaTeX Matrix
from pylatex.section import Paragraph

# For compiling the LaTeX file
import subprocess
import os
import platform
import shutil

# --- Step 0: Configuration ---
FILENAME = 'double_pendulum_lagrangian_derivation_v2' # Updated filename
# Set to True to include results for uniform rods
INCLUDE_UNIFORM_CASE = True # Keeping this True as I_cm part removed
# Set to True to attempt compiling PDF (requires LaTeX installed)
COMPILE_PDF = True
# Set to False to keep the .tex file after compilation
CLEAN_TEX = True

# --- Steps 1-5: SymPy Calculations ---
print("--- Running SymPy Calculations (v2) ---")
# Define time symbol
t = sp.symbols('t')
# Define constants
# Using U for control force instead of F
M, m1, m2, l1, l2, d1, d2, Icm1, Icm2, g, U = sp.symbols(
    'M m1 m2 l1 l2 d1 d2 Icm1 Icm2 g U', real=True # Assuming real is sufficient
)
# Collect symbols into a list for display purposes
params = [M, m1, m2, l1, l2, d1, d2, Icm1, Icm2, g, U]

# Define generalized coordinates as functions of time
x = sp.Function('x')(t)
th1 = sp.Function('theta1')(t)
th2 = sp.Function('theta2')(t)
q = sp.Matrix([x, th1, th2])

# Define first time derivatives (velocities)
q_d = q.diff(t)
x_d, th1_d, th2_d = q_d # Unpack

# Define second time derivatives (accelerations)
q_dd = q_d.diff(t)
x_dd, th1_dd, th2_dd = q_dd # Unpack

print("Coordinates defined:", q)
print("Velocities defined:", q_d)
print("Accelerations defined:", q_dd)


# 2. Kinematics: Define positions and calculate velocities
print("\n--- 2. Kinematics ---")
# Position vectors
r_cart = sp.Matrix([x, 0]) # Position of cart pivot
r_cm1 = sp.Matrix([x + d1*sin(th1), d1*cos(th1)])          # Position CM Rod 1
# r_p2 = sp.Matrix([x + l1*sin(th1), l1*cos(th1)]) # Position Pivot 2 (if needed)
r_cm2 = sp.Matrix([x + l1*sin(th1) + d2*sin(th2), l1*cos(th1) + d2*cos(th2)]) # Position CM Rod 2

# Calculate velocities by differentiating positions w.r.t. time 't'
v_cart = r_cart.diff(t)
v_cm1 = r_cm1.diff(t)
v_cm2 = r_cm2.diff(t)

# Extract velocity components for clarity
vx_cart, vy_cart = v_cart
vx_cm1, vy_cm1 = v_cm1
vx_cm2, vy_cm2 = v_cm2

print("Positions and Velocities calculated.")


# 3. Calculate Kinetic Energy (T)
print("\n--- 3. Kinetic Energy ---")
# Use v.dot(v) for squared magnitude
TCart = 0.5 * M * v_cart.dot(v_cart) # More explicit using v_cart
TRod1 = 0.5 * m1 * (v_cm1.dot(v_cm1)) + 0.5 * Icm1 * th1_d**2
TRod2 = 0.5 * m2 * (v_cm2.dot(v_cm2)) + 0.5 * Icm2 * th2_d**2
T = TCart + TRod1 + TRod2
print("Total Kinetic Energy (T) expression formed.")


# 4. Calculate Potential Energy (V)
print("\n--- 4. Potential Energy ---")
# Using y-coordinates (index 1) from r_cm1, r_cm2
V = m1 * g * r_cm1[1] + m2 * g * r_cm2[1]
print("Total Potential Energy (V) expression formed.")


# 5. Form the Lagrangian (L)
print("\n--- 5. Lagrangian ---")
L = T - V
print("Lagrangian (L = T - V) formed.")


# 6. Apply Euler-Lagrange Equations
print("\n--- 6. Euler-Lagrange Equations ---")
# Generalized forces: U corresponds to x, 0 for theta1, 0 for theta2
Q_gen = sp.Matrix([U, 0, 0])

Eqs_LHS_vec = sp.zeros(len(q), 1) # Initialize as column vector
for i in range(len(q)):
    dL_dqdot = sp.diff(L, q_d[i])
    dtdL_dqdot = sp.diff(dL_dqdot, t)
    dL_dq = sp.diff(L, q[i])
    Eqs_LHS_vec[i] = dtdL_dqdot - dL_dq
    print(f"LHS of Euler-Lagrange equations for q[{i}]: {Eqs_LHS_vec[i]}")

print("Calculated LHS of Euler-Lagrange equations.")


# 7. Solve for Accelerations (Mass Matrix Method)
print("\n--- 7. Solving for Accelerations ---")
M_matrix = sp.zeros(len(q), len(q))
for i in range(len(q)):
    for j in range(len(q)):
        M_matrix[i, j] = sp.simplify(Eqs_LHS_vec[i].coeff(q_dd[j]))
print(f"Mass matrix: {M_matrix}")

subs_dict = {acc: 0 for acc in q_dd}
C_G_vec = Eqs_LHS_vec.subs(subs_dict)
rhs_vector = Q_gen - C_G_vec
# Simplify vectors AFTER calculation
C_G_vec = sp.simplify(C_G_vec) # Contains C*q_d + G terms
rhs_vector = sp.simplify(rhs_vector) # Contains Tau - (C*q_d + G)

print("Solving M * q_ddot = RHS...")
accel_eqs_vec = sp.simplify(M_matrix.LUsolve(rhs_vector))
print(accel_eqs_vec)
print("Symbolic calculations complete.")

# --- Step 8: Generate LaTeX Document using PyLaTeX ---
print(f"\n--- Generating LaTeX Document ({FILENAME}.tex) ---")

# Document Setup
geometry_options = {"tmargin": "2cm", "lmargin": "2cm", "rmargin": "2cm", "bmargin":"2cm"}
doc = Document(FILENAME, geometry_options=geometry_options)
doc.packages.append(Package('amsmath'))
doc.packages.append(Package('graphicx'))
# Use physics package for \dot notation if desired, requires separate config
# doc.packages.append(Package('physics'))

# --- Modified Helper Functions ---
# Configure sympy's latex printer for dot notation (might need tweaking)
# Using derivative_brackets=False is a start. Need specific settings for \dot{}.
# Let's try defining a custom printer for velocity/acceleration only.
# Or rely on MathJax mode's defaults for derivatives first.

def sympy_to_latex(expr, use_dot_notation=False):
    """ Convert SymPy expression to LaTeX string. """
    # Basic latex conversion first
    # Inline mode often uses more compact derivative notation
    # REMOVED: derivative_brackets=False
    latex_str = sp.latex(expr, mode='inline')

    # Attempt manual replacement for dot notation (crude but might work)
    # This part is tricky and might need refinement based on sympy's output
    if use_dot_notation:
       # NOTE: Ensure these replacements match the actual output of sp.latex()
       # You might need to inspect the .tex file if dots don't appear correctly
       # and adjust these replacement patterns.
       latex_str = latex_str.replace(r'\frac{d}{d t} x{\left(t \right)}', r'\dot{x}')
       latex_str = latex_str.replace(r'\frac{d}{d t} \theta_{1}{\left(t \right)}', r'\dot{\theta}_{1}')
       latex_str = latex_str.replace(r'\frac{d}{d t} \theta_{2}{\left(t \right)}', r'\dot{\theta}_{2}')
       latex_str = latex_str.replace(r'\frac{d^{2}}{d t^{2}} x{\left(t \right)}', r'\ddot{x}')
       latex_str = latex_str.replace(r'\frac{d^{2}}{d t^{2}} \theta_{1}{\left(t \right)}', r'\ddot{\theta}_{1}')
       latex_str = latex_str.replace(r'\frac{d^{2}}{d t^{2}} \theta_{2}{\left(t \right)}', r'\ddot{\theta}_{2}')
       # Handle cases where functions might simplify, e.g., Derivative(x(t), t)
       latex_str = latex_str.replace(r'Derivative{\left(x{\left(t \right)}, t \right)}', r'\dot{x}')
       latex_str = latex_str.replace(r'Derivative{\left(\theta_{1}{\left(t \right)}, t \right)}', r'\dot{\theta}_{1}')
       latex_str = latex_str.replace(r'Derivative{\left(\theta_{2}{\left(t \right)}, t \right)}', r'\dot{\theta}_{2}')
       latex_str = latex_str.replace(r'Derivative{\left(x{\left(t \right)}, \left(t, 2\right) \right)}', r'\ddot{x}')
       latex_str = latex_str.replace(r'Derivative{\left(\theta_{1}{\left(t \right)}, \left(t, 2\right) \right)}', r'\ddot{\theta}_{1}')
       latex_str = latex_str.replace(r'Derivative{\left(\theta_{2}{\left(t \right)}, \left(t, 2\right) \right)}', r'\ddot{\theta}_{2}')

    return latex_str

def add_sympy_eq_rhs(doc_obj, lhs_str, sympy_rhs_expr, label_str, use_dot=False):
    """Adds 'LHS = RHS' equation from SymPy expr with label."""
    lhs_latex = NoEscape(lhs_str + r' = ')
    rhs_latex = NoEscape(sympy_to_latex(sympy_rhs_expr, use_dot_notation=use_dot))
    env = Environment()
    env._latex_name = 'equation'
    env.options = NoEscape(r'\label{eq:' + label_str + r'}') # Add label
    env.append(lhs_latex)
    env.append(rhs_latex)
    doc_obj.append(env)

def add_sympy_matrix_eq_rhs(doc_obj, lhs_str, sympy_matrix, label_str):
    """Adds 'LHS = Matrix' equation from SymPy matrix with label."""
    doc_obj.append(NoEscape(lhs_str + r' = ')) # Add "M = " before matrix
    math_cmd = NoEscape(sympy_to_latex(sympy_matrix)) # Use helper
    env = Environment()
    env._latex_name = 'equation*' # Use equation* for unnumbered matrix display if preferred
    # env._latex_name = 'equation' # Use equation for numbered matrix display
    env.options = NoEscape(r'\label{eq:' + label_str + r'}') # Label still works
    env.append(math_cmd)
    doc_obj.append(env)


# --- Populate Document (v2) ---

# Title
doc.preamble.append(Command('title', 'Lagrangian Derivation of Double Inverted Pendulum Dynamics (v2)'))
doc.preamble.append(Command('author', 'Generated by SymPy and PyLaTeX'))
doc.preamble.append(Command('date', NoEscape(r'\today')))
doc.append(NoEscape(r'\maketitle'))
doc.append(NoEscape(r'\tableofcontents')) # Add table of contents
doc.append(NoEscape(r'\newpage'))

# Introduction / Setup
with doc.create(Section('1. System Setup')):
    doc.append('This document details the derivation of the equations of motion for a double inverted pendulum on a cart using the Lagrangian method.')
    doc.append(Paragraph("Constants defined (U is control force):"))
    doc.append(Math(data=[NoEscape(sympy_to_latex(params))]))
    doc.append(Paragraph("Generalized coordinates (functions of time t):"))
    add_sympy_eq_rhs(doc, "q(t)", q, "gen_coords") # Using helper
    doc.append(Paragraph("Generalized velocities:"))
    add_sympy_eq_rhs(doc, r"\dot{q}(t)", q_d, "gen_vels", use_dot=True) # Using helper
    doc.append(Paragraph("Generalized accelerations:"))
    add_sympy_eq_rhs(doc, r"\ddot{q}(t)", q_dd, "gen_accels", use_dot=True) # Using helper

# Kinematics
with doc.create(Section('2. Kinematics')):
    doc.append('The positions and velocities of the cart and the centers of mass (CM) for Rod 1 and Rod 2 are required.')
    with doc.create(Subsection('Positions')):
         doc.append('Position of Cart Pivot:')
         add_sympy_eq_rhs(doc, r"\mathbf{r}_{cart}", r_cart, "r_cart")
         doc.append('Position of CM of Rod 1:')
         add_sympy_eq_rhs(doc, r"\mathbf{r}_{cm1}", r_cm1, "r_cm1")
         doc.append('Position of CM of Rod 2:')
         add_sympy_eq_rhs(doc, r"\mathbf{r}_{cm2}", r_cm2, "r_cm2")

    with doc.create(Subsection('Velocities')):
        doc.append('Velocity of Cart:')
        add_sympy_eq_rhs(doc, r"\mathbf{v}_{cart}", v_cart, "v_cart", use_dot=True)
        # Corrected line for cart velocity components description
        doc.append(NoEscape(r"Components are $v_{x,cart} = \dot{x}$ and $v_{y,cart} = 0$."))
        doc.append(Paragraph('Velocity of CM of Rod 1 <span class="math-inline">\\mathbf\{v\}\_\{cm1\} \= \[v\_\{x,cm1\}, v\_\{y,cm1\}\]^T</span>:'))
        add_sympy_eq_rhs(doc, "v_{x,cm1}", vx_cm1, "vx_cm1", use_dot=True)
        add_sympy_eq_rhs(doc, "v_{y,cm1}", vy_cm1, "vy_cm1", use_dot=True)

        doc.append(Paragraph('Velocity of CM of Rod 2 <span class="math-inline">\\mathbf\{v\}\_\{cm2\} \= \[v\_\{x,cm2\}, v\_\{y,cm2\}\]^T</span>:'))
        add_sympy_eq_rhs(doc, "v_{x,cm2}", vx_cm2, "vx_cm2", use_dot=True)
        add_sympy_eq_rhs(doc, "v_{y,cm2}", vy_cm2, "vy_cm2", use_dot=True)


# Kinetic Energy
with doc.create(Section('3. Kinetic Energy (T)')):
    doc.append(NoEscape(r'Total kinetic energy <span class="math-inline">T \= T\_\{cart\} \+ T\_\{rod1\} \+ T\_\{rod2\}</span>, where:'))
    # Using specific velocity components here makes it clearer
    doc.append(NoEscape(r'$$ T_{cart} = \frac{1}{2} M v_{x,cart}^2 <span class="math-block">'))
    # Corrected line for T_rod1 definition
    doc.append(NoEscape(r'$$ T_{rod1} = \frac{1}{2} m_1 (v_{x,cm1}^2 + v_{y,cm1}^2) + \frac{1}{2} I_{cm1} \dot{\theta}_1^2 $$'))
    # Corrected line for T_rod2 definition
    doc.append(NoEscape(
        r'$$ T_{rod2} = \frac{1}{2} m_2 (v_{x,cm2}^2 + v_{y,cm2}^2) + \frac{1}{2} I_{cm2} \dot{\theta}_2^2 $$'))
    doc.append('The full symbolic expression for T is calculated but omitted here for brevity.')


# Potential Energy
with doc.create(Section('4. Potential Energy (V)')):
    doc.append(NoEscape(r'Assuming zero potential energy at <span class="math-inline">y\=0</span>, the potential energy <span class="math-inline">V</span> is given by <span class="math-inline">V \= m\_1 g y\_\{cm1\} \+ m\_2 g y\_\{cm2\}</span>.'))
    add_sympy_eq_rhs(doc, "V", V, "potential_energy")

# Lagrangian
with doc.create(Section('5. Lagrangian (L)')):
    doc.append(NoEscape(r'The Lagrangian is defined as <span class="math-inline">L \= T \- V</span>.'))
    doc.append('The full symbolic expression for L is calculated internally but not displayed.')

# Euler-Lagrange Equations
with doc.create(Section('6. Euler-Lagrange Equations')):
    doc.append(NoEscape(r'The equations of motion are found using the Euler-Lagrange equation for each generalized coordinate <span class="math-inline">q\_i \\in \\\{x, \\theta\_1, \\theta\_2\\\}</span>:'))
    # Use equation* for unnumbered general equation
    doc.append(NoEscape(r'\begin{equation*} \frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) - \frac{\partial L}{\partial q_i} = Q_i \quad \label{eq:EL_general} \end{equation*}'))
    doc.append(NoEscape(r'Where <span class="math-inline">Q\_i</span> are the generalized forces. For this system, <span class="math-inline">Q \= \[U, 0, 0\]^T</span>.'))
    doc.append('Calculating these derivatives leads to three coupled second-order differential equations.')

# Solving for Accelerations
with doc.create(Section('7. Solving for Accelerations')):
    doc.append('The Euler-Lagrange equations can be written in the standard matrix form for manipulators:')
    # Use equation* for unnumbered general equation
    doc.append(NoEscape(r'\begin{equation*} M(q) \ddot{q} + C(q, \dot{q})\dot{q} + G(q) = \tau \quad \label{eq:matrix_form} \end{equation*}'))
    doc.append(NoEscape(r'Where <span class="math-inline">M\(q\)</span> is the mass matrix, <span class="math-inline">C\(q, \\dot\{q\}\)\\dot\{q\}</span> represents Coriolis/centrifugal terms (<span class="math-inline">C\\dot\{q\}</span> vector), <span class="math-inline">G\(q\)</span> represents gravitational terms (<span class="math-inline">G</span> vector), and <span class="math-inline">\\tau</span> is the vector of generalized forces (here <span class="math-inline">\\tau \= Q \= \[U, 0, 0\]^T</span>).'))
    doc.append(NoEscape(r'Our calculation identifies <span class="math-inline">M\(q\)</span> and the remaining terms (<span class="math-inline">C\\dot\{q\} \+ G</span>), rearranged as <span class="math-inline">M\(q\) \\ddot\{q\} \= \\tau \- \(C\(q, \\dot\{q\}\)\\dot\{q\} \+ G\(q\)\)</span>.'))

    doc.append(Paragraph(bold('Calculated Mass Matrix M(q):')))
    add_sympy_matrix_eq_rhs(doc, "M(q)", M_matrix, "mass_matrix")

    doc.append(Paragraph(bold(NoEscape(r'Calculated RHS Vector <span class="math-inline">\\tau \- \(C\(q, \\dot\{q\}\)\\dot\{q\} \+ G\(q\)\)</span>:'))))
    add_sympy_matrix_eq_rhs(doc, r"\tau - (C\dot{q} + G)", rhs_vector, "rhs_vector")

    doc.append(NoEscape(r'Solving <span class="math-inline">\\ddot\{q\} \= M\(q\)^\{\-1\} \(\\tau \- \(C\\dot\{q\} \+ G\)\)</span> gives the equations of motion.'))

# Results
with doc.create(Section('8. Equations of Motion (Accelerations)')):
    doc.append('The final expressions for the generalized accelerations are:')
    with doc.create(Subsection('Acceleration of the cart')):
        add_sympy_eq_rhs(doc, r"\ddot{x}", accel_eqs_vec[0], "x_ddot", use_dot=True)
    with doc.create(Subsection('Angular Acceleration of Rod 1')):
        add_sympy_eq_rhs(doc, r"\ddot{\theta}_1", accel_eqs_vec[1], "th1_ddot", use_dot=True)
    with doc.create(Subsection('Angular Acceleration of Rod 2')):
         add_sympy_eq_rhs(doc, r"\ddot{\theta}_2", accel_eqs_vec[2], "th2_ddot", use_dot=True)

# Optional: Uniform Rod Case
if INCLUDE_UNIFORM_CASE:
    try:
        with doc.create(Section('9. Uniform Rod Case')):
            doc.append('Substituting parameters for uniform rods (excluding <span class="math-inline">I\_\{cm\}</span>):')
            doc.append(NoEscape(r'$$ d_1 = l_1/2, \quad d_2 = l_2/2 $$'))

            # Use Rational for exact fractions
            uniform_rod_subs = {
                d1: l1/sp.Rational(2),
                d2: l2/sp.Rational(2)
                # Icm1 and Icm2 are NOT substituted per user request
            }
            print("\nSubstituting uniform rod parameters (d1, d2 only)...")
            M_matrix_uniform = sp.simplify(M_matrix.subs(uniform_rod_subs))
            rhs_vector_uniform = sp.simplify(rhs_vector.subs(uniform_rod_subs))

            print("Re-solving with uniform rod parameters...")
            accel_eqs_vec_uniform = sp.simplify(M_matrix_uniform.LUsolve(rhs_vector_uniform))

            doc.append(Paragraph(bold('Mass Matrix M(q) (Uniform Rods):')))
            add_sympy_matrix_eq_rhs(doc, "M_{uniform}(q)", M_matrix_uniform, "mass_matrix_uniform")
            doc.append(Paragraph(bold('Accelerations (Uniform Rods):')))
            add_sympy_eq_rhs(doc, r"\ddot{x}", accel_eqs_vec_uniform[0], "x_ddot_uniform", use_dot=True)
            add_sympy_eq_rhs(doc, r"\ddot{\theta}_1", accel_eqs_vec_uniform[1], "th1_ddot_uniform", use_dot=True)
            add_sympy_eq_rhs(doc, r"\ddot{\theta}_2", accel_eqs_vec_uniform[2], "th2_ddot_uniform", use_dot=True)

    except Exception as e:
        print(f"Error during uniform rod substitution/display: {e}")
        doc.append(Paragraph(bold('Error occurred during uniform rod calculation.')))


# --- Step 9: Generate and Compile PDF ---
print(f"\n--- Generating PDF ({FILENAME}.pdf) ---")

# Check if pdflatex command exists
compiler = 'pdflatex'
if not shutil.which(compiler):
    print(f"ERROR: '{compiler}' command not found.")
    print("Please install a LaTeX distribution (TeX Live or MiKTeX) and ensure it's in your system's PATH.")
    COMPILE_PDF = False

if COMPILE_PDF:
    try:
        # Generate PDF. Might need multiple runs for labels/TOC.
        print(f"Generating {FILENAME}.pdf (might take a moment)...")
        doc.generate_pdf(FILENAME, clean_tex=CLEAN_TEX, compiler=compiler, compiler_args=['-interaction=nonstopmode']) # Run twice for TOC/labels
        # Check if compilation succeeded (basic check: pdf exists)
        pdf_path = f"{FILENAME}.pdf"
        if os.path.exists(pdf_path):
             # Run again for references/TOC
             print("Running LaTeX compiler again for references...")
             doc.generate_pdf(FILENAME, clean_tex=CLEAN_TEX, compiler=compiler, compiler_args=['-interaction=nonstopmode'])
             print(f"Successfully generated {FILENAME}.pdf")

             # Optional: Open the PDF based on OS
             if os.path.exists(pdf_path): # Check again after second run
                 print(f"Attempting to open {pdf_path}...")
                 try:
                     if platform.system() == "Windows":
                         os.startfile(pdf_path)
                     elif platform.system() == "Darwin": # macOS
                         subprocess.call(['open', pdf_path])
                     else: # Linux
                         subprocess.call(['xdg-open', pdf_path])
                 except Exception as open_e:
                      print(f"Could not automatically open PDF: {open_e}")
             else:
                  print(f"Output PDF {pdf_path} not found after second compilation run.")
        else:
             print(f"Output PDF {pdf_path} not found after first compilation run. Check {FILENAME}.log.")

    except Exception as e:
        # Catch potential errors during PDF generation (e.g., LaTeX errors)
        print(f"\nError generating PDF: {e}")
        print(f"Check the LaTeX log file: {FILENAME}.log")
        print(f"The LaTeX source file is available at: {FILENAME}.tex (if clean_tex=False)")
else:
    # Generate only the .tex file if PDF compilation is disabled or compiler not found
    try:
        doc.generate_tex(FILENAME)
        print(f"Successfully generated {FILENAME}.tex. PDF compilation skipped.")
    except Exception as e:
        print(f"Error generating TEX file: {e}")

print("\n--- Script Finished ---")