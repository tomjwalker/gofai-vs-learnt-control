# analysis/visualize_mpc_costs.py
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# Add the project root to the Python path to allow importing from src
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the cost function classes
from src.algorithms.classic.mpc_costs import QuadraticCost, PendulumSwingupCost

def visualize_costs():
    print("Visualizing MPC cost functions...")

    # --- Configuration ---
    # Dummy weights (adjust if needed to highlight differences)
    Q_diag = [0.1, 5.0, 0.1, 0.1] # Example weights [x, theta, xd, thd]
    R_val = 0.01
    
    Q = ca.diag(Q_diag)
    R = ca.DM([R_val])
    
    # Instantiate cost calculators
    quad_cost_calc = QuadraticCost()
    swingup_cost_calc = PendulumSwingupCost()
    
    # Reference state (upright)
    X_ref = ca.DM.zeros(4, 1)
    
    # Define symbolic variables needed for evaluation
    Xk_sym = ca.SX.sym('Xk', 4)
    Uk_sym = ca.SX.sym('Uk', 1)
    
    # Create CasADi functions for numerical evaluation
    quad_cost_func = ca.Function('quad_cost', [Xk_sym, Uk_sym], 
                                 [quad_cost_calc.calculate_stage_cost(Xk_sym, Uk_sym, X_ref, Q, R)])
    swingup_cost_func = ca.Function('swingup_cost', [Xk_sym, Uk_sym], 
                                  [swingup_cost_calc.calculate_stage_cost(Xk_sym, Uk_sym, X_ref, Q, R)])

    # --- Plot 1: Cost vs. Theta --- 
    print("Plotting Cost vs. Theta...")
    theta_range = np.linspace(-np.pi - 0.5, np.pi + 0.5, 200)
    costs_quad_theta = []
    costs_swingup_theta = []
    
    # Evaluate costs at x=0, xd=0, thd=0, u=0
    state_base = np.array([0.0, 0.0, 0.0, 0.0])
    control_val = np.array([0.0])
    
    for theta in theta_range:
        state = state_base.copy()
        state[1] = theta
        costs_quad_theta.append(quad_cost_func(state, control_val).full().item())
        costs_swingup_theta.append(swingup_cost_func(state, control_val).full().item())
        
    plt.figure(figsize=(10, 6))
    plt.plot(theta_range, costs_quad_theta, label='Quadratic Cost (on Theta)')
    plt.plot(theta_range, costs_swingup_theta, label='Swingup Cost (1 - cos(Theta))', linestyle='--')
    plt.xlabel('Pole Angle (theta) [rad]')
    plt.ylabel('Stage Cost (Example Q/R)')
    plt.title('Comparison of Cost Functions vs. Pole Angle')
    # Add vertical lines for key angles
    plt.axvline(0, color='grey', linestyle=':', label='Upright (0 rad)')
    plt.axvline(np.pi, color='red', linestyle=':', label='Down (+pi rad)')
    plt.axvline(-np.pi, color='red', linestyle=':')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=-0.5) # Start y-axis slightly below 0
    plt.show()

    # --- Plot 2: Cost vs. Cart Position --- 
    print("Plotting Cost vs. Cart Position...")
    x_range = np.linspace(-1.5, 1.5, 100)
    costs_quad_x = []
    costs_swingup_x = []
    
    # Evaluate costs at theta=0, xd=0, thd=0, u=0
    state_base = np.array([0.0, 0.0, 0.0, 0.0])
    control_val = np.array([0.0])

    for x in x_range:
        state = state_base.copy()
        state[0] = x
        costs_quad_x.append(quad_cost_func(state, control_val).full().item())
        costs_swingup_x.append(swingup_cost_func(state, control_val).full().item())

    plt.figure(figsize=(10, 6))
    # Note: In this slice, the swingup cost only differs if Q[0,0] is different,
    # which isn't the primary change. They might look identical here.
    plt.plot(x_range, costs_quad_x, label='Quadratic Cost (on x)')
    plt.plot(x_range, costs_swingup_x, label='Swingup Cost (on x)', linestyle='--') # Still quadratic for x
    plt.xlabel('Cart Position (x) [m]')
    plt.ylabel('Stage Cost (Example Q/R)')
    plt.title('Comparison of Cost Functions vs. Cart Position')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # --- Plot 3: Cost vs. Control Input --- 
    print("Plotting Cost vs. Control Input...")
    u_range = np.linspace(-4, 4, 100)
    costs_quad_u = []
    costs_swingup_u = []
    
    # Evaluate costs at state = [0,0,0,0]
    state_base = np.array([0.0, 0.0, 0.0, 0.0])

    for u in u_range:
        control_val = np.array([u])
        costs_quad_u.append(quad_cost_func(state_base, control_val).full().item())
        costs_swingup_u.append(swingup_cost_func(state_base, control_val).full().item())

    plt.figure(figsize=(10, 6))
    # Note: The control cost term is identical in both formulations.
    plt.plot(u_range, costs_quad_u, label='Quadratic Cost (on u)')
    plt.plot(u_range, costs_swingup_u, label='Swingup Cost (on u)', linestyle='--') # Identical control cost term
    plt.xlabel('Control Input (u) [N]')
    plt.ylabel('Stage Cost (Example Q/R)')
    plt.title('Comparison of Cost Functions vs. Control Input')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    visualize_costs() 