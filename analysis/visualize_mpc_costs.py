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
from src.algorithms.classic.mpc_costs import QuadraticCost, PendulumSwingupCost, PendulumSwingupAtan2Cost

def visualize_costs():
    print("Visualizing MPC cost functions...")

    # --- Configuration ---
    Q_diag = [0.1, 5.0, 0.1, 0.1] # Example weights [x, theta, xd, thd]
    R_val = 0.01
    Q = ca.diag(Q_diag)
    R = ca.DM([R_val])
    
    # Instantiate cost calculators
    quad_cost_calc = QuadraticCost()
    swingup_cost_calc = PendulumSwingupCost()
    atan2_cost_calc = PendulumSwingupAtan2Cost()
    
    X_ref = ca.DM.zeros(4, 1)
    Xk_sym = ca.SX.sym('Xk', 4)
    Uk_sym = ca.SX.sym('Uk', 1)
    
    # Create CasADi functions
    quad_cost_func = ca.Function('quad_cost', [Xk_sym, Uk_sym], [quad_cost_calc.calculate_stage_cost(Xk_sym, Uk_sym, X_ref, Q, R)])
    swingup_cost_func = ca.Function('swingup_cost', [Xk_sym, Uk_sym], [swingup_cost_calc.calculate_stage_cost(Xk_sym, Uk_sym, X_ref, Q, R)])
    atan2_cost_func = ca.Function('atan2_cost', [Xk_sym, Uk_sym], [atan2_cost_calc.calculate_stage_cost(Xk_sym, Uk_sym, X_ref, Q, R)])

    # --- Create Figure and Axes (2x3 layout) --- 
    fig, axs = plt.subplots(2, 3, figsize=(21, 10))
    fig.suptitle(f'Comparison of MPC Cost Components (Q_diag={Q_diag}, R={R_val})')

    # --- Plot 1: Cost vs. Theta (axs[0, 0]) --- 
    print("Plotting Cost vs. Theta...")
    theta_range = np.linspace(-2*np.pi - 0.5, 2*np.pi + 0.5, 400)
    costs_quad_theta = []
    costs_swingup_theta = []
    costs_atan2_theta = []
    state_base = np.array([0.0, 0.0, 0.0, 0.0])
    control_val = np.array([0.0])
    
    for theta in theta_range:
        state = state_base.copy(); state[1] = theta
        costs_quad_theta.append(quad_cost_func(state, control_val).full().item())
        costs_swingup_theta.append(swingup_cost_func(state, control_val).full().item())
        costs_atan2_theta.append(atan2_cost_func(state, control_val).full().item())
        
    ax = axs[0, 0]
    max_y_non_quad = max(np.max(costs_swingup_theta), np.max(costs_atan2_theta)) * 1.1
    min_y_non_quad = min(np.min(costs_swingup_theta), np.min(costs_atan2_theta)) - 0.5
    ax.set_ylim(min_y_non_quad, max_y_non_quad)
    
    ax.plot(theta_range, costs_swingup_theta, label='Cos Cost (1-cos)', linestyle='--')
    ax.plot(theta_range, costs_atan2_theta, label='Atan2 Cost (atan2^2)', linestyle=':')
    ax.plot(theta_range, costs_quad_theta, label='Quad Cost (theta^2)')
    ax.set_xlabel('Pole Angle (theta) [rad]')
    ax.set_ylabel('Stage Cost')
    ax.set_title('Cost vs. Pole Angle')
    for k in [-2, -1, 0, 1, 2]:
        label = f'{k}pi rad' if k != 0 else 'Upright (0 rad)'
        color = 'red' if k != 0 else 'grey'
        ax.axvline(k * np.pi, color=color, linestyle=':', label=label if k in [-1, 0, 1] else None)
    ax.legend()
    ax.grid(True)

    # --- Plot 2: Cost vs. Cart Position (axs[0, 1]) --- 
    print("Plotting Cost vs. Cart Position...")
    x_range = np.linspace(-1.5, 1.5, 100)
    costs_quad_x = []; costs_swingup_x = []; costs_atan2_x = []
    state_base = np.array([0.0, 0.0, 0.0, 0.0]); control_val = np.array([0.0])
    for x in x_range:
        state = state_base.copy(); state[0] = x
        costs_quad_x.append(quad_cost_func(state, control_val).full().item())
        costs_swingup_x.append(swingup_cost_func(state, control_val).full().item())
        costs_atan2_x.append(atan2_cost_func(state, control_val).full().item())
    ax = axs[0, 1]
    ax.plot(x_range, costs_quad_x, label='Quad Cost')
    ax.plot(x_range, costs_swingup_x, label='Cos Cost', linestyle='--') 
    ax.plot(x_range, costs_atan2_x, label='Atan2 Cost', linestyle=':')
    ax.set_xlabel('Cart Position (x) [m]')
    ax.set_ylabel('Stage Cost')
    ax.set_title('Cost vs. Cart Position')
    ax.legend()
    ax.grid(True)
    
    # --- Plot 3: Cost vs. Control Input (axs[0, 2]) --- 
    print("Plotting Cost vs. Control Input...")
    u_range = np.linspace(-4, 4, 100)
    costs_quad_u = []; costs_swingup_u = []; costs_atan2_u = []
    state_base = np.array([0.0, 0.0, 0.0, 0.0])
    for u in u_range:
        control_val = np.array([u])
        costs_quad_u.append(quad_cost_func(state_base, control_val).full().item())
        costs_swingup_u.append(swingup_cost_func(state_base, control_val).full().item())
        costs_atan2_u.append(atan2_cost_func(state_base, control_val).full().item())
    ax = axs[0, 2]
    ax.plot(u_range, costs_quad_u, label='Quad Cost')
    ax.plot(u_range, costs_swingup_u, label='Cos Cost', linestyle='--')
    ax.plot(u_range, costs_atan2_u, label='Atan2 Cost', linestyle=':')
    ax.set_xlabel('Control Input (u) [N]')
    ax.set_ylabel('Stage Cost')
    ax.set_title('Cost vs. Control Input')
    ax.legend()
    ax.grid(True)

    # --- Plot 4: Cost vs. Cart Velocity (axs[1, 0]) --- 
    print("Plotting Cost vs. Cart Velocity...")
    x_dot_range = np.linspace(-2.0, 2.0, 100)
    costs_quad_xd = []; costs_swingup_xd = []; costs_atan2_xd = []
    state_base = np.array([0.0, 0.0, 0.0, 0.0]); control_val = np.array([0.0])
    for x_dot in x_dot_range:
        state = state_base.copy(); state[2] = x_dot
        costs_quad_xd.append(quad_cost_func(state, control_val).full().item())
        costs_swingup_xd.append(swingup_cost_func(state, control_val).full().item())
        costs_atan2_xd.append(atan2_cost_func(state, control_val).full().item())
    ax = axs[1, 0]
    ax.plot(x_dot_range, costs_quad_xd, label='Quad Cost')
    ax.plot(x_dot_range, costs_swingup_xd, label='Cos Cost', linestyle='--')
    ax.plot(x_dot_range, costs_atan2_xd, label='Atan2 Cost', linestyle=':')
    ax.set_xlabel('Cart Velocity (x_dot) [m/s]')
    ax.set_ylabel('Stage Cost')
    ax.set_title('Cost vs. Cart Velocity')
    ax.legend()
    ax.grid(True)

    # --- Plot 5: Cost vs. Pole Velocity (axs[1, 1]) --- 
    print("Plotting Cost vs. Pole Velocity...")
    theta_dot_range = np.linspace(-4.0, 4.0, 100)
    costs_quad_thd = []; costs_swingup_thd = []; costs_atan2_thd = []
    state_base = np.array([0.0, 0.0, 0.0, 0.0]); control_val = np.array([0.0])
    for theta_dot in theta_dot_range:
        state = state_base.copy(); state[3] = theta_dot
        costs_quad_thd.append(quad_cost_func(state, control_val).full().item())
        costs_swingup_thd.append(swingup_cost_func(state, control_val).full().item())
        costs_atan2_thd.append(atan2_cost_func(state, control_val).full().item())
    ax = axs[1, 1]
    ax.plot(theta_dot_range, costs_quad_thd, label='Quad Cost')
    ax.plot(theta_dot_range, costs_swingup_thd, label='Cos Cost', linestyle='--')
    ax.plot(theta_dot_range, costs_atan2_thd, label='Atan2 Cost', linestyle=':')
    ax.set_xlabel('Pole Angular Velocity (theta_dot) [rad/s]')
    ax.set_ylabel('Stage Cost')
    ax.set_title('Cost vs. Pole Velocity')
    ax.legend()
    ax.grid(True)

    # --- Plot 6: Empty (axs[1, 2]) --- 
    axs[1, 2].axis('off')

    # --- Show Plot --- 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_theta_cost_comparison():
    """Visualizes only the theta component of different cost functions."""
    print("Visualizing Theta Cost Comparison (Q=diag([1,1,1,1]), R=1)...")

    # --- Simplified Configuration ---
    Q_diag = [1.0, 1.0, 1.0, 1.0] # Simplified weights
    R_val = 1.0
    Q = ca.diag(Q_diag)
    R = ca.DM([R_val])

    # Instantiate cost calculators
    quad_cost_calc = QuadraticCost()
    swingup_cost_calc = PendulumSwingupCost()
    atan2_cost_calc = PendulumSwingupAtan2Cost()

    X_ref = ca.DM.zeros(4, 1)
    Xk_sym = ca.SX.sym('Xk', 4)
    Uk_sym = ca.SX.sym('Uk', 1)

    # Create CasADi functions
    quad_cost_func = ca.Function('quad_cost', [Xk_sym, Uk_sym], [quad_cost_calc.calculate_stage_cost(Xk_sym, Uk_sym, X_ref, Q, R)])
    swingup_cost_func = ca.Function('swingup_cost', [Xk_sym, Uk_sym], [swingup_cost_calc.calculate_stage_cost(Xk_sym, Uk_sym, X_ref, Q, R)])
    atan2_cost_func = ca.Function('atan2_cost', [Xk_sym, Uk_sym], [atan2_cost_calc.calculate_stage_cost(Xk_sym, Uk_sym, X_ref, Q, R)])

    # --- Create Figure and Axes (1x1 layout) ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f'Cost vs. Pole Angle (Q_diag={Q_diag}, R={R_val})')

    # --- Plot Cost vs. Theta ---
    theta_range = np.linspace(-2*np.pi - 0.5, 2*np.pi + 0.5, 400)
    costs_quad_theta = []
    costs_swingup_theta = []
    costs_atan2_theta = []
    state_base = np.array([0.0, 0.0, 0.0, 0.0])
    control_val = np.array([0.0])

    for theta in theta_range:
        state = state_base.copy(); state[1] = theta
        costs_quad_theta.append(quad_cost_func(state, control_val).full().item())
        costs_swingup_theta.append(swingup_cost_func(state, control_val).full().item())
        costs_atan2_theta.append(atan2_cost_func(state, control_val).full().item())

    max_y_non_quad = max(np.max(costs_swingup_theta), np.max(costs_atan2_theta)) * 1.1
    min_y_non_quad = min(np.min(costs_swingup_theta), np.min(costs_atan2_theta)) - 0.5
    ax.set_ylim(min_y_non_quad, max_y_non_quad) # Keep Y axis similar to original for comparison

    ax.plot(theta_range, costs_swingup_theta, label='Cos Cost (1-cos(theta))', linestyle='--')
    ax.plot(theta_range, costs_atan2_theta, label='Atan2 Cost (atan2(sin,cos)^2)', linestyle=':')
    ax.plot(theta_range, costs_quad_theta, label='Quad Cost (theta^2)')
    ax.set_xlabel('Pole Angle (theta) [rad]')
    ax.set_ylabel('Stage Cost (State Component Only)')
    ax.set_title('Impact of Pole Angle on Different Cost Formulations')
    for k in [-2, -1, 0, 1, 2]:
        label = f'{k}π rad' if k != 0 else 'Upright (0 rad)'
        color = 'red' if k != 0 else 'grey'
        ax.axvline(k * np.pi, color=color, linestyle=':', linewidth=1, label=label if k in [-1, 0, 1] else None)
    ax.legend()
    ax.grid(True)

    # --- Show Plot ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    visualize_costs()
    visualize_theta_cost_comparison() # Add call to the new function 