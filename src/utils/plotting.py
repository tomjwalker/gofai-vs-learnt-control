import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_diagnostics(history, plots_dir, episode=0, plot_cost: bool = True):
    """Generates and saves diagnostic plots for a single episode.

    Args:
        history (list): List of dictionaries, where each dict contains step data
                        (e.g., 'obs', 'u_next'/'action'). MPC history also includes
                        'cost', 'constraint_violation'.
        plots_dir (Path or str): Directory to save the plots.
        episode (int): Episode number for titling.
        plot_cost (bool): Whether to plot the cost/constraint subplot.
    """
    print(f"Generating diagnostic plots for Episode {episode}...")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    time_indices = np.arange(len(history))
    env_states = np.array([step["obs"] for step in history if "obs" in step])
    if "u_next" in history[0]:
        controls = np.array([step["u_next"] for step in history])
        control_label = "Force (N) [MPC]"
    elif "action" in history[0]:
        controls = np.array([step["action"] for step in history])
        control_label = "Action [DRL]"
    else:
        print("Warning: Could not determine control key ('u_next' or 'action') in history.")
        controls = np.zeros(len(history))
        control_label = "Control (Unknown)"
        
    costs = None
    constraint_violations = None
    if plot_cost:
        if "cost" in history[0] and "constraint_violation" in history[0]:
            costs = np.array([step.get("cost", np.nan) for step in history])
            constraint_violations = np.array([step.get("constraint_violation", np.nan) for step in history])
        else:
            print("Warning: 'cost' or 'constraint_violation' keys missing, cannot plot cost.")
            plot_cost = False

    num_subplots = 3 if plot_cost else 2
    fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 4 * num_subplots), sharex=True)
    if num_subplots == 1:
        axs = [axs]

    if env_states.size > 0:
        state_labels = [f"State_{i}" for i in range(env_states.shape[1])]
        if env_states.shape[1] == 4:
             state_labels = ["Cart Pos (x)", "Pole Angle (th)", "Cart Vel (x_dot)", "Pole Vel (th_dot)"]
        
        for i in range(env_states.shape[1]):
            axs[0].plot(time_indices, env_states[:, i], label=state_labels[i])
        axs[0].set_title(f"Episode {episode}: Environment States")
        axs[0].set_ylabel("State Value")
        axs[0].legend()
        axs[0].grid(True)
    else:
         axs[0].set_title(f"Episode {episode}: Environment States (No Data)")


    if controls.size > 0:
        axs[1].plot(time_indices, controls, marker='.', linestyle='-')
        axs[1].set_title("Control Input / Action Over Time")
        axs[1].set_ylabel(control_label)
        axs[1].grid(True)
    else:
         axs[1].set_title("Control Input / Action Over Time (No Data)")

    if plot_cost:
        if costs is not None and constraint_violations is not None and \
           len(costs) == len(time_indices) and len(constraint_violations) == len(time_indices):
            
            ax_cost = axs[2]
            ax_constraint = ax_cost.twinx()

            finite_costs = costs[~np.isnan(costs)]
            finite_constraints = constraint_violations[~np.isnan(constraint_violations)]
            
            color_cost = 'b'
            color_constraint = 'r'

            line_cost, = ax_cost.plot(time_indices, costs, color=color_cost, label='Cost')
            ax_cost.set_ylabel('Cost', color=color_cost)
            ax_cost.tick_params(axis='y', labelcolor=color_cost)
            if finite_costs.size > 0:
                cost_min, cost_max = np.min(finite_costs), np.max(finite_costs)
                cost_range = cost_max - cost_min if cost_max > cost_min else 1.0
                ax_cost.set_ylim(cost_min - 0.1 * cost_range, cost_max + 0.1 * cost_range)
            
            line_constraint, = ax_constraint.plot(time_indices, constraint_violations, color=color_constraint, label='Constraint Violation')
            ax_constraint.set_ylabel('Constraint Violation', color=color_constraint)
            ax_constraint.tick_params(axis='y', labelcolor=color_constraint)
            if finite_constraints.size > 0:
                 constraint_min, constraint_max = np.min(finite_constraints), np.max(finite_constraints)
                 constraint_range = constraint_max - constraint_min if constraint_max > constraint_min else 1.0
                 pad = 0.1 * constraint_range
                 ax_constraint.set_ylim(max(0, constraint_min - pad), constraint_max + pad) 
            else:
                 ax_constraint.set_ylim(0, 1)

            ax_cost.set_title("MPC Cost and Constraint Violation")
            lines = [line_cost, line_constraint]
            labels = [l.get_label() for l in lines]
            ax_cost.legend(lines, labels, loc='upper right')
            ax_cost.grid(True)
        else:
            axs[2].set_title("Cost/Constraint Data Unavailable or Mismatched Length")
            axs[2].grid(True)
        
        axs[2].set_xlabel("Timestep")
    else:
        axs[1].set_xlabel("Timestep")


    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_filename = plots_dir / f"episode_{episode}_diagnostic_plots.png"
    plt.savefig(plot_filename)
    print(f"Diagnostic plot saved to: {plot_filename}")
    plt.close(fig) 