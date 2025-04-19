import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.environments.casadi_dynamics import pendulum_dynamics

def simulate_pendulum(initial_theta=0.1, dt=0.01, duration=5.0):
    """
    Simulate the pendulum dynamics without control input.
    
    Args:
        initial_theta: Initial angle from vertical (radians). 
                      Note: θ=0 is upward vertical (inverted position)
        dt: Time step (seconds)
        duration: Total simulation time (seconds)
    """
    # Define parameters (using standard cartpole values)
    params = {
        'cart_mass': 1.0,
        'pole_mass': 0.1,
        'pole_half_length': 0.5,
        'pole_inertia_about_y': 0.0333,  # For a uniform rod: (1/12)*m*(2L)^2
        'gravity': 9.81
    }
    
    # Initial state: [x, theta, x_dot, theta_dot]
    x0 = np.array([0.0, initial_theta, 0.0, 0.0])
    
    # Convert to CasADi symbolic variables
    x = ca.SX.sym('x', 4)
    u = ca.SX.sym('u', 1)
    
    # Create the dynamics function
    f = ca.Function('f', [x, u], [pendulum_dynamics(x, u, dt, params)])
    
    # Simulate
    n_steps = int(duration / dt)
    states = np.zeros((n_steps, 4))
    states[0] = x0
    
    for i in range(1, n_steps):
        # No control input
        states[i] = np.array(f(states[i-1], 0.0)).flatten()
    
    return states, dt

def plot_states(states, dt):
    """Plot the state trajectories."""
    time = np.arange(len(states)) * dt
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Pendulum State Trajectories')
    
    # Position
    axs[0, 0].plot(time, states[:, 0])
    axs[0, 0].set_title('Cart Position (x)')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Position (m)')
    
    # Angle
    axs[0, 1].plot(time, states[:, 1])
    axs[0, 1].set_title('Pole Angle (θ)')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Angle (rad)')
    
    # Velocity
    axs[1, 0].plot(time, states[:, 2])
    axs[1, 0].set_title('Cart Velocity (x_dot)')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Velocity (m/s)')
    
    # Angular velocity
    axs[1, 1].plot(time, states[:, 3])
    axs[1, 1].set_title('Pole Angular Velocity (θ_dot)')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Angular Velocity (rad/s)')
    
    plt.tight_layout()
    plt.show()

def animate_pendulum(states, dt):
    """Create an animation of the pendulum."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set up the plot
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Create the cart and pole
    cart_width = 0.2
    cart_height = 0.1
    pole_length = 1.0  # Full length
    
    cart = plt.Rectangle((0, 0), cart_width, cart_height, fill=True)
    pole, = ax.plot([], [], 'b-', linewidth=2)
    ax.add_patch(cart)
    
    def update(frame):
        x = states[frame, 0]
        theta = states[frame, 1]
        
        # Update cart position
        cart.set_xy((x - cart_width/2, -cart_height/2))
        
        # Update pole position - note: theta=0 is upward vertical
        pole_x = [x, x + pole_length * np.sin(theta)]
        pole_y = [0, pole_length * np.cos(theta)]  # Changed sign here
        pole.set_data(pole_x, pole_y)
        
        return cart, pole
    
    anim = FuncAnimation(fig, update, frames=len(states),
                        interval=dt*1000, blit=True)
    plt.show()
    return anim

if __name__ == "__main__":
    # Simulate with a small initial angle
    states, dt = simulate_pendulum(initial_theta=0.1, dt=0.01, duration=5.0)
    
    # Plot the states
    plot_states(states, dt)
    
    # Create animation
    anim = animate_pendulum(states, dt) 