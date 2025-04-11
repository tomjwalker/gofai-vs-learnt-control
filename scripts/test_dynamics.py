import numpy as np
from src.environments.dynamics import pendulum_dynamics
from src.utils.parameters import load_inverted_pendulum_params

# Load parameters clearly
params = load_inverted_pendulum_params("../src/environments/inverted_pendulum_params.json")

# Define initial state and control input
x0 = np.array([0.0, 0.1, 0.0, 0.0])  # [x, theta, x_dot, theta_dot]. Small initial angle.
u = 0.0 # No control input
dt = 0.01  # Timestep for Euler integration

# Simulate a few steps
x = x0.copy()
for i in range(100):
    x = pendulum_dynamics(x, u, dt, params)
    print(f"Step {i+1}: x = {x}")
