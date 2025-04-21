import gymnasium as gym

"""
Quick script to print basic properties from the MuJoCo model.
NOTE: This is an analysis/diagnostic script.
"""

env = gym.make("InvertedPendulum-v5")
env.reset()

# MuJoCo underlying model properties:
model = env.unwrapped.model

# Masses of bodies (cart and pole)
body_masses = model.body_mass

# Number of bodies
n_bodies = model.nbody
n_geoms = model.ngeom

# Masses
print("\n== Body Masses ==")
for i in range(n_bodies):
    name = model.body(i).name
    mass = model.body_mass[i]
    print(f"Body '{name}': mass = {mass}")

# Lengths / positions of bodies
print("\n== Body Positions ==")
for i in range(n_bodies):
    name = model.body(i).name
    pos = model.body_pos[i]
    print(f"Body '{name}': position = {pos}")

# Geom properties (e.g. pole length, COM)
geom_size = model.geom_size
print("\n== Geom Properties ==")
for i in range(n_geoms):
    name = model.geom(i).name
    size = model.geom_size[i]
    geom_type = model.geom_type[i]
    print(f"Geom '{name}' (type {geom_type}): size/dimensions = {size}")

# Check inertia if needed
print("\n== Body Inertias ==")
for i in range(n_bodies):
    name = model.body(i).name
    inertia = model.body_inertia[i]
    print(f"Body '{name}': inertia = {inertia}")

env.close() # Close env 