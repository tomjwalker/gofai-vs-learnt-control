import gymnasium as gym

env = gym.make("InvertedPendulum-v4")
env.reset()

# MuJoCo underlying model properties:
model = env.unwrapped.model

# Masses of bodies (cart and pole)
body_names = model.body_names
body_masses = model.body_mass

# Number of bodies
n_bodies = model.nbody

# Masses
print("== Body Masses ==")
for i in range(n_bodies):
    name = model.body(i).name
    mass = model.body_mass[i]
    print(f"Body '{name}': mass = {mass}")

# Lengths / positions of bodies
body_pos = model.body_pos
for name, pos in zip(body_names, body_pos):
    print(f"Body '{name}': position = {pos}")

# Geom properties (e.g. pole length, COM)
geom_names = model.geom_names
geom_size = model.geom_size
for name, size in zip(geom_names, geom_size):
    print(f"Geom '{name}': size/dimensions = {size}")

# Check inertia if needed
body_inertia = model.body_inertia
for name, inertia in zip(body_names, body_inertia):
    print(f"Body '{name}': inertia = {inertia}")
