import gymnasium as gym
import numpy as np

"""
This script interrogates the MuJoCo InvertedPendulum-v4 environment to:
1. Inspect and print out all bodies, joints, and actuators.
2. Extract key parameters (cart mass, pole mass, pole length, inertia, etc.) 
   that we need for a 2D cart-pole dynamics or MPC model.
"""

# Create the environment (with a human render for visual debug)
env = gym.make("InvertedPendulum-v4", render_mode="human")
env.reset()

# Access the MuJoCo model directly
model = env.unwrapped.model

# MuJoCo geom type ID mapping (from MuJoCo docs)
geom_type_dict = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh"
}

print("\n==== MUJOCO INVERTED PENDULUM MODEL INSPECTION ====\n")

# -----------------------------------------------------------------------------------
#   1. Body Info (Masses, Positions, Inertias)
# -----------------------------------------------------------------------------------
print("== Body Masses and Basic Info ==")
cart_mass = None
pole_mass = None
pole_inertia_y = None  # The inertia around the y-axis is relevant if we do a 2D model

for i in range(model.nbody):
    name = model.body(i).name
    mass = model.body_mass[i]
    inertia = model.body_inertia[i]
    pos = model.body_pos[i]

    print(f"Body {i}: '{name}'")
    print(f"    mass = {mass:.4f} kg")
    print(f"    inertia (Ix, Iy, Iz) = {inertia}")
    print(f"    pos (initial offset) = {pos}\n")

    # Identify cart/pole by name (this is how Gym usually labels them)
    if name == "cart":
        cart_mass = mass
    elif name == "pole":
        pole_mass = mass
        # For a single hinge axis (y-axis in this env), the second inertia entry
        # is the relevant rotational inertia if we strictly do a 2D derivation.
        pole_inertia_y = inertia[1]

# -----------------------------------------------------------------------------------
#   2. Geom Info (Shapes, Sizes)
# -----------------------------------------------------------------------------------
print("== Geom Types and Sizes ==")
pole_half_length = None

for i in range(model.ngeom):
    name = model.geom(i).name
    geom_type_id = model.geom_type[i]
    geom_type = geom_type_dict.get(geom_type_id, f"unknown ({geom_type_id})")
    size = model.geom_size[i]
    print(f"Geom {i}: '{name}' — type = {geom_type}, size = {size}")

    # If it's the pole geometry, we can read the 'half-length' from size[1].
    if name == "cpole":
        # MuJoCo typically sets capsule shape with size[1] = half-length of cylinder
        pole_half_length = size[1]

print("")

# -----------------------------------------------------------------------------------
#   3. Joint Info
# -----------------------------------------------------------------------------------
print("== Joints ==")
for i in range(model.njnt):
    name = model.joint(i).name
    jtype = model.jnt_type[i]
    axis = model.jnt_axis[i]
    jpos = model.jnt_pos[i]
    print(f"Joint {i}: '{name}'")
    print(f"    type = {jtype} "
          f"(2=slide, 3=hinge, etc.)")
    print(f"    axis = {axis}  (the direction the joint moves/rotates about)")
    print(f"    pos  = {jpos}\n")

# -----------------------------------------------------------------------------------
#   4. Actuator Info (Mapping Control Action -> Force/Torque)
# -----------------------------------------------------------------------------------
print("== Actuators ==")
for i in range(model.nu):
    # gain and bias define how your action u is converted to a force or torque
    gain = model.actuator_gainprm[i]
    bias = model.actuator_biasprm[i]
    print(f"Actuator {i}: ")
    print(f"    gain parameters = {gain}")
    print(f"    bias parameters = {bias}")
    print("    (Force = gain[0] * u + bias[0], if other entries are zero.)\n")

print("====================================================\n")

# -----------------------------------------------------------------------------------
#   5. Extracted Key Parameters
# -----------------------------------------------------------------------------------
print("== Key Parameters for 2D Cart-Pole Model ==")
print(f"Cart Mass (M)    = {cart_mass:.4f} kg")
print(f"Pole Mass (m)    = {pole_mass:.4f} kg")

pole_length = None
if pole_half_length is not None:
    pole_length = 2.0 * pole_half_length
    print(f"Pole Length (l)  = {pole_length:.4f} m "
          f"(2 x half-length from MuJoCo geom)")

# If you want to do a more advanced 2D derivation with actual inertia:
if pole_inertia_y is not None:
    print(f"Pole Inertia about y-axis = {pole_inertia_y:.4f} kg·m^2 "
          "(MuJoCo's 3D inertia, ignoring other axes for 2D setup)")
else:
    print("No pole inertia found or identified!")

print("\n(For a simpler cart-pole approximation, you'd often treat the pole as a uniform rod with I = 1/3 m l^2.)\n")

# Done: close environment
env.close()
