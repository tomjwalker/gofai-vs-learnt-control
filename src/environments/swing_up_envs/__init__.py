"""
Custom swing-up environments for Gymnasium MuJoCo tasks.

This package defines wrappers around standard MuJoCo environments to
create swing-up tasks (starting from a downward position) instead of
just balancing tasks.

Environments are registered when the respective modules are imported,
OR they can be registered explicitly by calling the registration functions
(e.g., register_pendulum_swing_up) defined within each module.

Explicit registration is needed when using multiprocessing with the 'spawn'
start method (default on Windows), as imports in the main process might not
trigger registration in the child processes.
"""

# We no longer rely on automatic registration via import here.
# Registration will be handled explicitly in the training/evaluation scripts
# or within the environment-making functions used by subprocesses.

# from . import pendulum_su
# from . import double_pendulum_su

from gymnasium.envs.registration import register
from .pendulum_su import make_env

# register(
#     id="Pendulum-SwingUp",
#     entry_point="src.environments.swing_up_envs.pendulum_su:make_env",
#     max_episode_steps=500,
# )
