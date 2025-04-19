"""
Swing-up environments package
"""

from gymnasium.envs.registration import register
from .pendulum_su import make_env

register(
    id="Pendulum-SwingUp",
    entry_point="src.environments.swing_up_envs.pendulum_su:make_env",
    max_episode_steps=500,
)
