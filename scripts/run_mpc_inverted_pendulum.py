# run_mpc_inverted_pendulum.py
import gymnasium as gym
import numpy as np

from src.algorithms.classic.mpc_controller import MPCController


def run_mpc_on_inverted_pendulum(num_episodes=1, max_steps=200):
    """
    Demonstration script that runs the MPCController in the Gymnasium InvertedPendulum-v4 environment.
    """

    # Create the Gymnasium MuJoCo environment (change if you have a custom ID)
    env = gym.make("InvertedPendulum-v4", render_mode="human")

    # Instantiate your MPC controller. Make sure param_path matches where your
    # inverted_pendulum_params.json is located relative to this script.
    controller = MPCController(
        N=10,                      # MPC horizon
        dt=0.05,                   # Timestep (must match env if relevant)
        param_path="src/environments/inverted_pendulum_params.json"
    )

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        obs = np.array(obs, dtype=np.float64)  # ensure it's a NumPy array
        total_reward = 0.0

        for step in range(max_steps):
            # 1) Get MPC action
            #    Make sure the shape of obs aligns with what the controller expects.
            #    The standard InvertedPendulum-v4 observation is [theta, theta_dot, x, x_dot]
            #    But your controller might expect [x, theta, x_dot, theta_dot].
            #    If so, reorder accordingly, e.g.:
            # obs_controller = [obs[2], obs[0], obs[3], obs[1]]
            # But if your model matches the env exactly, just do:
            action = controller.step(obs)

            # 2) Step the environment
            #    The Gym environment expects actions in an array-like, even if scalar:
            obs, reward, done, truncated, info = env.step([action])

            obs = np.array(obs, dtype=np.float64)
            total_reward += reward

            # 3) (Optional) Render or log data
            env.render()

            # 4) Check if episode finished
            if done or truncated:
                print(f"Episode {episode} finished in {step+1} steps, total reward {total_reward:.2f}")
                break

    env.close()


if __name__ == "__main__":
    # You can tweak these parameters or parse them from sys.argv, etc.
    run_mpc_on_inverted_pendulum(num_episodes=1, max_steps=200)
