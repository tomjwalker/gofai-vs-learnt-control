from setuptools import setup, find_packages

setup(
    name="gofai-vs-learnt-control",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "gymnasium[mujoco]",
        "stable-baselines3",
        "casadi",
        "matplotlib",
        "imageio",
        "Pillow",
        "dash",
    ],
    python_requires=">=3.8",
    include_package_data=True,
) 