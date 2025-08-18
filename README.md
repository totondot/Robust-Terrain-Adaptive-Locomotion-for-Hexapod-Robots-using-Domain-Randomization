**Robust-Terrain-Adaptive-Locomotion-for-Hexapod-Robots-using-Domain-Randomization**

This project focuses on training a hexapod robot to walk robustly over various terrains. It utilizes the PyBullet physics engine for simulation and a reinforcement learning framework, specifically Proximal Policy Optimization (PPO), to achieve terrain-adaptive locomotion through domain randomization.

Prerequisites
To run this simulation, you will need to have Python and the following libraries installed.

**pybullet: **The core physics simulation engine for the environment.
**numpy: **Used for all numerical operations and data handling.
**gymnasium: **Provides the standard API for reinforcement learning environments.
**stable-baselines3: **An implementation of popular reinforcement learning algorithms, including PPO, which is used for training the robot.

File Sequence
The project is structured as a Python package. Here is the purpose of each main file:
**main.py:** This is the top-level training script. It sets up the reinforcement learning environment and trains a PPO agent to control the hexapod.
**hexapod_env.py:** This file defines the HexapodEnv class, a custom Gymnasium environment. It handles the low-level interactions with the PyBullet physics engine and defines the observation space, action space, and reward function for the RL agent.
**pexod.urdf:** This is the Universal Robot Description Format file for your hexapod. It's the blueprint for the robot, defining its links, joints, and physical properties. It is loaded by hexapod_env.py.
**generate_raw_heightmap.py:** This is a utility script used to create different terrains for the simulation. It is crucial for the domain randomization aspect, as it provides varied environments for the agent to train on.
**__init__.py:** A standard Python file that marks the hexapod_env directory as a Python package.

Installation on Windows
Install Python:
Download and install Python from the official website: https://www.python.org/downloads/
Make sure to check the box that says "Add Python to PATH" during installation.
Install Required Libraries:
Open a Command Prompt or PowerShell and run the following commands:
pip install pybullet
pip install numpy
pip install gymnasium
pip install stable-baselines3


Installation on Arch Linux
Install Python:
Python is usually pre-installed on Arch Linux. If not, you can install it using pacman:
sudo pacman -S python


Install Required Libraries:
Open a terminal and use pip to install the libraries:
pip install pybullet
pip install numpy
pip install gymnasium
pip install stable-baselines3


Usage
Once you have installed all the dependencies, you can run the simulation by executing the main.py script.
python main.py


This will start the training process for the PPO agent in the PyBullet GUI. To exit the simulation, simply close the PyBullet GUI window.
