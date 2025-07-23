import gym
from gym import spaces
import pybullet as p
import numpy as np
import os
import random
import pybullet_data

class HexapodEnv(gym.Env):
    def __init__(self, render=False, time_step=0.01):
        super(HexapodEnv, self).__init__()

        self.render = render
        self.time_step = time_step
        self.physicsClient = None
        self.robot_id = None

        # Define action space and observation space
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                                       high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def generate_random_heightmap(self, size=256, max_height=0.2):
        """
        Generate random heightmap data.
        :param size: Size of the grid (heightmap)
        :param max_height: Max height for the terrain
        :return: Random heightmap as flattened array
        """
        heightmap = np.random.uniform(low=-max_height, high=max_height, size=(size, size))
        return heightmap.flatten().tolist()

    def load_random_heightmap(self):
        """
        Select and load a random heightmap from the heightmaps folder.
        """
        heightmaps_folder = os.path.join(os.path.dirname(__file__), "../heightmaps")
        heightmap_files = [f for f in os.listdir(heightmaps_folder) if f.endswith(".raw")]

        # Choose a random file from the list
        selected_file = random.choice(heightmap_files)
        selected_file_path = os.path.join(heightmaps_folder, selected_file)
        
        # Load the heightmap file into PyBullet
        terrain_id = p.createHeightfieldTerrainFromFile(
            selected_file_path, 
            numHeightfieldZ=256,  # Grid size of the heightmap
            heightScale=1.0
        )

        print(f"Loaded terrain: {selected_file}")
        return terrain_id

    def reset(self, seed=None, options=None):
        def add_obstacles(self):
    # Add obstacles like cubes on the terrain
            for _ in range(5):  # Add 5 random obstacles
                x = np.random.uniform(-2, 2)
                y = np.random.uniform(-2, 2)
                z = np.random.uniform(0.2, 0.5)
                size = np.random.uniform(0.05, 0.1)
                p.createCollisionShape(p.GEOM_BOX, halfExtents=[size] * 3)
                p.createMultiBody(basePosition=[x, y, z])

        if self.physicsClient is not None:
            p.disconnect(self.physicsClient)

        self.physicsClient = p.connect(p.GUI if self.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        # COMMENT OUT or REMOVE the heightfield loading:
        # self.load_random_heightmap()

        # CREATE SIMPLE FLAT TERRAIN:
        terrain_id = p.createCollisionShape(p.GEOM_PLANE)
        terrain_visual_id = p.createVisualShape(p.GEOM_PLANE)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_id, baseVisualShapeIndex=terrain_visual_id)

        # Load hexapod model
        hexapod_urdf = os.path.join(os.path.dirname(__file__), "../hexapod_model/pexod.urdf")
        self.robot_id = p.loadURDF(hexapod_urdf, basePosition=[0, 0, 0.2], useFixedBase=False)

        # Return an initial observation (replace with actual state)
        observation = np.zeros(12)
        return observation, {}

    def step(self, action):
        # Apply action to the environment (control the hexapod)
        # For now, you should apply actions to your robot here
        # e.g., applying forces to the hexapod's joints
        
        # Get the current state of the environment (this is your observation)
        observation = np.zeros(12)  # Update with actual robot state

        # Calculate reward (for now it's a dummy value)
        reward = 1.0

        # Check if the episode is done (you can add your logic here)
        done = False

        # Truncated indicates if the episode is truncated (e.g., if max time is exceeded)
        truncated = False  # Set to True if the environment is truncated based on some condition

        # Return the observation, reward, done, truncated, and info (empty)
        info = {}
        return observation, reward, done, truncated, info
    def render(self, mode="human"):
        """
        Render the environment (if applicable)
        """
        pass
