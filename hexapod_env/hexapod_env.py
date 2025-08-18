import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
import random

# Get the absolute path to the directory containing this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class HexapodEnv(gym.Env):
    def __init__(self, render=False, time_step=0.01):
        super(HexapodEnv, self).__init__()

        self.render = render
        self.time_step = time_step
        self.physicsClient = p.connect(p.GUI if self.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(numSolverIterations=500)

        self.robot_id = -1
        self.num_joints = 18
        self.joint_indices = list(range(self.num_joints))
        self.joint_names = [f"joint_{i}" for i in self.joint_indices]
        self.joint_limits = [-1.57, 1.57]

        self.terrain_id = -1
        self.episode_counter = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)

        num_obs = 3 + 4 + 3 + 3 + self.num_joints + self.num_joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32)

    def load_random_heightmap(self):
        heightmaps_folder = os.path.join(CURRENT_DIR, "heightmaps")
        
        if not os.path.exists(heightmaps_folder) or not os.listdir(heightmaps_folder):
            print("No heightmap files found. Please run generate_raw_heightmap.py first.")
            return -1
        
        heightmap_files = [f for f in os.listdir(heightmaps_folder) if f.endswith(".raw")]
        selected_file = random.choice(heightmap_files)
        selected_file_path = os.path.join(heightmaps_folder, selected_file)

        width = 256
        length = 256
        with open(selected_file_path, 'rb') as f:
            heightfield_data = np.frombuffer(f.read(), dtype=np.float16).reshape((width, length))

        heightfield_data_pybullet = heightfield_data.flatten().tolist()
        
        terrain_shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD, 
            meshScale=[0.5, 0.5, 1],
            heightfieldTextureScaling=width,
            heightfieldData=heightfield_data_pybullet,
            numHeightfieldRows=width,
            numHeightfieldColumns=length
        )
        
        terrain_id = p.createMultiBody(
            baseMass=0, 
            baseCollisionShapeIndex=terrain_shape, 
            basePosition=[0, 0, 0]
        )

        # FIX: Load the texture from the local directory
        texture_path = os.path.join(CURRENT_DIR, "checkerboard.png")
        if os.path.exists(texture_path):
            p.changeVisualShape(terrain_id, -1, textureUniqueId=p.loadTexture(texture_path))
        else:
            print(f"Warning: Texture file not found at {texture_path}. Using a solid color.")
            p.changeVisualShape(terrain_id, -1, rgbaColor=[0.8, 0.8, 0.8, 1])
        
        return terrain_id

    def get_observation(self):
        if self.robot_id == -1:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        position, orientation_quat = p.getBasePositionAndOrientation(self.robot_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_id)

        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        observation = np.concatenate([
            np.array(position),
            np.array(orientation_quat),
            np.array(linear_velocity),
            np.array(angular_velocity),
            np.array(joint_positions),
            np.array(joint_velocities)
        ]).astype(np.float32)

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        if self.terrain_id != -1:
            try:
                p.removeBody(self.terrain_id)
            except p.error:
                pass

        self.terrain_id = self.load_random_heightmap()
        if self.terrain_id == -1:
            print("Using default flat plane as a fallback.")
            self.terrain_id = p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.5]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        
        urdf_path = os.path.join(CURRENT_DIR, "pexod.urdf")

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}.")

        self.robot_id = p.loadURDF(urdf_path, start_pos, start_ori)
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(self.num_joints))

        for i in self.joint_indices:
            p.resetJointState(self.robot_id, i, 0.0)

        observation = self.get_observation()
        self.initial_base_pos = observation[:3].copy()
        
        info = {}
        return observation, info

    def step(self, action):
        if self.robot_id == -1:
            return self.get_observation(), 0, True, False, {}

        scaled_action = self.joint_limits[0] + (action + 1.0) * (self.joint_limits[1] - self.joint_limits[0]) / 2.0
        
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=scaled_action,
            forces=[500] * self.num_joints
        )

        p.stepSimulation()

        observation = self.get_observation()
        current_base_pos = observation[:3]
        
        forward_progress = current_base_pos[0] - self.initial_base_pos[0]
        
        pitch, roll, _ = p.getEulerFromQuaternion(observation[3:7])
        pitch_roll_penalty = - (pitch**2 + roll**2) * 0.1

        joint_vels = observation[18:]
        joint_vel_penalty = - np.sum(np.abs(joint_vels)) * 0.001

        reward = forward_progress + pitch_roll_penalty + joint_vel_penalty

        done = current_base_pos[2] < 0.15 or np.abs(pitch) > 0.5 or np.abs(roll) > 0.5
        truncated = False

        info = {}
        return observation, reward, done, truncated, info

    def close(self):
        p.disconnect()
