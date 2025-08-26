import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import glob
import random

class HexapodEnv(gym.Env):
    """
    A Gym environment for the hexapod robot simulation in PyBullet.
    """
    def __init__(self, urdf_path, texture_path, terrains_path, time_step=1.0/240.0):
        super(HexapodEnv, self).__init__()

        self.urdf_path = urdf_path
        self.texture_path = texture_path
        self.terrains_path = terrains_path
        self.time_step = time_step
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # FIX: Add the terrains path to PyBullet's search paths to allow it to find the URDFs.
        # This is the key change to visualize the terrains.
        if os.path.exists(self.terrains_path):
            p.setAdditionalSearchPath(self.terrains_path)
            
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        self.robot_id = None
        self.joint_indices = list(range(18))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32)

        # Reward weights
        self.forward_vel_weight = 1.0
        self.height_weight = 0.5
        self.uprightness_weight = 0.1
        self.joint_vel_weight = 0.001
        self.desired_height = 0.3
        self.max_steps = 2048
        self.current_steps = 0
        

    def _get_obs(self):
        """
        Returns the current observation of the environment.
        """
        # Get robot state
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot_id)
        robot_vel, robot_angular_vel = p.getBaseVelocity(self.robot_id)
        
        # Get joint states
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        # Concatenate all observations
        obs = np.concatenate([
            robot_pos,
            robot_ori,
            robot_vel,
            robot_angular_vel,
            joint_positions,
            joint_velocities
        ]).astype(np.float32)

        return obs

    def _compute_reward(self):
        """
        Calculates the reward based on the current state of the robot.
        """
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot_id)
        robot_vel, _ = p.getBaseVelocity(self.robot_id)
        
        # Calculate forward velocity reward
        forward_vel_reward = self.forward_vel_weight * robot_vel[0]

        # Calculate height reward
        height_reward = -self.height_weight * abs(robot_pos[2] - self.desired_height)

        # Calculate uprightness penalty
        pitch = p.getEulerFromQuaternion(robot_ori)[1]
        roll = p.getEulerFromQuaternion(robot_ori)[0]
        uprightness_penalty = -self.uprightness_weight * (abs(pitch) + abs(roll))

        # Calculate joint velocity penalty
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_velocities = np.array([state[1] for state in joint_states])
        joint_vel_penalty = -self.joint_vel_weight * np.sum(np.square(joint_velocities))

        # Sum all rewards
        total_reward = forward_vel_reward + height_reward + uprightness_penalty + joint_vel_penalty

        return total_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()

        # Implement domain randomization
        # This now relies on the terrains path being added to PyBullet's search paths
        if os.path.exists(self.terrains_path) and os.path.isdir(self.terrains_path):
            terrain_files = glob.glob(os.path.join(self.terrains_path, "*.urdf"))
            if terrain_files:
                random_terrain_file = random.choice(terrain_files)
                p.loadURDF(random_terrain_file)
            else:
                p.loadURDF("plane.urdf")
        else:
            p.loadURDF("plane.urdf")
        
        if os.path.exists(self.texture_path):
            texture_id = p.loadTexture(self.texture_path)
            p.changeVisualShape(bodyUniqueId=0, linkIndex=-1, textureUniqueId=texture_id)
        
        start_pos = [0, 0, 0.5]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        
        # Check if the URDF path is valid before loading
        if os.path.exists(self.urdf_path):
            self.robot_id = p.loadURDF(self.urdf_path, start_pos, start_ori, useFixedBase=False)
        else:
            print(f"Error: URDF file not found at {self.urdf_path}. Cannot spawn robot.")
            # Set robot_id to None so subsequent functions don't fail
            self.robot_id = None


        # Only try to reset joints if the robot was successfully loaded
        if self.robot_id is not None:
            for i in range(p.getNumJoints(self.robot_id)):
                p.resetJointState(self.robot_id, i, 0.0)
        
        # Reset step counter
        self.current_steps = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """
        Takes a step in the environment given an action.
        """
        # Apply action to joints only if the robot exists
        if self.robot_id is not None:
            # FIX: Reduce the force to prevent the robot from flying off
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=action,
                forces=[10] * 18 # Changed from 500 to 10
            )
        
        p.stepSimulation()
        self.current_steps += 1

        # Get state and reward
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.current_steps >= self.max_steps
        info = {}

        return obs, reward, terminated, truncated, info
    
    def _is_terminated(self):
        """
        Checks if the episode has terminated.
        """
        # Only check if the robot exists in the simulation
        if self.robot_id is not None:
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            # Terminate if the robot falls over
            if robot_pos[2] < 0.15:
                return True
        return False

    def close(self):
        p.disconnect(self.client)

