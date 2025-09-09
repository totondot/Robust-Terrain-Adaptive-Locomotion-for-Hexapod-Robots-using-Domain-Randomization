import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

class HexapodEnv(gym.Env):
    def __init__(self, render=True):
        super(HexapodEnv, self).__init__()
        self.render = render
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)  # 6 legs * 3 joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        self.physics_client = p.connect(p.GUI if self.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = None
        self.robot_id = None
        self.constraint_ids = []
        self.joint_indices = []
        self.num_joints = 18
        self.dt = 1./480.  # Finer timestep
        self.max_steps = 3000
        self.current_step = 0
        self.initial_pos = [0, np.random.uniform(-0.1, 0.1), 0.4]  # Higher z for rough terrain
        self.friction = np.random.uniform(0.8, 1.2)  # Random friction
        self.mass_scale = np.random.uniform(0.8, 1.2)  # Random mass
        self.gait_phase = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            if self.plane_id is not None and self.plane_id > 0:
                p.removeBody(self.plane_id)
            if self.robot_id is not None and self.robot_id > 0:
                p.removeBody(self.robot_id)
            for cid in self.constraint_ids:
                if cid > 0:
                    p.removeConstraint(cid)
            self.constraint_ids = []

            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(self.dt)

            self.plane_id = self._load_random_terrain()
            self.robot_id = p.loadURDF(os.path.join('.', 'pexod.urdf'), self.initial_pos)
            self.joint_indices = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
            if len(self.joint_indices) != self.num_joints:
                raise RuntimeError(f"Expected {self.num_joints} joints, found {len(self.joint_indices)}")
            self._set_physics()

            neutral_pose = [0.0] * 6 + [-0.5, 0.0, -0.5, -0.5, 0.0, -0.5] + [-1.0, 0.0, -1.0, -1.0, 0.0, -1.0]
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_idx, targetValue=neutral_pose[i], targetVelocity=0.0)
            for _ in range(200):
                p.stepSimulation()

            self.current_step = 0
            self.gait_phase = 0.0
            obs = self._get_observation()
            return obs, {}
        except Exception as e:
            print(f"Reset error: {e}")
            raise

    def _load_random_terrain(self):
        try:
            # Generate higher-resolution grid (128x128) for more bumps
            coarse_size = 128
            coarse_heightfield = np.random.uniform(-0.3, 0.3, (coarse_size, coarse_size)).astype(np.float32)
            # Add fine-scale noise for roughness
            fine_noise = np.random.uniform(-0.05, 0.05, (coarse_size, coarse_size)).astype(np.float32)
            coarse_heightfield += fine_noise
            # Smooth minimally to preserve roughness
            coarse_heightfield = gaussian_filter(coarse_heightfield, sigma=1.0)
            # Interpolate to 256x256 for detail
            x = np.linspace(0, 1, coarse_size)
            y = np.linspace(0, 1, coarse_size)
            interp = RegularGridInterpolator((x, y), coarse_heightfield)
            x_new = np.linspace(0, 1, 256)
            y_new = np.linspace(0, 1, 256)
            X, Y = np.meshgrid(x_new, y_new)
            heightfield = interp((X, Y))
            heightfield_data = heightfield.ravel().tolist()
            col_shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[0.05, 0.05, 1.0],
                heightfieldTextureScaling=256,
                numHeightfieldRows=256,
                numHeightfieldColumns=256,
                heightfieldData=heightfield_data
            )
            terrain = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape)
            p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])
            p.changeDynamics(terrain, -1, lateralFriction=self.friction)
            print("Loaded rough terrain with more bumps")
            return terrain
        except Exception as e:
            print(f"Terrain load error: {e}")
            raise

    def _set_physics(self):
        for j in self.joint_indices:
            p.changeDynamics(self.robot_id, j, lateralFriction=self.friction)
        p.changeDynamics(self.robot_id, -1, mass=self.mass_scale)
        for i in range(p.getNumJoints(self.robot_id)):
            p.setCollisionFilterGroupMask(self.robot_id, i, -1, 0, 0)

    def step(self, action):
        try:
            action = np.clip(action, -0.3, 0.3)
            self.action = action
            current_pos = [p.getJointState(self.robot_id, idx)[0] for idx in self.joint_indices]

            # Gait controller: Tripod gait with sinusoidal swing
            self.gait_phase += self.dt * 2.0
            if self.gait_phase > 1.0:
                self.gait_phase -= 1.0
            gait_targets = np.zeros(18)
            for i in range(6):
                coxa_idx = i * 3
                femur_idx = coxa_idx + 1
                tibia_idx = coxa_idx + 2
                is_tripod1 = i in [0, 2, 4]  # Front-left, Middle-right, Rear-left
                phase = self.gait_phase if is_tripod1 else (self.gait_phase + 0.5) % 1.0
                swing = 0.5 * (1.0 + np.sin(2.0 * np.pi * phase - np.pi))
                if swing > 0.5:
                    gait_targets[coxa_idx] = 0.0
                    gait_targets[femur_idx] = -0.5 if is_tripod1 else 0.0
                    gait_targets[tibia_idx] = -1.0 if is_tripod1 else 0.0
                else:
                    gait_targets[coxa_idx] = 0.2 * swing
                    gait_targets[femur_idx] = -0.2 * swing
                    gait_targets[tibia_idx] = -0.5 * swing  # Higher lift for rough terrain

            # Blend gait with RL action
            target_pos = 0.7 * gait_targets + 0.3 * (np.array(current_pos) + action * 0.1)
            for i, idx in enumerate(self.joint_indices):
                current_vel = p.getJointState(self.robot_id, idx)[1]
                target_pos[i] = np.clip(target_pos[i], -1.57, 1.57) - 0.1 * current_vel
                p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=target_pos[i], force=10)

            p.stepSimulation()
            self.current_step += 1
            obs = self._get_observation()
            reward = self._compute_reward()
            terminated = self._is_terminated()
            truncated = self.current_step >= self.max_steps
            info = {"step": self.current_step}
            return obs, reward, terminated, truncated, info
        except Exception as e:
            print(f"Step error: {e}")
            raise

    def _get_observation(self):
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(base_ori)
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        obs = np.concatenate([
            np.array(base_pos),
            np.array(euler),
            np.array(base_lin_vel),
            np.array(base_ang_vel),
            np.array(joint_pos),
            np.array(joint_vel)
        ])
        obs = obs[:44]
        if len(obs) != 44:
            print(f"Warning: Observation size {len(obs)} after truncation, expected 44")
        return obs.astype(np.float32)

    def _compute_reward(self):
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(base_ori)
        vel_reward = 1.0 * (abs(base_lin_vel[0]) + abs(base_lin_vel[1]))
        pos_reward = 0.5 * (abs(base_pos[0]) + abs(base_pos[1]))
        alive_bonus = 1.0
        fall_penalty = -20.0 if base_pos[2] < 0.1 else 0.0
        ang_penalty = -0.1 * np.linalg.norm(base_ang_vel)
        ori_penalty = -0.3 * (abs(euler[0]) + abs(euler[1]))
        gait_bonus = 0.1 if abs(euler[0]) < 0.2 and abs(euler[1]) < 0.2 else -0.1
        energy_penalty = -0.0001 * np.sum(np.abs(self.action)) if hasattr(self, 'action') else 0.0
        return vel_reward + pos_reward + alive_bonus + fall_penalty + ang_penalty + ori_penalty + gait_bonus + energy_penalty

    def _is_terminated(self):
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(base_ori)
        return base_pos[2] < 0.1 or np.any(np.abs(euler) > np.pi/2)

    def close(self):
        if self.robot_id is not None and self.robot_id > 0:
            p.removeBody(self.robot_id)
        if self.plane_id is not None and self.plane_id > 0:
            p.removeBody(self.plane_id)
        for cid in self.constraint_ids:
            if cid > 0:
                p.removeConstraint(cid)
        p.disconnect()
