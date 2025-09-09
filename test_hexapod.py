import gymnasium as gym
from stable_baselines3 import PPO
from hexapod_env import HexapodEnv
import pybullet as p
import numpy as np
import time

# Load trained model
model = PPO.load("ppo_hexapod_flat")
env = HexapodEnv(render=True)

# Initialize control variables
obs, _ = env.reset()
total_reward = 0
steer = 0.0  # -1.0 (left) to 1.0 (right)
speed_factor = 1.2  # Moderate fast baseline

print("Manual Controls (Click PyBullet GUI window to focus for arrow keys):")
print("- LEFT ARROW: Turn left")
print("- RIGHT ARROW: Turn right")
print("- UP ARROW: Increase forward speed (max 2.0)")
print("- DOWN ARROW: Decrease forward speed (min 0.5)")
print("- SPACE: Pause/resume")
print("- ESC: Quit")
print("Steer will decay naturally when no keys are pressed.")

# Test for 5000 steps (or until ESC)
for step in range(5000):
    keys = p.getKeyboardEvents()
    
    # Steering: LEFT ARROW (65361) with WAS_TRIGGERED
    if 65361 in keys and keys[65361] & p.KEY_WAS_TRIGGERED:
        steer = np.clip(steer - 0.03, -1.0, 1.0)  # Turn left
    # Steering: RIGHT ARROW (65363) with WAS_TRIGGERED
    if 65363 in keys and keys[65363] & p.KEY_WAS_TRIGGERED:
        steer = np.clip(steer + 0.03, -1.0, 1.0)  # Turn right
    # Speed: UP ARROW (65362) increase
    if 65362 in keys and keys[65362] & p.KEY_WAS_TRIGGERED:
        speed_factor = np.clip(speed_factor + 0.05, 0.5, 2.0)  # Fixed max 2.0
    # Speed: DOWN ARROW (65364) decrease
    if 65364 in keys and keys[65364] & p.KEY_WAS_TRIGGERED:
        speed_factor = np.clip(speed_factor - 0.05, 0.5, 2.0)
    # Pause: SPACE (32)
    if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
        print("Paused. Press SPACE to resume.")
        while True:
            keys = p.getKeyboardEvents()
            if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
                break
            time.sleep(0.1)
        continue
    # Quit: ESC (27)
    if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
        print("ESC pressed. Quitting.")
        break
    
    # Natural decay for steer (stops turning when no input)
    steer *= 0.95  # Quicker decay for responsive stopping
    
    # Get action from model
    action, _ = model.predict(obs, deterministic=True)
    
    # Apply steering to coxa joints (left legs negative, right positive for turn)
    for leg in range(6):
        coxa_idx = leg * 3
        side_bias = -1 if leg % 2 == 0 else 1  # Left legs (0,2,4): negative for left turn
        action[coxa_idx] += steer * side_bias * 0.3  # Stronger steering influence
    
    # Apply moderated speed factor to all actions (safer amplification)
    action *= 1.2 * speed_factor  # Reduced from 1.5 to prevent overdrive
    
    action = np.clip(action, -0.3, 0.3)

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if step % 200 == 0:
        # Safe indexing for position and velocity from observation
        if len(obs) >= 11:
            base_pos_x = obs[0]
            base_pos_y = obs[1]
            base_vel_x = obs[8]  # Linear velocity x (index 8 in observation)
            print(f"Step {step}: Pos ({base_pos_x:.2f}, {base_pos_y:.2f}), Vel X {base_vel_x:.2f}, Steer {steer:.2f}, Speed {speed_factor:.2f}, Reward {reward:.2f}")
        else:
            print(f"Step {step}: Pos N/A, Vel N/A, Steer {steer:.2f}, Speed {speed_factor:.2f}, Reward {reward:.2f}")
    
    if terminated or truncated:
        obs, _ = env.reset()
        print(f"Episode reset at step {step}, total reward so far: {total_reward:.2f}")
    
    time.sleep(0.02)  # Slightly slower for better control visibility

print(f"Test complete. Final total reward: {total_reward:.2f}")
env.close()
