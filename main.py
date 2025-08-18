import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from hexapod_env.hexapod_env import HexapodEnv

# Create the HexapodEnv
def make_env():
    def _init():
        return HexapodEnv(render=True)
    return _init

# Create a vectorized environment
env = DummyVecEnv([make_env()])

# Set up the PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_hexapod_tensorboard/")

# Set up a checkpoint callback to save the model periodically
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./hexapod_checkpoints/", name_prefix="hexapod_ppo_model")

# Train the agent
try:
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
finally:
    model.save("ppo_hexapod.zip")

# Close the environment
env.close()
