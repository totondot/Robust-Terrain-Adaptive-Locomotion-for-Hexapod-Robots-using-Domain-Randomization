from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from hexapod_env.hexapod_env import HexapodEnv

def make_env():
    return HexapodEnv(render=True)

# Wrap the environment in a vectorized wrapper (even though you're using just one environment)
env = DummyVecEnv([make_env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

env.close()
