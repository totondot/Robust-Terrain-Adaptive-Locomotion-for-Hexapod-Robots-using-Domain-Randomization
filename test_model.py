import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from hexapod_env.hexapod_env import HexapodEnv

# --- Configuration ---
# Set the path to the trained model.
# Note: You may need to change this if you saved your model with a different name.
MODEL_PATH = "ppo_hexapod.zip"

# The number of episodes to test for
NUM_EPISODES_TO_TEST = 5

# --- Environment Setup ---
def make_env():
    """
    Function to create and return a single instance of the hexapod environment.
    The render=True flag ensures the simulation window is displayed.
    """
    return HexapodEnv(render=True)

if not os.path.exists(MODEL_PATH):
    print(f"Error: The model file '{MODEL_PATH}' was not found.")
    print("Please make sure you have trained a model and saved it using model.save('ppo_hexapod.zip').")
    exit()

# Wrap the environment in a vectorized wrapper, which is required by Stable Baselines3
# Even for a single environment, this is good practice.
env = DummyVecEnv([make_env])

# --- Model Loading ---
print(f"Loading the trained model from: {MODEL_PATH}")
try:
    # Load the PPO model from the specified path
    model = PPO.load(MODEL_PATH, env=env)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load the model. Error: {e}")
    print("Please ensure the model file is not corrupted and is in the correct directory.")
    env.close()
    exit()

# --- Testing Loop ---
print(f"\nStarting test run for {NUM_EPISODES_TO_TEST} episodes.")

for episode in range(NUM_EPISODES_TO_TEST):
    print(f"--- Episode {episode + 1}/{NUM_EPISODES_TO_TEST} ---")
    
    # Reset the environment at the beginning of each episode
    obs = env.reset()
    done = False
    
    # Run the simulation for a single episode
    while not done:
        # Get the action from the trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment with the action from the model
        obs, reward, done, info = env.step(action)
        
        # You can add a small sleep here to slow down the simulation if needed
        # time.sleep(0.01)

    print(f"Episode {episode + 1} finished.")
    
# --- Cleanup ---
print("\nTest run complete. Closing the environment.")
env.close()
