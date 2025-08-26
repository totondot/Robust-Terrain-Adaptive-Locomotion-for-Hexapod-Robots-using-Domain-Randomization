import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from hexapod_env.hexapod_env import HexapodEnv

# --- Training Configuration ---
LOG_DIR = "./logs/ppo_hexapod"
CHECKPOINT_DIR = "./checkpoints/ppo_hexapod"
TOTAL_TIMESTEPS = 1000000

def get_latest_checkpoint(checkpoint_dir):
    """
    Finds the most recent checkpoint file in the specified directory.
    """
    # Get a list of all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    
    if not checkpoint_files:
        return None
        
    # Find the file with the highest step number in its name
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint

def main():
    # Corrected File Paths to work on Arch
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # FIX: Correct the URDF path based on your screenshot
    URDF_PATH = os.path.join(PROJECT_ROOT, "hexapod_env", "pexod.urdf")
    
    TEXTURE_PATH = os.path.join(PROJECT_ROOT, "hexapod_model", "checkerboard.png")
    TERRAINS_PATH = os.path.join(PROJECT_ROOT, "terrains")
    
    # Check if the paths are valid before running
    print("Verifying file paths:")
    print(f"URDF Path: {URDF_PATH} - Exists: {os.path.exists(URDF_PATH)}")
    print(f"Texture Path: {TEXTURE_PATH} - Exists: {os.path.exists(TEXTURE_PATH)}")
    print(f"Terrains Path: {TERRAINS_PATH} - Exists: {os.path.exists(TERRAINS_PATH)}")

    # Create the environment
    env = make_vec_env(lambda: HexapodEnv(URDF_PATH, TEXTURE_PATH, TERRAINS_PATH), n_envs=1)
    
    # Create a checkpoint callback to save the model periodically
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=CHECKPOINT_DIR, name_prefix='hexapod_model')

    # Initialize the PPO model
    # Check if a saved model exists and load it
    latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"Loading existing model from {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, device="cpu")
    else:
        print("No existing model found, creating a new one...")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, device="cpu")
    
    print("Starting training...")
    
    # Train the agent
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    
    print("Training finished.")

if __name__ == "__main__":
    main()
