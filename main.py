import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from hexapod_env import HexapodEnv

try:
    # Create env
    env = make_vec_env(lambda: HexapodEnv(render=True), n_envs=1)

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99
    )

    # Eval callback
    eval_env = HexapodEnv(render=False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train
    model.learn(total_timesteps=200000, callback=eval_callback)

    # Save
    model.save("ppo_hexapod_flat")

    # Test
    obs, _ = env.reset()
    for step in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if step % 100 == 0:
            print(f"Step {step}, Reward: {rewards}, Info: {info}")
        if terminated or truncated:
            obs, _ = env.reset()

except Exception as e:
    print(f"Training error: {e}")
finally:
    env.close()
    print("Training complete. Check ppo_hexapod_flat.zip for the model.")
