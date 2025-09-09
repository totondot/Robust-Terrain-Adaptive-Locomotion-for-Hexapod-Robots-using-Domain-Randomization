import matplotlib.pyplot as plt
import os
import numpy as np
from stable_baselines3.common.monitor import load_results

def plot_simulated_curve():
    """Fallback: Simulate a typical curve for analysis."""
    timesteps = np.linspace(0, 200000, 1000)
    # Simulate: Initial falls, improvement via gait, convergence on rough terrain
    rewards = -5 + 15 * (1 - np.exp(-timesteps / 50000)) + np.random.normal(0, 1.5, len(timesteps))
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, label='Simulated Episode Reward', color='green')
    plt.fill_between(timesteps, rewards, alpha=0.3)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Simulated Hexapod Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('simulated_learning_curve.png')
    plt.show()
    
    # Simulated stats
    mean_reward = np.mean(rewards)
    final_mean = np.mean(rewards[-200:])
    std_final = np.std(rewards[-200:])
    convergence_step = np.argmax(rewards > 5) * 200 if np.any(rewards > 5) else -1  # Approx step where reward >5
    print(f"Simulated overall mean reward: {mean_reward:.2f}")
    print(f"Simulated final 20% mean reward: {final_mean:.2f}")
    print(f"Simulated final 20% std: {std_final:.2f}")
    print(f"Approximate convergence step (reward >5): {convergence_step}")

log_dir = './logs/'
if os.path.exists(log_dir):
    try:
        results = load_results(log_dir)
        if not results.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(results['t'], results['r'], label='Episode Reward', color='blue')
            plt.fill_between(results['t'], results['r'], alpha=0.3)
            plt.xlabel('Timesteps')
            plt.ylabel('Mean Reward')
            plt.title('Hexapod Learning Curve (From Your Logs)')
            plt.legend()
            plt.grid(True)
            plt.savefig('learning_curve.png')
            plt.show()
            
            # Stats
            mean_reward = results['r'].mean()
            final_mean = results['r'].tail(20).mean() if len(results) > 20 else mean_reward
            std_final = results['r'].tail(20).std() if len(results) > 20 else 0
            print(f"Overall mean reward: {mean_reward:.2f}")
            print(f"Final 20% mean reward: {final_mean:.2f}")
            print(f"Final 20% std: {std_final:.2f}")
            print("Plot saved as learning_curve.png and displayed.")
        else:
            print("No data in logs. Plotting simulated curve instead.")
            plot_simulated_curve()
    except Exception as e:
        print(f"Error loading logs: {e}. Plotting simulated curve instead.")
        plot_simulated_curve()
else:
    print("No ./logs/ directory. Retrain first, then rerun.")
    plot_simulated_curve()