import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import load_results
from scipy.stats import describe

def summarize_and_compare_training_data(log_dir='./logs/'):
    """
    Summarizes training data from Stable-Baselines3 logs and makes comparisons.
    - Loads monitor.csv from log_dir.
    - Computes stats: mean/final reward, std, convergence time.
    - Plots learning curve.
    - Compares phases (early/mid/late) and simulated vs. real (if no real data).
    - Handles missing logs with simulated data for demo.
    """
    
    # Check for logs and load data
    if os.path.exists(log_dir):
        try:
            results = load_results(log_dir)
            if not results.empty:
                df = results.reset_index()
                real_data = True
                print("Real data loaded successfully.")
            else:
                real_data = False
                print("No data in logs. Using simulated data for demo.")
        except Exception as e:
            real_data = False
            print(f"Error loading logs: {e}. Using simulated data for demo.")
    else:
        real_data = False
        print("No ./logs/ directory. Using simulated data for demo.")
    
    if not real_data:
        # Simulated data (typical for hexapod PPO on rough terrain)
        timesteps = np.linspace(0, 200000, 1000)
        rewards = -5 + 15 * (1 - np.exp(-timesteps / 50000)) + np.random.normal(0, 1.5, len(timesteps))
        df = pd.DataFrame({'t': timesteps, 'r': rewards})
        print("Generated simulated data for analysis.")
    
    # Summary Stats
    print("\n=== TRAINING SUMMARY ===")
    overall_mean = df['r'].mean()
    overall_std = df['r'].std()
    final_mean = df['r'].tail(200).mean() if len(df) > 200 else overall_mean
    final_std = df['r'].tail(200).std() if len(df) > 200 else overall_std
    convergence_step = df[df['r'] > 5]['t'].min() if any(df['r'] > 5) else np.nan
    
    print(f"Overall Mean Reward: {overall_mean:.2f} ± {overall_std:.2f}")
    print(f"Final 20% Mean Reward: {final_mean:.2f} ± {final_std:.2f}")
    print(f"Convergence Step (Reward >5): {convergence_step:.0f} timesteps")
    
    # Phase Comparison (Early: 0-20%, Mid: 20-60%, Late: 60-100%)
    n = len(df)
    early_df = df.iloc[:int(0.2*n)]
    mid_df = df.iloc[int(0.2*n):int(0.6*n)]
    late_df = df.iloc[int(0.6*n):]
    
    print("\n=== PHASE COMPARISON ===")
    print("Early Phase (0-20%): Mean {0:.2f}, Std {1:.2f} (Exploration/Falls)".format(early_df['r'].mean(), early_df['r'].std()))
    print("Mid Phase (20-60%): Mean {0:.2f}, Std {1:.2f} (Gait Adaptation)".format(mid_df['r'].mean(), mid_df['r'].std()))
    print("Late Phase (60-100%): Mean {0:.2f}, Std {1:.2f} (Convergence)".format(late_df['r'].mean(), late_df['r'].std()))
    
    # Improvement Metrics
    early_mean = early_df['r'].mean()
    late_improvement = ((late_df['r'].mean() - early_mean) / abs(early_mean)) * 100 if early_mean != 0 else np.inf
    print(f"Late vs. Early Improvement: +{late_improvement:.1f}%")
    
    # Plot Learning Curve
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df['t'], df['r'], label='Episode Reward', color='blue')
    plt.fill_between(df['t'], df['r'], alpha=0.3)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Hexapod Learning Curve')
    plt.legend()
    plt.grid(True)
    
    # Phase Markers
    plt.axvline(x=df['t'].iloc[int(0.2*len(df))], color='orange', linestyle='--', label='Early/Mid Phase')
    plt.axvline(x=df['t'].iloc[int(0.6*len(df))], color='red', linestyle='--', label='Mid/Late Phase')
    plt.legend()
    
    # Bar Chart for Phase Comparison
    plt.subplot(2, 1, 2)
    phases = ['Early', 'Mid', 'Late']
    means = [early_df['r'].mean(), mid_df['r'].mean(), late_df['r'].mean()]
    stds = [early_df['r'].std(), mid_df['r'].std(), late_df['r'].std()]
    plt.bar(phases, means, yerr=stds, capsize=5, color=['red', 'orange', 'green'], alpha=0.7)
    plt.ylabel('Mean Reward')
    plt.title('Reward Comparison by Phase')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('learning_curve_analysis.png')
    plt.show()
    print("Plot saved as learning_curve_analysis.png")
    
    # Detailed Comparison Table
    print("\n=== DETAILED COMPARISON TABLE ===")
    comparison = pd.DataFrame({
        'Phase': ['Early (0-20%)', 'Mid (20-60%)', 'Late (60-100%)'],
        'Mean Reward': [f"{early_df['r'].mean():.2f}", f"{mid_df['r'].mean():.2f}", f"{late_df['r'].mean():.2f}"],
        'Std Deviation': [f"{early_df['r'].std():.2f}", f"{mid_df['r'].std():.2f}", f"{late_df['r'].std():.2f}"],
        'Episodes (Est.)': [f"{len(early_df)}", f"{len(mid_df)}", f"{len(late_df)}"],
        'Improvement vs. Early': ['-', f"+{((mid_df['r'].mean() - early_df['r'].mean()) / abs(early_df['r'].mean()) * 100):.1f}%", f"+{((late_df['r'].mean() - early_df['r'].mean()) / abs(early_df['r'].mean()) * 100):.1f}%"]
    })
    print(comparison.to_string(index=False))

if __name__ == "__main__":
    summarize_and_compare_training_data()