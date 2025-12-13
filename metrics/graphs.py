import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def load_data(metrics_dir="metrics"):
    all_files = glob.glob(os.path.join(metrics_dir, "*.csv"))
    df_list = []
    
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # Extract agent name from filename (e.g., "metrics/Random.csv" -> "Random")
            agent_name = os.path.splitext(os.path.basename(filename))[0]
            df['Agent'] = agent_name
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

def plot_metrics(df, output_dir="metrics/plots"):
    if df.empty:
        print("No data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    metrics_to_plot = ["total_reward", "steps", "brick_hits"]
    titles = ["Total Reward", "Steps per Episode", "Brick Hits"]
    
    metrics_to_plot = ["total_reward", "steps", "brick_hits"]
    titles = ["Total Reward", "Steps per Episode", "Brick Hits"]
    
    # Define custom colors for each agent
    palette = {
        "Random": "#95a5a6",      # Gray
        "FollowBall": "#f1c40f",  # Yellow/Gold
        "QLearning": "#2ecc71",   # Green
        "DQN": "#e74c3c"          # Red
    }
    
    for i, metric in enumerate(metrics_to_plot):
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Agent", y=metric, data=df, palette=palette, showfliers=False)
        plt.title(titles[i])
        plt.xlabel("Agent")
        plt.ylabel(metric)
        
        # Add strip plot to show individual points
        sns.stripplot(x="Agent", y=metric, data=df, color=".3", alpha=0.4, size=3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{metric}_boxplot.png")
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        plt.close()

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming CSVs are in the same directory as the script
    metrics_dir = script_dir
    
    print(f"Loading data from {metrics_dir}...")
    df = load_data(metrics_dir)
    
    if not df.empty:
        print(f"Loaded {len(df)} records.")
        print("Agents found:", df['Agent'].unique())
        
        # Output directory for plots
        output_dir = os.path.join(script_dir, "plots")
        
        # Print statistics
        print("\n" + "="*40)
        print("METRIC STATISTICS (Mean & Std Dev)")
        print("="*40)
        metrics_cols = ["total_reward", "steps", "brick_hits"]
        # Filter cols that exist
        metrics_cols = [c for c in metrics_cols if c in df.columns]
        if metrics_cols:
            stats = df.groupby("Agent")[metrics_cols].agg(['mean', 'std'])
            print(stats)
        else:
            print("Metrics columns not found.")
        print("="*40 + "\n")
        
        plot_metrics(df, output_dir)
    else:
        print("No data found.")

if __name__ == "__main__":
    main()
