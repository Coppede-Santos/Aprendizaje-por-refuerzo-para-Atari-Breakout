import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_and_plot(log_dir, output_dir):
    # Find the event file
    event_files = [f for f in os.listdir(log_dir) if 'tfevents' in f]
    if not event_files:
        print(f"No event file found in {log_dir}")
        return

    event_file = os.path.join(log_dir, event_files[0])
    print(f"Processing {event_file}...")

    # Load the event accumulator
    ea = EventAccumulator(event_file)
    ea.Reload()

    # Get available tags
    tags = ea.Tags()['scalars']
    print(f"Found tags: {tags}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot each scalar
    # Define titles and labels mapping
    plot_config = {
        "rollout/ep_rew_mean": {"title": "Recompensa Media por Episodio (Rollout)", "ylabel": "Recompensa Media"},
        "rollout/ep_len_mean": {"title": "Duración Media del Episodio (Rollout)", "ylabel": "Pasos"},
        "eval/mean_reward": {"title": "Recompensa Media (Evaluación)", "ylabel": "Recompensa Media"},
        "train/loss": [
            {"title": "Pérdida de Entrenamiento (Loss)", "ylabel": "Loss", "smoothing": 50},
            {"title": "Pérdida de Entrenamiento (Loss) - Zoom", "ylabel": "Loss", "smoothing": 50, "filename_suffix": "_zoom", "auto_zoom": True},
            {"title": "Pérdida de Entrenamiento (Loss) - Primeros 3M", "ylabel": "Loss", "smoothing": 50, "filename_suffix": "_first_3m", "max_step": 3000000},
            {"title": "Pérdida de Entrenamiento (Loss) - Últimos 3M", "ylabel": "Loss", "smoothing": 50, "filename_suffix": "_last_3m", "last_n_steps": 3000000}
        ],
        "train/learning_rate": {"title": "Tasa de Aprendizaje", "ylabel": "Learning Rate"},
        "rollout/exploration_rate": {"title": "Tasa de Exploración (Epsilon)", "ylabel": "Epsilon"},
        "time/fps": {"title": "Cuadros por Segundo (FPS)", "ylabel": "FPS"}
    }

    # Plot each scalar
    for tag in tags:
        if tag not in plot_config:
            continue
            
        configs = plot_config[tag]
        if isinstance(configs, dict):
            configs = [configs]

        data = ea.Scalars(tag)
        original_values = [x.value for x in data]
        steps = [x.step for x in data]
        
        for config in configs:
            current_values = original_values[:] 
            current_steps = steps[:]

            # Filter by step range if requested
            start_step = 0
            end_step = float('inf')

            if "max_step" in config:
                end_step = config["max_step"]
            
            if "last_n_steps" in config:
                max_actual_step = max(current_steps) if current_steps else 0
                start_step = max(0, max_actual_step - config["last_n_steps"])

            # Apply filtering
            if start_step > 0 or end_step != float('inf'):
                filtered_data = [(s, v) for s, v in zip(current_steps, current_values) if start_step <= s <= end_step]
                if not filtered_data:
                    print(f"No data found for config: {config.get('title')}")
                    continue
                current_steps, current_values = zip(*filtered_data)
                current_steps = list(current_steps)
                current_values = list(current_values)

            # Apply smoothing if requested
            if "smoothing" in config:
                window = config["smoothing"]
                current_values = pd.Series(current_values).rolling(window=window, min_periods=2).mean()
            
            plt.figure(figsize=(10, 6))
            plt.plot(current_steps, current_values, linewidth=2)
            plt.title(config["title"], fontsize=14)
            plt.xlabel("Pasos de Entrenamiento", fontsize=12)
            plt.ylabel(config["ylabel"], fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Apply auto-zoom if requested
            if config.get("auto_zoom", False):
                # Calculate 95th percentile to ignore spikes
                limit = pd.Series(current_values).quantile(0.95)
                plt.ylim(0, limit)
            
            # Sanitize filename
            suffix = config.get("filename_suffix", "")
            filename = tag.replace("/", "_") + suffix + ".png"
            output_path = os.path.join(output_dir, filename)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Saved {output_path}")

def main():
    # Path to the tensorboard logs
    # Based on findings: agents/dqn/entrenamiento_10/RESULTADOS_BREAKOUT_10M/entrenamiento_breakout_10M/tensorboard
    # We need to go into the specific run directory (e.g., DQN_1)
    base_log_dir = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'agents', 'dqn', 'entrenamiento_10', 'RESULTADOS_BREAKOUT_10M', 'entrenamiento_breakout_10M', 'tensorboard'
    )
    
    # Find the run directory (e.g., DQN_1)
    if os.path.exists(base_log_dir):
        subdirs = [d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d))]
        if subdirs:
            # Use the first one found, or iterate if needed. Assuming DQN_1 for now.
            log_dir = os.path.join(base_log_dir, subdirs[0])
        else:
            log_dir = base_log_dir
    else:
        print(f"Log directory not found: {base_log_dir}")
        return

    output_dir = os.path.join(os.path.dirname(__file__), "plots", "tensorboard")
    
    extract_and_plot(log_dir, output_dir)

if __name__ == "__main__":
    main()
