import os
import matplotlib.pyplot as plt
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
        "train/loss": {"title": "Pérdida de Entrenamiento (Loss)", "ylabel": "Loss"},
        "train/learning_rate": {"title": "Tasa de Aprendizaje", "ylabel": "Learning Rate"},
        "rollout/exploration_rate": {"title": "Tasa de Exploración (Epsilon)", "ylabel": "Epsilon"},
        "time/fps": {"title": "Cuadros por Segundo (FPS)", "ylabel": "FPS"}
    }

    # Plot each scalar
    for tag in tags:
        if tag not in plot_config:
            continue
            
        data = ea.Scalars(tag)
        steps = [x.step for x in data]
        values = [x.value for x in data]

        config = plot_config[tag]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, linewidth=2)
        plt.title(config["title"], fontsize=14)
        plt.xlabel("Pasos de Entrenamiento", fontsize=12)
        plt.ylabel(config["ylabel"], fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Sanitize filename
        filename = tag.replace("/", "_") + ".png"
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
