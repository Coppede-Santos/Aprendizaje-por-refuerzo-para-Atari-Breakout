import gymnasium as gym
import ale_py
import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt  # <-- para graficar

# Aseguramos poder importar el agente desde la raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.qLearning.qlearning_agent import QLearningAgent


def run_training_episode(env, epsilon, epsilon_decay, epsilon_min, alpha=0.5, gamma=0.99, max_steps=10000):
    """
    Entrena un QLearningAgent durante un episodio con los hiperparámetros dados
    y devuelve la recompensa total obtenida.
    """
    agent = QLearningAgent(
        env.action_space,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )

    obs, info = env.reset()
    agent.reset_episode()
    terminated = False
    truncated = False
    total_reward = 0.0

    prev_obs = obs
    prev_lives = info.get("lives", 5)
    prev_ball_x, prev_ball_y, prev_paddle_x = agent.get_positions(obs)

    steps = 0

    while not (terminated or truncated):
        steps += 1
        if steps > max_steps:
            # Por seguridad, cortamos episodios demasiado largos
            break

        action = agent.get_action(prev_obs, training=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Reward shaping (igual que en tu script de entrenamiento) ---
        shaped_reward = reward

        ball_x, ball_y, paddle_x = agent.get_positions(obs)
        current_lives = info.get("lives", 5)

        # 1) Penalizar pérdida de vida
        if current_lives < prev_lives:
            shaped_reward -= 1.0

        if ball_x is not None and prev_ball_x is not None:
            # 2) Pelota acercándose a la pala (hacia abajo en coords de imagen)
            if ball_y > prev_ball_y:
                shaped_reward += 0.1
            # 3) Pelota alejándose
            elif ball_y < prev_ball_y:
                shaped_reward -= 0.05

            # 4) Pala moviéndose hacia la pelota cuando esta baja
            if ball_y > prev_ball_y and paddle_x is not None and prev_paddle_x is not None:
                dist = abs(ball_x - paddle_x)
                prev_dist = abs(prev_ball_x - prev_paddle_x)
                if dist < prev_dist:
                    shaped_reward += 0.05

        # Actualización Q-learning con la recompensa moldeada
        agent.update(prev_obs, action, shaped_reward, obs, done=(terminated or truncated))

        total_reward += reward  # Métrica: recompensa original del entorno

        # Actualizar variables previas
        prev_obs = obs
        prev_lives = current_lives
        prev_ball_x, prev_ball_y, prev_paddle_x = ball_x, ball_y, paddle_x

    # Decaimos tasas al final del episodio (aunque aquí no se reutiliza el agente)
    agent.decay_epsilon()
    agent.decay_alpha()

    return total_reward


def sweep_epsilon(
    epsilon_values,
    episodes_per_value=3,
    epsilon_decay=0.9995,
    epsilon_min=0.01,
    alpha=0.5,
    gamma=0.99,
):
    """
    Barre sobre una lista de valores iniciales de epsilon.
    Para cada epsilon:
      - ejecuta varios episodios de entrenamiento cortos
      - devuelve la recompensa media y std.
    """
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode=None,           # Sin render para ir más rápido
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb",
    )

    results = []

    for eps in epsilon_values:
        print(f"\n===== Probando epsilon inicial = {eps} =====")
        rewards = []
        for ep in range(episodes_per_value):
            total_reward = run_training_episode(
                env,
                epsilon=eps,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                alpha=alpha,
                gamma=gamma,
            )
            rewards.append(total_reward)
            print(f"  Episodio {ep + 1}/{episodes_per_value}: recompensa total = {total_reward}")

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        results.append((eps, mean_reward, std_reward))
        print(f"--> epsilon = {eps}: recompensa media = {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return results


def plot_epsilon_sweep(results, output_path=None):
    """
    Grafica recompensa media vs epsilon con barras de error (std).
    """
    epsilons = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]

    plt.figure(figsize=(8, 5))
    plt.errorbar(epsilons, means, yerr=stds, fmt='-o', capsize=5)
    plt.xlabel("Epsilon inicial")
    plt.ylabel("Recompensa media por episodio")
    plt.title("Barrido de epsilon para QLearningAgent en Breakout")
    plt.grid(True)

    if output_path is None:
        # Por defecto, guardamos en el mismo directorio que este script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "epsilon_sweep.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado en: {output_path}")


def main():
    # Lista de valores de epsilon a probar (ajusta según necesites)
    epsilon_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    episodes_per_value = 100  # Episodios por cada epsilon

    results = sweep_epsilon(
        epsilon_values=epsilon_values,
        episodes_per_value=episodes_per_value,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        alpha=1,
        gamma=0.99,
    )

    print("\n===== RESUMEN BARRIDO EPSILON =====")
    for eps, mean_r, std_r in results:
        print(f"epsilon = {eps:>4}: media = {mean_r:7.2f}, std = {std_r:7.2f}")

    # Graficar y guardar en epsilon_sweep.png
    plot_epsilon_sweep(results)


if __name__ == "__main__":
    main()