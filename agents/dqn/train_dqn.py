import numpy as np
import gymnasium as gym
import ale_py

from dqn_agent import ConvDQNAgent

CHECKPOINT_PATH = "conv_dqn_breakout_latest.pth"


def main():
    env = gym.make(
        "ALE/Breakout-v5",
        frameskip=4,
        render_mode=None,     # poné "human" si querés ver, pero es mucho más lento
    )

    agent = ConvDQNAgent(
        env.action_space,
        gamma=0.99,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999995,
        buffer_size=200_000,
        batch_size=32,
        target_update_freq=10_000,
        learn_start=50_000,
        num_frames=4,
    )

    # Continuar entrenamiento si ya existe un modelo
    agent.load(CHECKPOINT_PATH)

    num_episodes = 2000
    max_steps_per_ep = 10_000

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        # inicializar el frame stack del agente
        agent.reset_episode(obs)

        steps = 0
        while not done and steps < max_steps_per_ep:
            steps += 1

            # elegir acción con el stack interno de frames
            action = agent.get_action(training=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # reward clipping típico en Atari
            clipped_reward = np.sign(reward)

            # avisar un paso al agente (guarda transición + entrena)
            agent.step(next_obs, action, clipped_reward, done)

            obs = next_obs
            total_reward += reward

        print(
            f"Episodio {ep+1}/{num_episodes} | "
            f"Reward: {total_reward:.1f} | "
            f"epsilon: {agent.epsilon:.3f} | "
            f"steps totales: {agent.total_steps}"
        )

        # guardar siempre sobre el mismo archivo
        agent.save(CHECKPOINT_PATH)

    env.close()


if __name__ == "__main__":
    main()

