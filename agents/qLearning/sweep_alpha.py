import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
from agents.qlearning_agent import QLearningAgent
import os

def train_agent(alpha, episodes=100):
    """
    Trains a Q-Learning agent with a specific alpha and returns the list of total rewards per episode.
    """
    print(f"Starting training with alpha={alpha}")
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode=None,
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb"
    )

    agent = QLearningAgent(env.action_space, alpha=alpha)
    rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        prev_obs = obs
        prev_lives = info.get("lives", 5)
        prev_ball_x, prev_ball_y, prev_paddle_x = agent.get_positions(obs)

        while not (terminated or truncated):
            action = agent.get_action(prev_obs, training=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reward Shaping (Same as in train_qlearning.py)
            shaped_reward = reward
            ball_x, ball_y, paddle_x = agent.get_positions(obs)
            current_lives = info.get("lives", 5)
            
            if current_lives < prev_lives:
                shaped_reward -= 1.0
            
            if ball_x is not None and prev_ball_x is not None:
                if ball_y > prev_ball_y:
                    shaped_reward += 0.1
                elif ball_y < prev_ball_y:
                    shaped_reward -= 0.05
                
                if ball_y > prev_ball_y and paddle_x is not None and prev_paddle_x is not None:
                    dist = abs(ball_x - paddle_x)
                    prev_dist = abs(prev_ball_x - prev_paddle_x)
                    if dist < prev_dist:
                        shaped_reward += 0.05
            
            agent.update(prev_obs, action, shaped_reward, obs)
            
            total_reward += reward
            prev_obs = obs
            prev_lives = current_lives
            prev_ball_x, prev_ball_y, prev_paddle_x = ball_x, ball_y, paddle_x
        
        agent.decay_epsilon()
        rewards.append(total_reward)
        
        if (episode + 1) % 20 == 0:
            print(f"  [Alpha {alpha}] Episode {episode + 1}/{episodes}: Reward = {total_reward}")

    env.close()
    return rewards

def main():
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    episodes = 100
    all_rewards = {}

    for alpha in alphas:
        rewards = train_agent(alpha, episodes)
        all_rewards[alpha] = rewards

    # Plotting
    plt.figure(figsize=(10, 6))
    for alpha, rewards in all_rewards.items():
        # Smooth rewards for better visualization (moving average)
        window_size = 10
        if len(rewards) >= window_size:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, episodes), smoothed_rewards, label=f'Alpha = {alpha}')
        else:
            plt.plot(range(episodes), rewards, label=f'Alpha = {alpha}')

    plt.xlabel('Episodes')
    plt.ylabel('Total Reward (Smoothed)')
    plt.title('Q-Learning Performance by Learning Rate (Alpha)')
    plt.legend()
    plt.grid(True)
    
    output_file = 'agents/qLearning/alpha_sweep.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
