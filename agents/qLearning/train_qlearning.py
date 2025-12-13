import gymnasium as gym
import ale_py
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent

def plot_metrics(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df['Total Reward'], alpha=0.3, color='gray', label='Original')
    plt.plot(df['Episode'], df['Total Reward'].rolling(window=10).mean(), color='blue', label='Avg (10 ep)')
    plt.title('Q-Learning: Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'reward.png'))
    plt.close()

    # 2. Steps
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df['Steps'], color='orange')
    plt.plot(df['Episode'], df['Steps'].rolling(window=10).mean(), color='red', label='Avg (10 ep)')
    plt.title('Q-Learning: Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'steps.png'))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def main():
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode=None, # Faster training without rendering
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb"
    )

    agent = QLearningAgent(env.action_space)
    
    # Load existing model if available
    model_path = "q_table.pkl"
    agent.load(model_path)

    episodes = 100000
    save_interval = 10
    
    metrics_data = [] # List to store metrics

    for episode in range(episodes):
        obs, info = env.reset()
        agent.reset_episode()
        total_reward = 0
        steps_episode = 0
        terminated = False
        truncated = False
        
        # Need to keep track of previous observation for update
        prev_obs = obs
        prev_lives = info.get("lives", 5)
        
        # For reward shaping
        prev_ball_x, prev_ball_y, prev_paddle_x = agent.get_positions(obs)
        while not (terminated or truncated):
            action = agent.get_action(prev_obs, training=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reward Shaping
            shaped_reward = reward
            
            ball_x, ball_y, paddle_x = agent.get_positions(obs)
            current_lives = info.get("lives", 5)
            
            # 1. Life lost
            if current_lives < prev_lives:
                shaped_reward -= 1.0
            
            if ball_x is not None and prev_ball_x is not None:
                # 2. Ball approaching paddle (dy > 0 means going down in image coords)
                if ball_y > prev_ball_y:
                    shaped_reward += 0.1
                # 3. Ball moving away
                elif ball_y < prev_ball_y:
                    shaped_reward -= 0.05
                
                # 4. Paddle moving towards ball
                # Only relevant if ball is coming down? Maybe always?
                # Let's say if ball is coming down, we want paddle to be close to ball_x
                if ball_y > prev_ball_y and paddle_x is not None and prev_paddle_x is not None:
                    dist = abs(ball_x - paddle_x)
                    prev_dist = abs(prev_ball_x - prev_paddle_x)
                    if dist < prev_dist:
                        shaped_reward += 0.05
            
            agent.update(prev_obs, action, shaped_reward, obs, done=(terminated or truncated))
            
            total_reward += reward # Keep tracking original reward for metrics
            steps_episode += 1
            prev_obs = obs
            prev_lives = current_lives
            prev_ball_x, prev_ball_y, prev_paddle_x = ball_x, ball_y, paddle_x

        agent.decay_epsilon()
        agent.decay_alpha()
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")

        # Store metrics
        metrics_data.append({
            "Episode": episode + 1,
            "Total Reward": total_reward,
            "Steps": steps_episode,
            "Epsilon": agent.epsilon,
            "Alpha": agent.alpha
        })

        if (episode + 1) % save_interval == 0:
            agent.save(model_path)
    
    # Save metrics to CSV
    metrics_file = os.path.join(os.path.dirname(__file__), "..", "..", "metrics", "QLearning_training.csv")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    df = pd.DataFrame(metrics_data)
    df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")
    
    # Plot
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "..", "metrics", "plots", "qlearning")
    plot_metrics(df, plot_dir)

    env.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
