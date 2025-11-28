import gymnasium as gym
import ale_py
import time
import os
from qlearning_agent import QLearningAgent

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
    model_path = "agents/qLearning/q_table.pkl"
    #agent.load(model_path)

    episodes = 100
    save_interval = 10

    for episode in range(episodes):
        obs, info = env.reset()
        agent.reset_episode()
        total_reward = 0
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
            prev_obs = obs
            prev_lives = current_lives
            prev_ball_x, prev_ball_y, prev_paddle_x = ball_x, ball_y, paddle_x

        agent.decay_epsilon()
        agent.decay_alpha()
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")

        if (episode + 1) % save_interval == 0:
            agent.save(model_path)

    env.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
