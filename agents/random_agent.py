import gymnasium as gym
import ale_py
import time
import numpy as np

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation):
        return self.action_space.sample()

def main():
    # Create environment
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="human",
        obs_type="rgb"
    )

    agent = RandomAgent(env.action_space)

    print("Action Space:", env.action_space)
    print("Action Meanings:", env.unwrapped.get_action_meanings())

    num_episodes = 5
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(0.01) # Speed up slightly compared to human play

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {steps}")

    env.close()

if __name__ == "__main__":
    main()
