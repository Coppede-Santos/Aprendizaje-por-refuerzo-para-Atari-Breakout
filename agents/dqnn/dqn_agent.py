from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import os

class StableBaselinesDQNAgent:
    def __init__(self, env):
        self.env = DummyVecEnv([lambda: env])
        self.model = DQN("CnnPolicy", self.env, verbose=0)

    def get_action(self, obs, training=True):
        # The observation is already handled by the VecEnv wrapper
        action, _states = self.model.predict(obs, deterministic=not training)
        return action

    def train(self, total_timesteps=25000):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path + ".zip"):
            self.model = DQN.load(path, env=self.env)
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}, starting from scratch.")

    def get_model(self):
        return self.model
