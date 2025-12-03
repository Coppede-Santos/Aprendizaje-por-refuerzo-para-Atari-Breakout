import gymnasium as gym
import ale_py
import os
import sys
import numpy as np
import imageio
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.random_agent import RandomAgent
from agents.follow_ball_agent import FollowBallAgent
from agents.qLearning.qlearning_agent import QLearningAgent

def create_env(agent_name):
    if agent_name == "DQN":
        # SB3 DQN uses a vectorized environment
        # We use n_envs=1
        # Disable terminal_on_life_loss to allow full game recording
        # Disable clip_reward (though not strictly necessary for video)
        env = make_atari_env(
            "BreakoutNoFrameskip-v4", 
            n_envs=1, 
            seed=42, 
            wrapper_kwargs={"terminal_on_life_loss": False, "clip_reward": False}
        )
        env = VecFrameStack(env, n_stack=4)
        return env
    else:
        # Standard agents use the standard Gym environment
        # We need render_mode="rgb_array" to capture frames
        env = gym.make(
            "ALE/Breakout-v5",
            render_mode="rgb_array",
            full_action_space=False,
            repeat_action_probability=0.1,
            obs_type="rgb"
        )
        return env

def get_agent(agent_name, env):
    if agent_name == "Random":
        return RandomAgent(env.action_space)
    elif agent_name == "FollowBall":
        return FollowBallAgent(env.action_space)
    elif agent_name == "QLearning":
        agent = QLearningAgent(env.action_space)
        q_table_path = os.path.join(os.path.dirname(__file__), '..', 'agents', 'qLearning', 'q_table.pkl')
        if os.path.exists(q_table_path):
             agent.load(q_table_path)
        elif os.path.exists("q_table.pkl"):
            agent.load("q_table.pkl")
        else:
            print("Warning: q_table.pkl not found, using initialized Q-table.")
        return agent
    elif agent_name == "DQN":
        model_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'agents', 'dqn', 'entrenamiento_10', 'RESULTADOS_BREAKOUT_10M', 'entrenamiento_breakout_10M', 'modelos', 'best_model.zip'
        )
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"DQN model not found at {model_path}")
        
        model = DQN.load(model_path, env=env)
        return model
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

def record_episode(env, agent, agent_name):
    frames = []
    obs = env.reset()
    
    if agent_name != "DQN":
        obs, info = obs
    
    done = False
    terminated = False
    truncated = False
    
    # Handle lives for DQN force fire logic
    if agent_name == "DQN":
        lives = 5
    else:
        lives = info.get("lives", 5)
    
    prev_lives = lives
    force_fire = False

    while not done:
        # Capture frame
        if agent_name == "DQN":
            frame = env.render(mode="rgb_array")
        else:
            frame = env.render()
        
        frames.append(frame)

        # Determine action
        if agent_name == "DQN":
            action, _ = agent.predict(obs, deterministic=True)
            if force_fire:
                action = np.array([1])
                force_fire = False
        elif isinstance(agent, QLearningAgent):
            action = agent.get_action(obs, training=False)
        else:
            action = agent.get_action(obs)

        # Step
        if agent_name == "DQN":
            next_obs, reward, dones, infos = env.step(action)
            terminated = dones[0]
            info = infos[0]
            
            lives = info.get("lives", 0)
            was_truncated = info.get("TimeLimit.truncated", False)
            
            if terminated and lives > 0 and not was_truncated:
                done = False
            else:
                done = terminated
            
            if lives < prev_lives and lives > 0:
                force_fire = True
            
            prev_lives = lives
        else:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        obs = next_obs
        
        # Limit frames to avoid huge files (e.g., max 2000 frames)
        if len(frames) > 2000:
            print("Warning: Reached 2000 frames, stopping recording.")
            break

    return frames

def main():
    agents = ["Random", "FollowBall", "QLearning", "DQN"]
    output_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(output_dir, exist_ok=True)

    for agent_name in agents:
        print(f"\nRecording video for {agent_name}...")
        
        try:
            env = create_env(agent_name)
            agent = get_agent(agent_name, env)
            
            frames = record_episode(env, agent, agent_name)
            env.close()
            
            if frames:
                output_path = os.path.join(output_dir, f"{agent_name}.gif")
                print(f"Saving {len(frames)} frames to {output_path}...")
                # Save as GIF
                imageio.mimsave(output_path, frames, fps=30)
                print(f"Video saved: {output_path}")
            else:
                print(f"No frames captured for {agent_name}")
                
        except Exception as e:
            print(f"Error recording {agent_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
