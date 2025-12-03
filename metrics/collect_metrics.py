import argparse
import csv
import gymnasium as gym
import ale_py
import os
import sys
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.random_agent import RandomAgent
from agents.follow_ball_agent import FollowBallAgent
from agents.qLearning.qlearning_agent import QLearningAgent

def create_env(agent_name, render_mode=None):
    if agent_name == "DQN":
        # SB3 DQN uses a vectorized environment with frame stacking
        # We use n_envs=1 for evaluation
        # Disable terminal_on_life_loss to allow full game evaluation
        # Disable clip_reward to see actual game score
        env = make_atari_env(
            "BreakoutNoFrameskip-v4", 
            n_envs=1, 
            seed=42, 
            wrapper_kwargs={"terminal_on_life_loss": False, "clip_reward": False}
        )
        env = VecFrameStack(env, n_stack=4)
        # Note: render_mode handling for VecEnv is different, usually done via env.render() call explicitly
        return env
    else:
        # Standard agents use the standard Gym environment
        env = gym.make(
            "ALE/Breakout-v5",
            render_mode=render_mode,
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
        # Assuming q_table.pkl is in the agents/qLearning directory or root
        # Let's try to find it relative to the script or project root
        q_table_path = os.path.join(os.path.dirname(__file__), '..', 'agents', 'qLearning', 'q_table.pkl')
        if os.path.exists(q_table_path):
             agent.load(q_table_path)
        elif os.path.exists("q_table.pkl"):
            agent.load("q_table.pkl")
        else:
            print("Warning: q_table.pkl not found, using initialized Q-table.")
        return agent
    elif agent_name == "DQN":
        # Load the specific SB3 model
        # Path from cargar_agente.py: agents/dqn/entrenamiento_10/RESULTADOS_BREAKOUT_10M/entrenamiento_breakout_10M/modelos/dqn_breakout_FINAL_10M.zip
        model_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'agents', 'dqn', 'entrenamiento_10', 'RESULTADOS_BREAKOUT_10M', 'entrenamiento_breakout_10M', 'modelos', 'dqn_breakout_FINAL_10M.zip'
        )
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"DQN model not found at {model_path}")
        
        model = DQN.load(model_path, env=env)
        return model
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

def run_episode(env, agent, agent_name, render=False):
    obs = env.reset()
    # VecEnv reset returns just obs, Gym reset returns (obs, info)
    if agent_name != "DQN":
        obs, info = obs
    
    total_reward = 0
    steps = 0
    brick_hits = 0
    terminated = False
    truncated = False
    done = False

    # For SB3 VecEnv, lives might be in info
    # We need to handle the initial state
    if agent_name == "DQN":
        # VecEnv reset returns just obs. We need to step or assume default lives.
        # Since we just reset, lives should be 5.
        lives = 5
    else:
        lives = info.get("lives", 5)
        
    prev_lives = lives
    force_fire = False

    while not done:
        # Determine action based on agent type
        if agent_name == "DQN":
            # SB3 predict returns (action, state)
            # deterministic=True is usually better for evaluation
            action, _ = agent.predict(obs, deterministic=True)
            
            # Force Fire if life lost (and not game over) to restart
            if force_fire:
                action = np.array([1]) # Fire action is usually 1 in Breakout
                force_fire = False
            
        elif isinstance(agent, QLearningAgent):
            action = agent.get_action(obs, training=False)
        else:  # RandomAgent, FollowBallAgent
            action = agent.get_action(obs)

        if agent_name == "DQN":
            # VecEnv step returns (obs, reward, done, info)
            # done is an array of booleans
            next_obs, reward, dones, infos = env.step(action)
            # Since n_envs=1, we take the first element
            terminated = dones[0]
            
            reward = reward[0]
            info = infos[0] # infos is a list of dicts
            
            # Check if game is truly over (all lives lost)
            # EpisodicLifeEnv wrapper causes done=True on life loss
            # We want to continue if lives > 0, unless it was truncated (time limit)
            lives = info.get("lives", 0)
            was_truncated = info.get("TimeLimit.truncated", False)
            
            if terminated and lives > 0 and not was_truncated:
                done = False
            else:
                done = terminated
            
            if lives < prev_lives and lives > 0:
                force_fire = True
            
            prev_lives = lives
            truncated = False # VecEnv handles auto-reset

            if render:
                env.render("human")
        else:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if render:
                time.sleep(0.01)

        obs = next_obs
        total_reward += reward
        steps += 1

        if reward > 0:
            brick_hits += 1

    # For SB3 VecEnv, lives might be in info
    lives = info.get("lives", 0)

    return {
        "total_reward": total_reward,
        "steps": steps,
        "brick_hits": brick_hits,
        "lives_left": lives
    }

def main():
    parser = argparse.ArgumentParser(description="Collect metrics for all agents.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run per agent")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--agents", nargs="+", default=["Random", "FollowBall", "QLearning", "DQN"], help="List of agents to run")

    args = parser.parse_args()

    for agent_name in args.agents:
        print(f"\n--- Processing Agent: {agent_name} ---")
        
        output_file = f"metrics/{agent_name}.csv"
        
        # Create environment
        render_mode = "human" if args.render and agent_name != "DQN" else None
        try:
            env = create_env(agent_name, render_mode)
        except Exception as e:
            print(f"Failed to create environment for {agent_name}: {e}")
            continue

        # Load agent
        try:
            agent = get_agent(agent_name, env)
        except Exception as e:
            print(f"Failed to load agent {agent_name}: {e}")
            env.close()
            continue

        results = []
        print(f"Running {agent_name} for {args.episodes} episodes...")

        # Determine starting episode ID if file exists
        start_episode_id = 1
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    start_episode_id = int(rows[-1]["episode_id"]) + 1

        for i in range(args.episodes):
            metrics = run_episode(env, agent, agent_name, args.render)
            metrics["episode_id"] = start_episode_id + i
            results.append(metrics)
            print(f"Episode {metrics['episode_id']}: Reward={metrics['total_reward']}, Steps={metrics['steps']}")

        env.close()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write to CSV (append if exists)
        file_exists = os.path.exists(output_file)
        mode = "a" if file_exists else "w"
        
        keys = ["episode_id", "total_reward", "steps", "brick_hits", "lives_left"]
        with open(output_file, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)

        print(f"Metrics saved to {output_file}")

if __name__ == "__main__":
    main()
