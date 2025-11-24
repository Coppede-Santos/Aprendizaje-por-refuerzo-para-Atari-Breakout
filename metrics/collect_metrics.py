import argparse
import csv
import gymnasium as gym
import ale_py
import os
import sys
import time

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.random_agent import RandomAgent
from agents.follow_ball_agent import FollowBallAgent

def get_agent(agent_name, env):
    if agent_name == "RandomAgent":
        return RandomAgent(env.action_space)
    elif agent_name == "FollowBallAgent":
        return FollowBallAgent(env.action_space)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

def run_episode(env, agent, render=False):
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    brick_hits = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if reward > 0:
            brick_hits += 1
            
        if render:
            time.sleep(0.01)

    return {
        "total_reward": total_reward,
        "steps": steps,
        "brick_hits": brick_hits,
        "lives_left": info.get("lives", 0)
    }

def main():
    parser = argparse.ArgumentParser(description="Collect metrics for an agent.")
    parser.add_argument("--agent", type=str, default="RandomAgent", help="Name of the agent to run")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--output", type=str, default=None, help="Path to output CSV file")
    parser.add_argument("--render", action="store_true", help="Enable rendering")

    args = parser.parse_args()

    # Default output filename if not provided
    if args.output is None:
        args.output = f"metrics/{args.agent}.csv"

    render_mode = "human" if args.render else None
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode=render_mode,
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb"
    )

    try:
        agent = get_agent(args.agent, env)
    except ValueError as e:
        print(e)
        return

    results = []
    print(f"Running {args.agent} for {args.episodes} episodes...")

    # Determine starting episode ID if file exists
    start_episode_id = 1
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                start_episode_id = int(rows[-1]["episode_id"]) + 1

    for i in range(args.episodes):
        metrics = run_episode(env, agent, args.render)
        metrics["episode_id"] = start_episode_id + i
        results.append(metrics)
        print(f"Episode {metrics['episode_id']}: Reward={metrics['total_reward']}, Steps={metrics['steps']}")

    env.close()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Write to CSV (append if exists)
    file_exists = os.path.exists(args.output)
    mode = "a" if file_exists else "w"
    
    keys = ["episode_id", "total_reward", "steps", "brick_hits", "lives_left"]
    with open(args.output, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"Metrics saved to {args.output}")

if __name__ == "__main__":
    main()
