"""
Quick trajectory generator for Segment 4 visualization
Generates sample trajectories for frontend display
"""

import numpy as np
import json
import sys
import os
from stable_baselines3 import PPO

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.env.track import Track
from backend.rewards.segment4 import ExploitableReward
from backend.agents.es import ESAgent


def generate_trajectory(env, agent, agent_type, seed=4001, max_steps=500):
    """Generate a sample trajectory for visualization"""
    np.random.seed(seed)
    
    if hasattr(agent, 'reset'):
        agent.reset()
    env.reward_fn.reset()
    obs, _ = env.reset(seed=seed)
    
    trajectory = []
    
    for step in range(max_steps):
        if agent_type == "RL":
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action, _ = agent.predict(obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Save trajectory point
        trajectory.append({
            "x": float(env.car.x),
            "y": float(env.car.y),
            "heading": float(env.car.heading),
            "velocity": float(env.car.velocity),
            "reward": float(reward)
        })
        
        if terminated or truncated:
            break
    
    return trajectory


if __name__ == "__main__":
    print("\nGenerating Segment 4 sample trajectories...")
    
    # Setup
    track = Track(track_type="figure8")
    env = CarEnv(track_type="figure8", reward_fn=ExploitableReward(track))
    
    # Load agents
    print("Loading agents...")
    rl_agent = PPO.load("models/seg4_exploit_rl", env=env)
    es_agent = ESAgent(input_dim=4, output_dim=2, hidden_dim=64)
    es_agent.load("models/seg4_exploit_es.pkl")
    print("✓ Agents loaded")
    
    # Generate trajectories
    print("Generating RL trajectory...")
    rl_traj = generate_trajectory(env, rl_agent, "RL", seed=4001)
    
    print("Generating ES trajectory...")
    es_traj = generate_trajectory(env, es_agent, "ES", seed=4001)
    
    # Save
    output = {
        "RL": {"sample_trajectory": rl_traj},
        "ES": {"sample_trajectory": es_traj}
    }
    
    output_path = "frontend/public/segment4_trajectories.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Trajectories saved to: {output_path}")
    print(f"  RL: {len(rl_traj)} steps")
    print(f"  ES: {len(es_traj)} steps")
