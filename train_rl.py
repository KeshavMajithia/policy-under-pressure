import sys
import os
import argparse

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.agents.rl import RLAgentFactory
from stable_baselines3.common.monitor import Monitor

def train_rl(timesteps=10000, save_path="models/ppo_car"):
    print(f"Starting RL Training (PPO) for {timesteps} steps...")
    
    # 1. Create Env
    env = CarEnv(reward_type="progress")
    env = Monitor(env) # For logging rewards
    
    # 2. Create Agent
    model = RLAgentFactory.create(env, verbose=1)
    
    # 3. Train
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # 4. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()
    
    train_rl(timesteps=args.steps)
