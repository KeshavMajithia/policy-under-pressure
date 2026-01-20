
import numpy as np
import os
import sys
from stable_baselines3 import PPO
sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.rewards.control import ControlReward

def train_rl_fixed():
    print("=" * 60)
    print("PHASE 2 PREP: RL RETRAINING (FIXED REWARD)")
    print("=" * 60)
    
    # 1. Setup Environment with FIGURE-8 (Harder than oval, good for learning)
    # Using 'random' might be even better for robustness, but Figure-8 is the standard benchmark
    track_type = "figure8" 
    print(f"Track: {track_type}")
    
    # CRITICAL: Reward Type "progress" triggers the ControlReward assignment below?
    # No, CarEnv's reward_type is just for initial setup, we override it manually.
    env = CarEnv(track_type=track_type, reward_type="progress")
    
    # OVERRIDE with Fixed ControlReward (Progress * 500)
    env.reward_fn = ControlReward(env.track)
    print("Reward Function: ControlReward (Scaled Progress)")
    
    # 2. Setup PPO Agent
    # Same config as Phase 1, just fixing the reward signal
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
    )
    
    # 3. Train
    # 500k steps should be enough for "perfect" driving given ES did it in 5 mins
    steps = 500_000
    print(f"Training for {steps} steps...")
    model.learn(total_timesteps=steps)
    
    # 4. Save
    save_path = "models/rl_phase2_fixed"
    model.save(save_path)
    print(f"✓ RL (Fixed) saved to {save_path}")

if __name__ == "__main__":
    train_rl_fixed()
