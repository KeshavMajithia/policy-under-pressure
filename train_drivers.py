"""
Simple training: Teach agents to DRIVE.
Goal: Move forward while staying on the centerline.
"""
from stable_baselines3 import PPO
from backend.env.car_env import CarEnv
from backend.rewards.driving import DrivingReward
from backend.agents.es import ESAgent
import numpy as np

def train_driving():
    print("=== TRAINING: BASIC DRIVING ===")
    print("Goal: Move + Steer = Follow the track")
    
    # Use Oval track (simpler than Figure-8)
    track_type = "oval"
    
    # 1. Train RL
    print(f"\n[1/2] Training RL (PPO) on {track_type}...")
    env = CarEnv(track_type=track_type, reward_type="progress")
    env.reward_fn = DrivingReward(env.track)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,  # More steps per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
    )
    # MASSIVE training for mastery
    model.learn(total_timesteps=500000)
    model.save(f"models/ppo_driver_{track_type}")
    print(f"✓ RL Driver saved")
    
    # 2. Train ES
    print(f"\n[2/2] Training ES on {track_type}...")
    es_agent = ESAgent(5, 2)
    
    curr_weights = es_agent.get_flat_weights()
    sigma = 0.1
    alpha = 0.01  # Smaller learning rate for stability
    
    # MASSIVE ES training (500 generations)
    for gen in range(500):
        rewards = []
        noise = []
        
        # Larger population
        for _ in range(20):
            eps = np.random.randn(len(curr_weights))
            noise.append(eps)
            
            es_agent.set_flat_weights(curr_weights + sigma * eps)
            env.reward_fn.reset()
            obs, _ = env.reset()
            tot_rew = 0
            
            # Longer episodes
            for step in range(500):
                action, _ = es_agent.predict(obs)
                obs, r, terminated, truncated, _ = env.step(action)
                tot_rew += r
                if terminated or truncated:
                    break
                    
            rewards.append(tot_rew)
        
        # Update
        rewards = np.array(rewards)
        rews_std = rewards.std() + 1e-6
        norm_rews = (rewards - rewards.mean()) / rews_std
        
        grad = np.zeros_like(curr_weights)
        for i, eps in enumerate(noise):
            grad += norm_rews[i] * eps
        curr_weights += alpha * grad / sigma
        
        if gen % 25 == 0:
            print(f"ES Gen {gen}: Mean={rewards.mean():.1f}, Max={rewards.max():.1f}")
    
    es_agent.set_flat_weights(curr_weights)
    es_agent.save(f"models/es_driver_{track_type}.pkl")
    print(f"✓ ES Driver saved")
    
    print("\n=== TRAINING COMPLETE ===")

if __name__ == "__main__":
    train_driving()
