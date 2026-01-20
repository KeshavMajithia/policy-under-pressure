"""
Train RL and ES agents to MASTER DRIVING on Figure-8 track.
Focus: Competency first, competition later.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from backend.env.car_env import CarEnv
from backend.rewards.mastery import MasteryReward
from backend.agents.es import ESAgent
import numpy as np

def train_mastery():
    print("=== PHASE 1: DRIVING MASTERY TRAINING ===")
    print("Goal: Teach agents to follow the track and complete laps")
    
    # 1. Train RL with Mastery Reward
    print("\n[1/2] Training RL (PPO) with Mastery Reward on Figure-8...")
    env = CarEnv(track_type="figure8", reward_type="progress")
    env.reward_fn = MasteryReward()  # Override with mastery reward
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    # Long training to ensure mastery (500k steps)
    model.learn(total_timesteps=500000)
    model.save("models/ppo_master_fig8")
    print("✓ RL Master saved to models/ppo_master_fig8.zip")
    
    # 2. Train ES with Mastery Reward
    print("\n[2/2] Training ES with Mastery Reward on Figure-8...")
    es_agent = ESAgent(5, 2)
    
    curr_weights = es_agent.get_flat_weights()
    sigma = 0.1
    alpha = 0.01  # Slower learning for stability
    
    # Extended ES training (200 generations)
    for gen in range(200):
        rewards = []
        noise = []
        
        # Larger population for better exploration
        for _ in range(20):
            eps = np.random.randn(len(curr_weights))
            noise.append(eps)
            
            # Eval perturbed weights
            es_agent.set_flat_weights(curr_weights + sigma * eps)
            env.reward_fn.reset()  # Reset reward state
            obs, _ = env.reset()
            tot_rew = 0
            
            # Longer episodes to encourage lap completion
            for step in range(500):
                action, _ = es_agent.predict(obs)
                obs, r, terminated, truncated, _ = env.step(action)
                tot_rew += r
                if terminated or truncated:
                    break
                    
            rewards.append(tot_rew)
        
        # Update weights
        rewards = np.array(rewards)
        rews_std = rewards.std() + 1e-6
        norm_rews = (rewards - rewards.mean()) / rews_std
        
        grad = np.zeros_like(curr_weights)
        for i, eps in enumerate(noise):
            grad += norm_rews[i] * eps
        curr_weights += alpha * grad / sigma
        
        if gen % 10 == 0:
            print(f"ES Gen {gen}: Mean Reward {rewards.mean():.2f}, Max {rewards.max():.2f}")
    
    es_agent.set_flat_weights(curr_weights)
    es_agent.save("models/es_master_fig8.pkl")
    print("✓ ES Master saved to models/es_master_fig8.pkl")
    
    print("\n=== MASTERY TRAINING COMPLETE ===")
    print("Next: Run generate_log.py to create race replays")

if __name__ == "__main__":
    train_mastery()
