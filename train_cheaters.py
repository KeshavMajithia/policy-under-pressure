import gymnasium as gym
from stable_baselines3 import PPO
from backend.env.car_env import CarEnv
from backend.rewards.cheating import CheatingReward
from backend.agents.es import ESAgent
import numpy as np
import os
import time

def train_cheaters():
    print("--- Experiment 2: Reward Exploitation (The 'Cheating' Test) ---")
    
    # 1. Train RL Cheater on Figure-8
    print("Training RL Cheater on Figure-8...")
    # Track type must be Figure-8 to match the Master Race
    env = CarEnv(track_type="figure8", reward_type="cheating") 
    env.reward_fn = CheatingReward() 
    
    model = PPO("MlpPolicy", env, verbose=1)
    # Master-level training (200k) to ensure competence before cheating
    model.learn(total_timesteps=200000) 
    model.save("models/ppo_cheater_fig8")
    print("RL Cheater (Figure-8) Saved.")
    
    # 2. Train ES Cheater
    print("Training ES Cheater...")
    # Manual loop similar to train_es but short
    es_agent = ESAgent(5, 2)
    # ... (Reuse ES logic or simplified loop for brevity? Let's assume they behave similarly to standard training but with new reward)
    # Actually, lets mostly reuse the RL check for now as it's the most prone to reward hacking in literature.
    # But for completeness:
    
    # Quick ES Loop
    curr_weights = es_agent.get_flat_weights()
    sigma = 0.1
    alpha = 0.02
    for gen in range(20):
        # ... logic ...
        # Simplified: Just one step of updates
        rewards = []
        noise = []
        for _ in range(10):
            eps = np.random.randn(len(curr_weights))
            noise.append(eps)
            
            # Eval +
            es_agent.set_flat_weights(curr_weights + sigma*eps)
            obs, _ = env.reset()
            tot_rew = 0
            for _ in range(100):
                action, _ = es_agent.predict(obs)
                obs, r, t, tr, _ = env.step(action)
                tot_rew += r
                if t or tr: break
            rewards.append(tot_rew)
            
        # Update
        rewards = np.array(rewards)
        rews_std = rewards.std() + 1e-6
        norm_rews = (rewards - rewards.mean()) / rews_std
        
        grad = np.zeros_like(curr_weights)
        for i, eps in enumerate(noise):
            grad += norm_rews[i] * eps
        curr_weights += alpha * grad / sigma
        
        print(f"ES Gen {gen}: Mean Reward {rewards.mean():.1f}")
        
    es_agent.set_flat_weights(curr_weights)
    es_agent.save("models/es_cheater.pkl")
    print("ES Cheater Saved.")

if __name__ == "__main__":
    train_cheaters()
