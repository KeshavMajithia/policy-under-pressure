import gymnasium as gym
from stable_baselines3 import PPO
from backend.env.car_env import CarEnv
from backend.agents.es import ESAgent
import numpy as np
import os
import time

def train_generalization():
    print("--- Experiment 3: Generalization (Figure 8 Track) ---")
    
    # 1. Train RL on Figure 8
    print("\nTraining RL on Figure 8...")
    env = CarEnv(track_type="figure8", reward_type="progress") 
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=300000) # 10x Training for mastery
    model.save("models/ppo_figure8")
    print("RL Figure 8 Agent Saved.")
    
    # 2. Train ES on Figure 8
    print("\nTraining ES on Figure 8...")
    es_agent = ESAgent(5, 2)
    
    curr_weights = es_agent.get_flat_weights()
    sigma = 0.1
    alpha = 0.02
    
    # Rigorous ES Loop (100 gens)
    for gen in range(100):
        rewards = []
        noise = []
        for _ in range(10): # Population size
            eps = np.random.randn(len(curr_weights))
            noise.append(eps)
            
            es_agent.set_flat_weights(curr_weights + sigma*eps)
            obs, _ = env.reset()
            tot_rew = 0
            # Shorter episodes for speed
            for _ in range(200):
                action, _ = es_agent.predict(obs)
                obs, r, t, tr, _ = env.step(action)
                tot_rew += r
                if t or tr: break
            rewards.append(tot_rew)
            
        rewards = np.array(rewards)
        if rewards.std() > 1e-6:
            norm_rews = (rewards - rewards.mean()) / rewards.std()
        else:
            norm_rews = np.zeros_like(rewards)
        
        grad = np.zeros_like(curr_weights)
        for i, eps in enumerate(noise):
            grad += norm_rews[i] * eps
        curr_weights += alpha * grad / sigma
        
        print(f"ES Gen {gen}: Mean Reward {rewards.mean():.1f}")
        
    es_agent.set_flat_weights(curr_weights)
    es_agent.save("models/es_figure8.pkl")
    print("ES Figure 8 Agent Saved.")

if __name__ == "__main__":
    train_generalization()
