"""
Train ONLY ES (RL already works)

Use this when iterating on ES improvements.
RL model is already saved in models/rl_pretrained.zip
"""

import numpy as np
from backend.env.car_env import CarEnv
from backend.rewards.control import ControlReward
from backend.agents.es import ESAgent

def train_es_only():
    print("=" * 60)
    print("TRAINING: ES ONLY (Geometric Control)")
    print("=" * 60)
    
    track_type = "oval"
    env = CarEnv(track_type=track_type, reward_type="progress")
    env.reward_fn = ControlReward(env.track)
    
    print("\nES Configuration:")
    print("- Observation: [LatErr, HeadErr, Speed, Curvature]")
    print("- Network: 4-64-2")
    
    es_agent = ESAgent(input_dim=4, output_dim=2, hidden_dim=64)
    curr_weights = es_agent.get_flat_weights()
    sigma = 0.1
    alpha = 0.01
    
    # 50 Generations is enough for ES to find the -1000 policy
    for gen in range(50):
        rewards = []
        noise = []
        
        for _ in range(40):
            eps = np.random.randn(len(curr_weights))
            noise.append(eps)
            
            es_agent.set_flat_weights(curr_weights + sigma * eps)
            es_agent.obs_mean = np.zeros(4)
            es_agent.obs_std = np.ones(4)
            es_agent.obs_count = 0
            es_agent.prev_action = np.zeros(2)
            
            env.reward_fn.reset()
            obs, _ = env.reset()
            tot_rew = 0
            
            for step in range(1000):
                action, _ = es_agent.predict(obs)
                obs, r, terminated, truncated, _ = env.step(action)
                tot_rew += r
                if terminated or truncated:
                    break
            
            rewards.append(tot_rew)
        
        rewards = np.array(rewards)
        if rewards.std() > 0:
            norm_rews = (rewards - rewards.mean()) / rewards.std()
            grad = np.zeros_like(curr_weights)
            for i, eps in enumerate(noise):
                grad += norm_rews[i] * eps
            curr_weights += alpha * grad / sigma
        
        if gen % 20 == 0:
            print(f"Gen {gen:3d}: Mean={rewards.mean():7.1f}, Max={rewards.max():7.1f}")
    
    es_agent.set_flat_weights(curr_weights)
    # Reset stats
    es_agent.obs_mean = np.zeros(4)
    es_agent.obs_std = np.ones(4)
    es_agent.obs_count = 0
    es_agent.prev_action = np.zeros(2)
    es_agent.save("models/es_pretrained.pkl")
    
    print("\n✓ ES training complete")

if __name__ == "__main__":
    train_es_only()
