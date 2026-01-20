"""
PHASE 1: MOTOR SKILL LEARNING (ES-Optimized)

Key improvements for ES:
- Soft crashes (slow down, don't terminate)
- Strong survival signal
- Action smoothing
- Observation normalization  
- Larger network capacity
- Throttle floor

This creates a SMOOTH fitness landscape for ES.
"""

import numpy as np
from stable_baselines3 import PPO
from backend.env.car_env import CarEnv
from backend.rewards.control import ControlReward
from backend.agents.es import ESAgent

def train_phase1_control():
    print("=" * 60)
    print("PHASE 1: GEOMETRIC CONTROL LEARNING")
    print("=" * 60)
    print("\nGoal: Minimize Lateral and Heading Errors")
    print("Observation: [LatError, HeadError, Speed, Curvature]")
    print("\n")
    
    track_type = "oval"
    
    # ========================================
    # Step 1: Train RL (PPO)
    # ========================================
    print("[1/2] Training RL (PPO)...")
    env = CarEnv(track_type=track_type, reward_type="progress")
    env.reward_fn = ControlReward(env.track)
    
    # Observations are now normalized geometric errors, so PPO should learn fast
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
    
    print("Training for 300,000 steps...")
    model.learn(total_timesteps=300_000)
    model.save("models/rl_pretrained")
    print("✓ RL pre-trained\n")
    
    # ========================================
    # Step 2: Train ES
    # ========================================
    print("[2/2] Training ES (Control-Optimized)...")
    
    # INPUT DIM IS NOW 4 (LatErr, HeadErr, Speed, Curvature)
    es_agent = ESAgent(input_dim=4, output_dim=2, hidden_dim=64) # 64 is enough for control
    
    curr_weights = es_agent.get_flat_weights()
    sigma = 0.1 
    alpha = 0.01 
    
    for gen in range(500):
        rewards = []
        noise = []
        
        # Population 40
        for _ in range(40):
            eps = np.random.randn(len(curr_weights))
            noise.append(eps)
            
            es_agent.set_flat_weights(curr_weights + sigma * eps)
            
            # Reset ES internals
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
    # Reset stats for save
    es_agent.obs_mean = np.zeros(4)
    es_agent.obs_std = np.ones(4)
    es_agent.obs_count = 0
    es_agent.prev_action = np.zeros(2)
    es_agent.save("models/es_pretrained.pkl")
    print("\n✓ ES pre-trained")

if __name__ == "__main__":
    train_phase1_control()
