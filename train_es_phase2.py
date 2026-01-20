
import numpy as np
import os
import sys
import pickle
sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.rewards.control import ControlReward
from backend.agents.es import ESAgent

def train_es_phase2():
    print("=" * 60)
    print("PHASE 2 PREP: ES RETRAINING (STRICT + POSITIVE)")
    print("=" * 60)
    
    # 1. Setup Environment
    track_type = "figure8"
    print(f"Track: {track_type}")
    
    env = CarEnv(track_type=track_type, reward_type="progress")
    # CRITICAL: Override with current STRICT ControlReward
    # (Scalar=5000.0, Lat=20.0, Head=10.0)
    env.reward_fn = ControlReward(env.track)
    print("Reward Function: ControlReward (Strict Penalties, Boosted Progress)")
    
    # 2. Setup ES Agent
    # Input: 4 (LatErr, HeadErr, Speed, Curvature)
    # Output: 2 (Steering, Throttle)
    es_agent = ESAgent(input_dim=4, output_dim=2, hidden_dim=64)
    
    # Hyperparameters
    population_size = 40
    generations = 500
    sigma = 0.05 # Lower noise for precision fine-tuning
    alpha = 0.05 # Increased LR to break stall 
    
    # Load Pretrained (Fine-Tuning Strategy)
    print("Loading Latest Strict Checkpoint for Precision Tuning...")
    try:
        es_agent.load("models/es_phase2_strict.pkl")
        print("✓ Loaded models/es_phase2_strict.pkl")
    except Exception as e:
        print(f"Warning: Could not load strict checkpoint: {e}")
        # Random init fallback if fails
    
    curr_weights = es_agent.get_flat_weights()
    best_reward_so_far = -np.inf
    
    print(f"Training ES for {generations} generations...")
    
    for gen in range(generations):
        rewards = []
        noise = []
        
        # Evaluate Population
        for _ in range(population_size):
            eps = np.random.randn(len(curr_weights))
            noise.append(eps)
            
            # Perturb weights
            test_weights = curr_weights + sigma * eps
            es_agent.set_flat_weights(test_weights)
            
            # Reset Env
            es_agent.reset() # Reset internal state filters if any
            env.reward_fn.reset()
            obs, _ = env.reset()
            tot_rew = 0
            
            # Run Episode
            for step in range(1000): # 1000 steps max
                # Predict
                action, _ = es_agent.predict(obs)
                obs, r, terminated, truncated, _ = env.step(action)
                tot_rew += r
                if terminated or truncated:
                    break
            
            rewards.append(tot_rew)
        
        rewards = np.array(rewards)
        mean_rew = rewards.mean()
        max_rew = rewards.max()
        
        # Update Weights (Canonical ES)
        if rewards.std() > 1e-6:
            # Rank transformation or Z-score
            norm_rews = (rewards - mean_rew) / rewards.std()
            grad = np.zeros_like(curr_weights)
            for i, eps in enumerate(noise):
                grad += norm_rews[i] * eps
            curr_weights += alpha * grad / (population_size * sigma)
        
        # Logging
        if max_rew > best_reward_so_far:
            best_reward_so_far = max_rew
            # Save intermediate best
            es_agent.set_flat_weights(curr_weights)
            es_agent.save("models/es_phase2_strict.pkl")
            
        if gen % 10 == 0:
            print(f"Gen {gen:3d}: Mean={mean_rew:7.1f}, Max={max_rew:7.1f} | Best={best_reward_so_far:7.1f}")

    # Final Save
    es_agent.set_flat_weights(curr_weights)
    es_agent.save("models/es_phase2_strict.pkl")
    print("\n✓ ES Phase 2 (Strict) saved to models/es_phase2_strict.pkl")

if __name__ == "__main__":
    train_es_phase2()
