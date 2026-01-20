import sys
import os
import numpy as np
import argparse
import time

sys.path.append(os.getcwd())
from backend.env.car_env import CarEnv
from backend.agents.es import ESAgent

def evaluate(agent, env, seed=None):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    
    while not done:
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            done = True
            
    return total_reward

def train_es(n_users=50, sigma=0.1, alpha=0.01, generations=100, save_path="models/es_car.pkl"):
    print(f"Starting ES Training (Pop={n_users}, Sigma={sigma}, Alpha={alpha})...")
    
    env = CarEnv(reward_type="progress")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    
    # Center agent
    center_agent = ESAgent(input_dim, output_dim)
    curr_weights = center_agent.get_flat_weights()
    n_params = len(curr_weights)
    print(f"Agent has {n_params} parameters.")
    
    for gen in range(1, generations + 1):
        t0 = time.time()
        
        # 1. Create Perturbations
        # Use antithetic sampling for stability: epsilon and -epsilon
        noise_vectors = []
        half_pop = n_users // 2
        for _ in range(half_pop):
            eps = np.random.randn(n_params)
            noise_vectors.append(eps)
            noise_vectors.append(-eps)
            
        rewards = []
        
        # 2. Evaluate Population
        for eps in noise_vectors:
            # Create candidate
            candidate_weights = curr_weights + sigma * eps
            center_agent.set_flat_weights(candidate_weights)
            
            # Eval
            r = evaluate(center_agent, env)
            rewards.append(r)
            
        rewards = np.array(rewards)
        
        # 3. Update
        # Normalize rewards (rank shaping or standardization) is crucial for ES stability
        # Standardization:
        rew_std = rewards.std()
        if rew_std < 1e-6: rew_std = 1.0
        normalized_rewards = (rewards - rewards.mean()) / rew_std
        
        # Gradient estimate
        # w_new = w + alpha * (1 / (N * sigma)) * sum(F_i * eps_i)
        grad = np.zeros_like(curr_weights)
        for i, eps in enumerate(noise_vectors):
            grad += normalized_rewards[i] * eps
            
        grad /= len(noise_vectors) # Expected value
        
        curr_weights += alpha * grad / sigma
        
        # Stats
        mean_rew = np.mean(rewards)
        max_rew = np.max(rewards)
        
        t1 = time.time()
        print(f"Gen {gen}/{generations}: Mean={mean_rew:.2f}, Max={max_rew:.2f}, Time={t1-t0:.2f}s")
        
        # Save center agent
        center_agent.set_flat_weights(curr_weights)
        if gen % 10 == 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            center_agent.save(save_path)
            
    print("Training Complete.")
    center_agent.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=50)
    args = parser.parse_args()
    
    train_es(generations=args.gens)
