"""
Segment 4 Training Script: Misalignment Agents

Trains RL and ES agents with MisalignedReward to test whether
agents learn "correct" driving or literal optimization.

Protocol: segment4_protocol.md (Experiment 3)
Seeds: RL=5030, ES=5031
"""

import numpy as np
import os
import sys
from stable_baselines3 import PPO

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.env.track import Track
from backend.rewards.segment4 import MisalignedReward
from backend.agents.es import ESAgent


def train_misaligned_rl(timesteps=500_000, seed=5030):
    """Train RL agent with MisalignedReward"""
    print("="*60)
    print("SEGMENT 4 - EXPERIMENT 3: ALIGNMENT TEST")
    print("Training RL (PPO) with MisalignedReward")
    print("="*60)
    
    # Environment setup
    track = Track(track_type="figure8")
    env = CarEnv(track_type="figure8", reward_fn=MisalignedReward(track))
    
    print(f"Track: figure8")
    print(f"Reward: MisalignedReward (speed-focused)")
    print(f"Timesteps: {timesteps}")
    print(f"Seed: {seed}")
    
    # Set seed
    np.random.seed(seed)
    env.reset(seed=seed)
    
    # PPO Agent
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
        seed=seed
    )
    
    # Train
    print(f"\nTraining for {timesteps} steps...")
    model.learn(total_timesteps=timesteps)
    
    # Save
    save_path = "models/seg4_misaligned_rl"
    model.save(save_path)
    print(f"\n✓ RL Misaligned agent saved to {save_path}.zip")
    
    return model


def train_misaligned_es(generations=500, seed=5031):
    """Train ES agent with MisalignedReward"""
    print("\n" + "="*60)
    print("Training ES (Evolution) with MisalignedReward")
    print("="*60)
    
    # Environment setup
    track = Track(track_type="figure8")
    env = CarEnv(track_type="figure8", reward_fn=MisalignedReward(track))
    
    print(f"Track: figure8")
    print(f"Reward: MisalignedReward (speed-focused)")
    print(f"Generations: {generations}")
    print(f"Seed: {seed}")
    
    # Set seed
    np.random.seed(seed)
    
    # ES Agent
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    agent = ESAgent(input_dim, output_dim, hidden_dim=64)
    
    # ES Hyperparameters
    population_size = 100
    sigma = 0.1
    alpha = 0.01
    
    curr_weights = agent.get_flat_weights()
    best_reward = -np.inf
    
    print(f"\nTraining for {generations} generations...")
    print(f"Population Size: {population_size}")
    
    for gen in range(generations):
        rewards = []
        noise = []
        
        # Evaluate population
        for _ in range(population_size):
            eps = np.random.randn(len(curr_weights))
            noise.append(eps)
            
            # Perturb weights
            test_weights = curr_weights + sigma * eps
            agent.set_flat_weights(test_weights)
            
            # Reset
            agent.reset()
            env.reward_fn.reset()
            obs, _ = env.reset()
            tot_reward = 0
            
            # Run episode
            for step in range(1000):
                action, _ = agent.predict(obs)
                obs, r, terminated, truncated, _ = env.step(action)
                tot_reward += r
                if terminated or truncated:
                    break
            
            rewards.append(tot_reward)
        
        rewards = np.array(rewards)
        mean_reward = rewards.mean()
        max_reward = rewards.max()
        
        # Update weights
        if rewards.std() > 1e-6:
            norm_rewards = (rewards - mean_reward) / rewards.std()
            grad = np.zeros_like(curr_weights)
            for i, eps in enumerate(noise):
                grad += norm_rewards[i] * eps
            curr_weights += alpha * grad / (population_size * sigma)
        
        # Save best
        if max_reward > best_reward:
            best_reward = max_reward
            agent.set_flat_weights(curr_weights)
            agent.save("models/seg4_misaligned_es.pkl")
        
        # Log progress
        if gen % 50 == 0:
            print(f"Gen {gen:3d} | Mean: {mean_reward:8.2f} | Max: {max_reward:8.2f} | Best: {best_reward:8.2f}")
    
    # Final save
    agent.set_flat_weights(curr_weights)
    save_path = "models/seg4_misaligned_es.pkl"
    agent.save(save_path)
    print(f"\n✓ ES Misaligned agent saved to {save_path}")
    
    return agent


if __name__ == "__main__":
    print("\n🚀 Starting Segment 4 - Experiment 3 Training\n")
    
    # Check if models already exist
    rl_model_exists = os.path.exists("models/seg4_misaligned_rl.zip")
    es_model_exists = os.path.exists("models/seg4_misaligned_es.pkl")
    
    # Train RL agent (skip if already exists)
    if rl_model_exists:
        print("✓ RL Misaligned agent already exists - skipping training")
        print(f"  Found: models/seg4_misaligned_rl.zip\n")
    else:
        rl_agent = train_misaligned_rl(timesteps=500_000, seed=5030)
    
    # Train ES agent (skip if already exists)
    if es_model_exists:
        print("✓ ES Misaligned agent already exists - skipping training")
        print(f"  Found: models/seg4_misaligned_es.pkl\n")
    else:
        es_agent = train_misaligned_es(generations=500, seed=5031)
    
    print("\n" + "="*60)
    print("✅ MISALIGNMENT AGENTS TRAINING COMPLETE")
    print("="*60)
    print("\nModels saved:")
    print("  - models/seg4_misaligned_rl.zip")
    print("  - models/seg4_misaligned_es.pkl")
    print("\nNext: Run evaluation with run_segment4_alignment.py")
