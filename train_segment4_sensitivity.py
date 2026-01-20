"""
Segment 4 Training Script: Sensitivity Agents

Trains 10 agents (5 RL + 5 ES) with varying reward coefficients
to test sensitivity to reward shaping.

Protocol: segment4_protocol.md (Experiment 2)
Seeds: RL=5010-5014, ES=5020-5024
"""

import numpy as np
import os
import sys
from stable_baselines3 import PPO

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.env.track import Track
from backend.rewards.segment4 import SensitivityReward, SENSITIVITY_CONFIGS
from backend.agents.es import ESAgent


def train_sensitivity_rl(config_name, config, seed):
    """Train single RL agent with specific reward configuration"""
    print(f"\n{'='*60}")
    print(f"Training RL - Config: {config_name}")
    print(f"  Lat Penalty: {config['lat_penalty']}")
    print(f"  Heading Penalty: {config['heading_penalty']}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}")
    
    # Environment setup
    track = Track(track_type="figure8")
    reward_fn = SensitivityReward(
        track, 
        lat_penalty=config['lat_penalty'],
        heading_penalty=config['heading_penalty']
    )
    env = CarEnv(track_type="figure8", reward_fn=reward_fn)
    
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
    
    # Train (300k steps - less than exploit since we're doing 5 agents)
    model.learn(total_timesteps=300_000)
    
    # Save
    save_path = f"models/seg4_sensitivity_rl_{config_name}"
    model.save(save_path)
    print(f"✓ Saved to {save_path}.zip")
    
    return model


def train_sensitivity_es(config_name, config, seed):
    """Train single ES agent with specific reward configuration"""
    print(f"\n{'='*60}")
    print(f"Training ES - Config: {config_name}")
    print(f"  Lat Penalty: {config['lat_penalty']}")
    print(f"  Heading Penalty: {config['heading_penalty']}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}")
    
    # Environment setup
    track = Track(track_type="figure8")
    reward_fn = SensitivityReward(
        track,
        lat_penalty=config['lat_penalty'],
        heading_penalty=config['heading_penalty']
    )
    env = CarEnv(track_type="figure8", reward_fn=reward_fn)
    
    # Set seed
    np.random.seed(seed)
    
    # ES Agent
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    agent = ESAgent(input_dim, output_dim, hidden_dim=64)
    
    # ES Hyperparameters (reduced generations since doing 5 configs)
    generations = 300
    population_size = 100
    sigma = 0.1
    alpha = 0.01
    
    curr_weights = agent.get_flat_weights()
    best_reward = -np.inf
    
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
            save_path = f"models/seg4_sensitivity_es_{config_name}.pkl"
            agent.save(save_path)
        
        # Log every 50 generations
        if gen % 50 == 0:
            print(f"  Gen {gen:3d} | Mean: {mean_reward:8.2f} | Max: {max_reward:8.2f}")
    
    # Final save
    agent.set_flat_weights(curr_weights)
    save_path = f"models/seg4_sensitivity_es_{config_name}.pkl"
    agent.save(save_path)
    print(f"✓ Saved to {save_path}")
    
    return agent


if __name__ == "__main__":
    print("\n🚀 Starting Segment 4 - Experiment 2: Sensitivity Training\n")
    print(f"Training {len(SENSITIVITY_CONFIGS)} configs × 2 algorithms = {len(SENSITIVITY_CONFIGS)*2} agents")
    print("(Running in background mode - minimal logging)\n")
    
    # Train RL agents (seeds 5010-5014)
    print("\n" + "="*60)
    print("PHASE 1: RL AGENTS")
    print("="*60)
    
    rl_seed = 5010
    for config_name, config in SENSITIVITY_CONFIGS.items():
        # Check if model exists
        model_path = f"models/seg4_sensitivity_rl_{config_name}.zip"
        if os.path.exists(model_path):
            print(f"✓ RL {config_name} already exists - skipping")
            rl_seed += 1
            continue
        
        train_sensitivity_rl(config_name, config, rl_seed)
        rl_seed += 1
    
    # Train ES agents (seeds 5020-5024)
    print("\n" + "="*60)
    print("PHASE 2: ES AGENTS")
    print("="*60)
    
    es_seed = 5020
    for config_name, config in SENSITIVITY_CONFIGS.items():
        # Check if model exists
        model_path = f"models/seg4_sensitivity_es_{config_name}.pkl"
        if os.path.exists(model_path):
            print(f"✓ ES {config_name} already exists - skipping")
            es_seed += 1
            continue
        
        train_sensitivity_es(config_name, config, es_seed)
        es_seed += 1
    
    print("\n" + "="*60)
    print("✅ SENSITIVITY AGENTS TRAINING COMPLETE")
    print("="*60)
    print(f"\nTrained {len(SENSITIVITY_CONFIGS)*2} agents:")
    print("\nRL Agents:")
    for config_name in SENSITIVITY_CONFIGS.keys():
        print(f"  - models/seg4_sensitivity_rl_{config_name}.zip")
    print("\nES Agents:")
    for config_name in SENSITIVITY_CONFIGS.keys():
        print(f"  - models/seg4_sensitivity_es_{config_name}.pkl")
    print("\nNext: Run evaluation with run_segment4_sensitivity.py")

