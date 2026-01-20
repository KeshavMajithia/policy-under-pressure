"""
Segment 4 Evaluation: Experiment 2 - Reward Sensitivity

Evaluates sensitivity agents and computes:
1. Policy divergence from baseline
2. Performance variance across configs
3. Training stability indicators

Protocol: segment4_protocol.md (Experiment 2)
Seeds: [4001, 4002, 4003, 4004, 4005]
Configs: baseline, minus_20, minus_10, plus_10, plus_20
"""

import numpy as np
import json
import sys
import os
from stable_baselines3 import PPO

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.env.track import Track
from backend.rewards.control import ControlReward
from backend.agents.es import ESAgent


EVAL_SEEDS = [4001, 4002, 4003, 4004, 4005]
CONFIGS = ["baseline", "minus_20", "minus_10", "plus_10", "plus_20"]


def evaluate_agent(env, agent, agent_type, seed, episode_length=1000):
    """Evaluate agent performance"""
    np.random.seed(seed)
    
    if hasattr(agent, 'reset'):
        agent.reset()
    env.reward_fn.reset()
    obs, _ = env.reset(seed=seed)
    
    total_reward = 0
    actions = []
    speeds = []
    lat_errors = []
    
    for step in range(episode_length):
        if agent_type == "RL":
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action, _ = agent.predict(obs)
        
        actions.append(action.copy())
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        speeds.append(info.get('speed', 0))
        lat_errors.append(abs(info.get('lateral_error', 0)))
        
        if terminated or truncated:
            break
    
    return {
        "seed": seed,
        "total_reward": float(total_reward),
        "mean_speed": float(np.mean(speeds)),
        "lat_error_rms": float(np.sqrt(np.mean(np.array(lat_errors)**2))),
        "actions": np.array(actions)
    }


def compute_policy_divergence(baseline_actions, config_actions):
    """
    Compute L2 distance between action sequences
    """
    min_len = min(len(baseline_actions), len(config_actions))
    baseline_clip = baseline_actions[:min_len]
    config_clip = config_actions[:min_len]
    
    divergence = np.mean(np.linalg.norm(baseline_clip - config_clip, axis=1))
    return float(divergence)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SEGMENT 4 - EXPERIMENT 2: REWARD SENSITIVITY EVALUATION")
    print("="*60)
    
    # Setup environment (use ControlReward for evaluation)
    track = Track(track_type="figure8")
    env = CarEnv(track_type="figure8", reward_fn=ControlReward(track))
    
    results = {
        "RL": {},
        "ES": {}
    }
    
    # Load and evaluate all agents
    print("\n" + "="*60)
    print("Loading and Evaluating Agents")
    print("="*60)
    
    # RL Agents
    print("\n--- RL Agents ---")
    rl_baseline_actions = None
    
    for config in CONFIGS:
        print(f"\nConfig: {config}")
        model_path = f"models/seg4_sensitivity_rl_{config}"
        agent = PPO.load(model_path, env=env)
        
        config_results = []
        config_actions_all = []
        
        for seed in EVAL_SEEDS:
            metrics = evaluate_agent(env, agent, "RL", seed)
            actions = metrics.pop("actions")
            config_actions_all.append(actions)
            config_results.append(metrics)
        
        # Store baseline actions for divergence computation
        if config == "baseline":
            rl_baseline_actions = config_actions_all
        
        results["RL"][config] = config_results
        
        # Compute divergence from baseline
        if config != "baseline" and rl_baseline_actions is not None:
            divergences = []
            for i in range(len(EVAL_SEEDS)):
                div = compute_policy_divergence(rl_baseline_actions[i], config_actions_all[i])
                divergences.append(div)
            
            avg_divergence = np.mean(divergences)
            print(f"  Divergence from baseline: {avg_divergence:.4f}")
            
            # Add to results
            for i, res in enumerate(results["RL"][config]):
                res["divergence"] = divergences[i]
    
    # ES Agents
    print("\n--- ES Agents ---")
    es_baseline_actions = None
    
    for config in CONFIGS:
        print(f"\nConfig: {config}")
        model_path = f"models/seg4_sensitivity_es_{config}.pkl"
        agent = ESAgent(input_dim=4, output_dim=2, hidden_dim=64)
        agent.load(model_path)
        
        config_results = []
        config_actions_all = []
        
        for seed in EVAL_SEEDS:
            metrics = evaluate_agent(env, agent, "ES", seed)
            actions = metrics.pop("actions")
            config_actions_all.append(actions)
            config_results.append(metrics)
        
        if config == "baseline":
            es_baseline_actions = config_actions_all
        
        results["ES"][config] = config_results
        
        # Compute divergence
        if config != "baseline" and es_baseline_actions is not None:
            divergences = []
            for i in range(len(EVAL_SEEDS)):
                div = compute_policy_divergence(es_baseline_actions[i], config_actions_all[i])
                divergences.append(div)
            
            avg_divergence = np.mean(divergences)
            print(f"  Divergence from baseline: {avg_divergence:.4f}")
            
            for i, res in enumerate(results["ES"][config]):
                res["divergence"] = divergences[i]
    
    # Compute aggregated statistics
    print("\n" + "="*60)
    print("Computing Aggregated Metrics")
    print("="*60)
    
    aggregated = {"RL": {}, "ES": {}}
    
    for agent_type in ["RL", "ES"]:
        print(f"\n{agent_type}:")
        for config in CONFIGS:
            data = results[agent_type][config]
            
            agg = {
                "mean_reward": float(np.mean([d["total_reward"] for d in data])),
                "std_reward": float(np.std([d["total_reward"] for d in data])),
                "mean_speed": float(np.mean([d["mean_speed"] for d in data])),
                "mean_lat_error": float(np.mean([d["lat_error_rms"] for d in data]))
            }
            
            if config != "baseline":
                agg["mean_divergence"] = float(np.mean([d["divergence"] for d in data]))
                agg["std_divergence"] = float(np.std([d["divergence"] for d in data]))
            
            aggregated[agent_type][config] = agg
            
            print(f"  {config}: Reward={agg['mean_reward']:.1f}, Divergence={agg.get('mean_divergence', 0):.4f}")
    
    # Compute variance across configs (sensitivity measure)
    rl_reward_variance = np.var([aggregated["RL"][c]["mean_reward"] for c in CONFIGS])
    es_reward_variance = np.var([aggregated["ES"][c]["mean_reward"] for c in CONFIGS])
    
    print(f"\nReward Variance Across Configs:")
    print(f"  RL: {rl_reward_variance:.1f}")
    print(f"  ES: {es_reward_variance:.1f}")
    print(f"  Ratio (RL/ES): {rl_reward_variance / (es_reward_variance + 1e-8):.2f}x")
    
    # Save results
    output = {
        "experiment": "sensitivity",
        "seeds": EVAL_SEEDS,
        "configs": CONFIGS,
        "raw_data": results,
        "aggregated": aggregated,
        "summary": {
            "rl_reward_variance": float(rl_reward_variance),
            "es_reward_variance": float(es_reward_variance),
            "sensitivity_ratio": float(rl_reward_variance / (es_reward_variance + 1e-8))
        }
    }
    
    output_path = "frontend/public/segment4_sensitivity.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ SENSITIVITY EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_path}")
    print(f"\nKey Finding: RL is {rl_reward_variance / (es_reward_variance + 1e-8):.1f}x more sensitive to reward changes")
