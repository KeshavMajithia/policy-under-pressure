import numpy as np
import json
import os
import argparse
from stable_baselines3 import PPO
from backend.agents.es import ESAgent
from backend.env.car_env import CarEnv
from backend.experiments.core import ExperimentRunner, ExperimentConfig
from backend.experiments.wrappers import NoiseWrapper

def make_env(config: ExperimentConfig):
    # Factory function for the env
    env = CarEnv(reward_type=config.reward_type)
    return env

def load_agents():
    agents = {}
    
    # Load RL
    try:
        if os.path.exists("models/ppo_car.zip"):
            agents["RL"] = PPO.load("models/ppo_car")
            print("Loaded RL Agent")
    except Exception as e:
        print(f"Failed to load RL: {e}")

    # Load ES
    try:
        if os.path.exists("models/es_car.pkl"):
            # Determine dimensions from env
            dummy_env = CarEnv()
            input_dim = dummy_env.observation_space.shape[0]
            output_dim = dummy_env.action_space.shape[0]
            
            es_agent = ESAgent(input_dim, output_dim)
            es_agent.load("models/es_car.pkl")
            agents["ES"] = es_agent
            print(f"Loaded ES Agent (In: {input_dim}, Out: {output_dim})")
    except Exception as e:
        print(f"Failed to load ES: {e}")
        
    return agents

def run_suite():
    agents = load_agents()
    if not agents:
        print("No agents found!")
        return

    runner = ExperimentRunner(make_env)
    
    # Compare across Noise Levels
    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    
    results_data = {
        "metadata": {
            "experiment": "Robustness to Observation Noise",
            "noise_levels": noise_levels,
            "agents": list(agents.keys())
        },
        "results": {}
    }
    
    for agent_name, agent in agents.items():
        agent_results = []
        print(f"\n--- Benchmarking {agent_name} ---")
        
        for noise in noise_levels:
            config = ExperimentConfig(
                name=f"{agent_name}_Noise_{noise}",
                num_episodes=20, # Statistical significance
                noise_level=noise,
                wrappers=[(NoiseWrapper, {"obs_noise_std": noise})]
            )
            
            result = runner.run(agent, config)
            
            summary = {
                "noise": float(noise),
                "mean_reward": float(result.metrics["mean_reward"]),
                "std_reward": float(result.metrics["std_reward"]),
                "success_rate": float(result.success_rate),
                "mean_length": float(result.metrics["mean_length"])
            }
            agent_results.append(summary)
            print(f"Noise {noise}: Reward={summary['mean_reward']:.1f} +/- {summary['std_reward']:.1f}")
            
        results_data["results"][agent_name] = agent_results

    # Save to Frontend
    out_path = "frontend/public/exp_robustness.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_data, f, indent=2)
        
    print(f"\nExperiment Complete. Results saved to {out_path}")

if __name__ == "__main__":
    run_suite()
