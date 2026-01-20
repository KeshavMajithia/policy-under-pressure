
import numpy as np
import json
import os
import sys

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.agents.rl import RLAgentFactory
from backend.agents.es import ESAgent
from backend.utils.json_utils import NumpyEncoder

# Configuration
RESULTS_PATH = "frontend/public/gradient_results.json"
MODELS_DIR = "models"
RL_MODEL = "rl_pretrained"
ES_MODEL = "es_pretrained"

SWEEPS = {
    "friction": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    "noise": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
    "delay": [0, 2, 5, 10, 15, 20],
    "mask": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

# 5 Seeds for statistical significance (Prototype level)
SEEDS = [3001, 3002, 3003, 3004, 3005] 

def evaluate_episode(env, agent, sweep_type, level, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    steps = 0
    total_reward = 0
    
    # For Delay Sweep, we might need to re-init env, but here we handle it via logic or config
    # Actually, Reward Delay requires Env init param.
    # We will handle that in the main loop by rebuilding env if needed.
    
    while not done and steps < 1000:
        # 1. Observation Noise
        if sweep_type == "noise":
            noise = np.random.normal(0, level, size=obs.shape).astype(np.float32)
            obs_input = obs + noise
            
            # Manual Clip with Epsilon to avoid float precision errors at bounds
            eps = 1e-4
            pi_safe = np.pi - eps
            
            # 1: Heading (-pi, pi)
            obs_input[1] = np.clip(obs_input[1], -pi_safe, pi_safe)
            # 2: Speed (0, Inf)
            obs_input[2] = max(0.0, obs_input[2])
            # 3: Curvature (-pi, pi)
            obs_input[3] = np.clip(obs_input[3], -pi_safe, pi_safe)
            
            obs_input = obs_input.astype(np.float32)
        else:
            obs_input = obs
            
        # 2. Predict
        try:
            action, _ = agent.predict(obs_input)
        except Exception as e:
            print(f"Predict Error: {e}")
            print(f"Obs: {obs_input}")
            raise e
        
        # 3. Action Masking
        if sweep_type == "mask":
            if np.random.rand() < level:
                action = np.array([0.0, 0.0], dtype=np.float32) # Coast
                
        # 4. Step
        step_config = {}
        if sweep_type == "friction":
            step_config["friction"] = level
            
        obs, r, term, trunc, info = env.step(action, config=step_config)
        
        total_reward += r
        steps += 1
        done = term or trunc
        
    return {
        "return": total_reward,
        "steps": steps,
        "survived": steps >= 1000
    }

def main():
    print("--- Starting Quantitative Gradient Sweep ---")
    
    # 1. Load Agents
    # We need a dummy env to load agents
    dummy_env = CarEnv(track_type="figure8") # Master agents trained on Fig8
    
    print(f"Loading RL Agent: {RL_MODEL}...")
    # Pass None to avoid observation space mismatch check (SB3 is strict)
    rl_agent = RLAgentFactory.load(f"{MODELS_DIR}/{RL_MODEL}.zip", None)
    
    print(f"Loading ES Agent: {ES_MODEL}...")
    es_agent = ESAgent(dummy_env.observation_space.shape[0], dummy_env.action_space.shape[0], hidden_dim=64)
    es_agent.load(f"{MODELS_DIR}/{ES_MODEL}.pkl")
    
    results = {
        "friction": {"RL": [], "ES": []},
        "noise": {"RL": [], "ES": []},
        "delay": {"RL": [], "ES": []},
        "mask": {"RL": [], "ES": []}
    }
    
    # 2. Run Sweeps
    for sweep_name, levels in SWEEPS.items():
        print(f"\nRunning Sweep: {sweep_name.upper()}")
        
        for level in levels:
            print(f"  Level: {level}", end="", flush=True)
            
            # Prepare Env
            # Delay requires specific init
            reward_delay = level if sweep_name == "delay" else 0
            # Track type: Use 'random' for robustness? Or 'fig8' for consistency?
            # Protocol says "Generalization" is separate.
            # To test PURE robustness to physics/noise, we should keep Track CONSTANT (Fig8).
            # Otherwise we conflate map difficulty with severity.
            env = CarEnv(track_type="figure8", reward_delay_steps=reward_delay)
            
            # RL
            rl_metrics = []
            for seed in SEEDS:
                m = evaluate_episode(env, rl_agent, sweep_name, level, seed)
                rl_metrics.append(m)
            
            # ES
            es_metrics = []
            for seed in SEEDS:
                m = evaluate_episode(env, es_agent, sweep_name, level, seed)
                es_metrics.append(m)
                
            # Aggregation
            def aggregate(ms):
                return {
                    "level": level,
                    "survival_rate": np.mean([1.0 if m["survived"] else 0.0 for m in ms]),
                    "avg_return": np.mean([m["return"] for m in ms]),
                    "avg_steps": np.mean([m["steps"] for m in ms]),
                    "std_return": np.std([m["return"] for m in ms])
                }
                
            results[sweep_name]["RL"].append(aggregate(rl_metrics))
            results[sweep_name]["ES"].append(aggregate(es_metrics))
            
            print(" [Done]")
            
    # 3. Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
        
    print(f"\nSaved Gradient Results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
