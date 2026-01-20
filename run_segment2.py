
import numpy as np
import json
import os
import sys
import shutil

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.agents.rl import RLAgentFactory
from backend.agents.es import ESAgent
from backend.rewards.cheating import SpeedDemonReward, ParkerReward, SensitivityReward
from backend.utils.json_utils import NumpyEncoder

EXPERIMENT_RESULTS_PATH = "frontend/public/segment2_results.json"
MODELS_DIR = "models/segment2"

# --- TRAINING HARNESS ---

def train_rl(env, name, total_timesteps=100000):
    """Trains a new PPO agent from scratch, or loads if exists."""
    path = f"{MODELS_DIR}/{name}"
    zip_path = f"{path}.zip"
    
    if os.path.exists(zip_path):
        print(f"  [RECOVERY] Loading existing RL Agent: {name}...")
        agent = RLAgentFactory.load(zip_path, env)
        return agent
        
    print(f"  Training RL Agent: {name}...")
    agent = RLAgentFactory.create(env, verbose=0)
    agent.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    agent.save(path)
    return agent

def train_es(env, name, generations=50, pop_size=64):
    """Trains a new ES agent from scratch, or loads if exists."""
    path = f"{MODELS_DIR}/{name}.pkl"
    
    if os.path.exists(path):
        print(f"  [RECOVERY] Loading existing ES Agent: {name}...")
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.shape[0]
        agent = ESAgent(input_dim, output_dim, hidden_dim=64)
        agent.load(path)
        return agent

    print(f"  Training ES Agent: {name}...")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    agent = ESAgent(input_dim, output_dim, hidden_dim=64)
    
    # Simple training loop for ES
    # We will just run for 'generations'
    for g in range(generations):
        # 1. Perturb and Evaluate
        results = []
        for i in range(pop_size):
            noise_key, perturbed_model = agent.get_perturbed_agent()
            
            # Run Episode
            obs, _ = env.reset()
            total_r = 0
            done = False
            steps = 0
            while not done and steps < 1000: # Training max steps (shorter than eval)
                action, _ = perturbed_model.predict(obs)
                obs, r, term, trunc, _ = env.step(action)
                total_r += r
                steps += 1
                done = term or trunc
                
            results.append((noise_key, total_r))
            
        # 2. Update
        agent.update(results)
        
        if (g+1) % 10 == 0:
            print(f"    Gen {g+1}/{generations} - Best Reward: {np.max([r for k,r in results]):.2f}")
            
    agent.save(path)
    return agent

# --- EVALUATION HARNESS ---

def evaluate(env, agent, n_runs=3, max_steps=3500):
    """Runs evaluation episodes with shared seeds logic."""
    seeds = [1001, 1002, 1003] # Fixed evaluation seeds
    trajectories = []
    metrics = []
    
    for i in range(n_runs):
        seed = seeds[i]
        np.random.seed(seed)
        obs, _ = env.reset(seed=seed)
        
        traj = []
        total_r = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
             action, _ = agent.predict(obs)
             obs, r, term, trunc, info = env.step(action)
             
             traj.append({
                 "t": steps * 0.1,
                 "x": float(info["x"]),
                 "y": float(info["y"]),
                 "speed": float(info["speed"]),
                 "lat_error": float(info["lateral_error"]),
                 "heading_error": float(info["heading_error"])
             })
             
             total_r += r
             steps += 1
             done = term or trunc
             
        trajectories.append(traj)
        
        # Calculate Metrics
        pct_off_track = np.mean([1 if abs(t["lat_error"]) > 4.0 else 0 for t in traj])
        metrics.append({
            "return": total_r,
            "steps": steps,
            "pct_off_track": pct_off_track,
            "avg_speed": np.mean([t["speed"] for t in traj])
        })
        
    return {
        "metrics": metrics,
        "sample_trajectory": trajectories[0]
    }

# --- EXPERIMENTS ---

def exp_1_speed_demon(env_builder):
    print("\n--- Exp 1: The Speed Demon (Cheating) ---")
    results = {}
    
    # 1. Setup Env with Cheating Reward
    env = env_builder(reward_fn=SpeedDemonReward())
    
    # 2. Train New Agents
    # RL
    rl_agent = train_rl(env, "rl_speed_demon", total_timesteps=50000)
    res_rl = evaluate(env, rl_agent)
    
    # ES
    es_agent = train_es(env, "es_speed_demon", generations=20)
    res_es = evaluate(env, es_agent)
    
    results["RL"] = res_rl
    results["ES"] = res_es
    
    return results

def exp_2_sensitivity(env_builder):
    print("\n--- Exp 2: The Butterfly Effect (Sensitivity) ---")
    results = {"0.8": {}, "1.0": {}, "1.2": {}}
    
    # We test sensitivity of RL primarily (as per hypothesis)
    coeffs = [0.8, 1.0, 1.2]
    
    for k in coeffs:
        print(f"  Testing Penalty Coefficient: {k}")
        # Build env with SPECIFIC safety coefficient
        # Note: We need to pass this to the Reward Function
        reward_fn = SensitivityReward(safety_coeff=k)
        env = env_builder(reward_fn=reward_fn)
        
        # Train new RL agent
        name = f"rl_sens_{k}"
        try:
             agent = train_rl(env, name, total_timesteps=50000)
             eval_res = evaluate(env, agent)
        except Exception as e:
             print(f"  [CRASH] Training failed for {name}: {e}")
             # Return dummy metric showing failure
             eval_res = {
                 "metrics": [{"return": 0.0, "steps": 0, "pct_off_track": 1.0, "avg_speed": 0.0}],
                 "sample_trajectory": []
             }
        
        results[str(k)] = eval_res
        
    return results

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(EXPERIMENT_RESULTS_PATH), exist_ok=True)
    
    final_output = {}
    
    # Helper to build env
    def build_env(reward_fn):
        # Always use Complex Spline (Random generic or Fixed? Protocol says Complex Spline)
        # But 'random' track type generates new track on reset.
        # We want CONSISTENT track for training? Or random?
        # Generalization test used random.
        # For Cheating, we want them to learn to cheat on THIS track or generally?
        # Protocol: "Map: Complex Spline (25 Anchors)".
        # Let's use 'random' type but seeded in run_experiments?
        # For training, let's use a Fixed Complex Track (generated once).
        # Actually CarEnv(track_type='random') regenerates on reset unless we fix seed.
        # To make training stable, let's use proper random procedural generation (generalization style).
        return CarEnv(track_type="random", reward_fn=reward_fn)

    # Exp 1
    final_output["exp_1"] = exp_1_speed_demon(build_env)
    
    # Exp 2
    final_output["exp_2"] = exp_2_sensitivity(build_env)
    
    # Save
    with open(EXPERIMENT_RESULTS_PATH, "w") as f:
        json.dump(final_output, f, indent=2, cls=NumpyEncoder)
        
    print(f"\nSaved Segment 2 results to {EXPERIMENT_RESULTS_PATH}")

if __name__ == "__main__":
    main()
