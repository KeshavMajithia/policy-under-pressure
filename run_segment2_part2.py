
import numpy as np
import json
import os
import sys

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.agents.rl import RLAgentFactory
from backend.agents.es import ESAgent
from backend.rewards.cheating import SpeedDemonReward, ParkerReward, SensitivityReward
from backend.utils.json_utils import NumpyEncoder

EXPERIMENT_RESULTS_PATH = "frontend/public/segment2_part2_results.json"
MODELS_DIR = "models/segment2"

# --- REUSED TRAINING FUNCTIONS ---
# (Cloned for independence)

def train_rl(env, name, total_timesteps=50000):
    """Trains a new PPO agent, or loads if exists."""
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

def train_es(env, name, generations=20, pop_size=64):
    """Trains a new ES agent, or loads if exists."""
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
    
    for g in range(generations):
        results = []
        for i in range(pop_size):
            noise_key, perturbed_model = agent.get_perturbed_agent()
            obs, _ = env.reset()
            total_r = 0
            done = False
            steps = 0
            while not done and steps < 1000:
                action, _ = perturbed_model.predict(obs)
                obs, r, term, trunc, _ = env.step(action)
                total_r += r
                steps += 1
                done = term or trunc
            results.append((noise_key, total_r))
        agent.update(results)
        if (g+1) % 5 == 0:
            print(f"    Gen {g+1}/{generations} - Best: {np.max([r for k,r in results]):.2f}")
            
    agent.save(path)
    return agent

def evaluate(env, agent, n_runs=3, max_steps=3500):
    seeds = [2001, 2002, 2003]
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
        # Exp 3: Forward Distance vs Alignment
        # Exp 4: Return
        # Exp 5: Lat Deviation sums
        metrics.append({
            "return": total_r,
            "steps": steps,
            "final_x": float(info["x"]),
            "cum_lat_error": np.sum([abs(t["lat_error"]) for t in traj]),
            "avg_speed": np.mean([t["speed"] for t in traj])
        })
        
    return {
        "metrics": metrics,
        "sample_trajectory": trajectories[0]
    }

# --- EXPERIMENTS 3, 4, 5 ---

def exp_3_parker(env_builder):
    print("\n--- Exp 3: The Parker (Alignment) ---")
    results = {}
    
    # 1. Setup Env with Parker Reward (Proxy)
    env = env_builder(reward_fn=ParkerReward())
    
    # 2. Train
    rl_agent = train_rl(env, "rl_parker")
    res_rl = evaluate(env, rl_agent)
    
    es_agent = train_es(env, "es_parker")
    res_es = evaluate(env, es_agent)
    
    results["RL"] = res_rl
    results["ES"] = res_es
    return results

def exp_4_lag(env_builder):
    print("\n--- Exp 4: The Lag (Reward Delay) ---")
    results = {}
    
    # Setup Env with Delay=10 steps (1.0s)
    # Using standard ProgressReward (default in CarEnv if reward_fn=None)
    # But pass reward_delay_steps=10
    env = env_builder(reward_delay_steps=10)
    
    rl_agent = train_rl(env, "rl_lag")
    res_rl = evaluate(env, rl_agent)
    
    es_agent = train_es(env, "es_lag")
    res_es = evaluate(env, es_agent)
    
    results["RL"] = res_rl
    results["ES"] = res_es
    return results

def exp_5_drift(env_builder):
    print("\n--- Exp 5: The Silent Drift (Friction) ---")
    results = {}
    
    # Setup Env with Friction=0.95 constant
    env = env_builder(friction_scale=0.95)
    
    rl_agent = train_rl(env, "rl_drift")
    res_rl = evaluate(env, rl_agent)
    
    es_agent = train_es(env, "es_drift")
    res_es = evaluate(env, es_agent)
    
    results["RL"] = res_rl
    results["ES"] = res_es
    return results

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Helper to build env with various params
    def build_env(**kwargs):
        # Default to random track 
        return CarEnv(track_type="random", **kwargs)

    # Run New Experiments
    part2_results = {}
    part2_results["exp_3"] = exp_3_parker(build_env)
    part2_results["exp_4"] = exp_4_lag(build_env)
    part2_results["exp_5"] = exp_5_drift(build_env)
    
    # Save to dedicated Part 2 file
    with open(EXPERIMENT_RESULTS_PATH, "w") as f:
        json.dump(part2_results, f, indent=2, cls=NumpyEncoder)
        
    print(f"\nSaved Segment 2 Part 2 results to {EXPERIMENT_RESULTS_PATH}")

if __name__ == "__main__":
    main()
