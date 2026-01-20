
import numpy as np
import json
import os
import sys

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.agents.rl import RLAgentFactory
from backend.agents.es import ESAgent
from backend.rewards.control import ControlReward

EXPERIMENT_RESULTS_PATH = "frontend/public/experiment_results.json"

# --- HELPER FUNCTIONS ---

def run_episode(env, agent, config=None, max_steps=3500, desc="", seed=None):
    # Set seed if provided for reproducible map generation
    if seed is not None:
        np.random.seed(seed) # Force global seed for Track generation
        
    options = config or {}
    obs, _ = env.reset(options=options, seed=seed)
    
    trajectory = []
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and steps < max_steps:
        # Construct Step Config for dynamic effects
        step_config = {}
        
        # Exp 2: Ice Patch Logic (Dynamic Friction)
        if config and config.get("ice_patch"):
            # Trigger between step 200 and 220 (20s to 22s)
            if 200 <= steps <= 220:
                step_config["friction"] = 0.3
        
        # Exp 3: Constant Noise
        if config and config.get("noise_type"):
            if config["noise_type"] == "heading":
                step_config["noise"] = [0.0, np.random.normal(0, 0.1), 0.0, 0.0]
            elif config["noise_type"] == "lateral":
                step_config["noise"] = [np.random.normal(0, 0.5), 0.0, 0.0, 0.0]
                
        # Exp 4: Blindfold (Constant Mask)
        if config and config.get("blindfold"):
            step_config["mask"] = [2] # Mask Index 2 (Speed)
            
        action, _ = agent.predict(obs)
        if isinstance(action, np.ndarray): action = action.tolist()
        
        obs, reward, terminated, truncated, info = env.step(action, config=step_config)
        
        # Log data
        trajectory.append({
            "t": round(steps * 0.1, 2),
            "x": float(info["x"]),
            "y": float(info["y"]),
            "speed": float(info["speed"]),
            "lat_error": float(info["lateral_error"]),
            "heading_error": float(info["heading_error"]),
            "friction": step_config.get("friction", 1.0)
        })
        
        total_reward += reward
        steps += 1
        
    return {
        "trajectory": trajectory,
        "total_reward": total_reward,
        "steps": steps,
        "completed": (info["progress"] > 0.95),
        "mean_speed": np.mean([t["speed"] for t in trajectory]),
        "max_lat_error": np.max([abs(t["lat_error"]) for t in trajectory])
    }

# --- EXPERIMENTS ---

def exp_1_new_world_shared(env, agents, n_runs=5):
    # Generalization: Random Tracks with SHARED SEEDS
    print("  Running Exp 1: New World (Shared Seeds)...")
    
    # Generate N random seeds
    seeds = [np.random.randint(0, 10000) for _ in range(n_runs)]
    
    final_results = {}
    
    for name, agent in agents.items():
        print(f"    Agent: {name}")
        results = []
        
        # 1. Baseline (Figure 8) - No seed needed for fixed track
        env.track = env.track.__class__(track_type="figure8") 
        base_run = run_episode(env, agent, desc="Baseline")
        
        # 2. Random Tracks (Shared Seeds)
        env.track = env.track.__class__(track_type="random") 
        
        for i, seed in enumerate(seeds):
            # Pass seed to enforce identical map generation for both agents
            run = run_episode(env, agent, seed=seed, desc=f"Random {i}")
            results.append(run)
            
        # Metrics
        avg_speed_random = np.mean([r["mean_speed"] for r in results])
        efficiency = avg_speed_random / (base_run["mean_speed"] + 1e-6)
        survival = np.mean([1.0 if r["completed"] else 0.0 for r in results])
        
        final_results[name] = {
            "metrics": {"efficiency": efficiency, "survival": survival},
            "sample_trajectory": results[0]["trajectory"] # Show first seed for all agents
        }
        
    return final_results

def exp_2_ice_patch(env, agent, n_runs=5):
    # Physics: Friction Drop
    print("  Running Exp 2: Ice Patch...")
    # Reset to Figure 8
    env.track = env.track.__class__(track_type="figure8")
    
    results = []
    for i in range(n_runs):
        run = run_episode(env, agent, config={"ice_patch": True}, desc=f"Ice {i}")
        results.append(run)
        
    # Analyze Deviation during Ice Patch (Steps 200-250)
    deviations = []
    recoveries = []
    
    for r in results:
        # Slice trajectory around ice patch
        patch_logs = r["trajectory"][200:250] if len(r["trajectory"]) > 250 else r["trajectory"]
        if not patch_logs: continue
        
        max_dev = np.max([abs(t["lat_error"]) for t in patch_logs])
        deviations.append(max_dev)
        recoveries.append(1.0 if r["completed"] else 0.0)
        
    return {
        "metrics": {
            "max_deviation": float(np.mean(deviations)) if deviations else 0.0,
            "recovery_rate": float(np.mean(recoveries))
        },
        "sample_trajectory": results[0]["trajectory"]
    }

def exp_3_foggy_sensor(env, agent, n_runs=5):
    # Noise: Heading vs Lateral
    print("  Running Exp 3: Foggy Sensor...")
    env.track = env.track.__class__(track_type="figure8")
    
    # Sub A: Heading Noise
    res_h = []
    for _ in range(n_runs):
        res_h.append(run_episode(env, agent, config={"noise_type": "heading"}))
        
    # Sub B: Lateral Noise
    res_l = []
    for _ in range(n_runs):
        res_l.append(run_episode(env, agent, config={"noise_type": "lateral"}))

    return {
        "metrics": {
            "heading_survival": float(np.mean([r["completed"] for r in res_h])),
            "lateral_survival": float(np.mean([r["completed"] for r in res_l])),
            "heading_wobble": float(np.mean([r["max_lat_error"] for r in res_h])),
            "lateral_wobble": float(np.mean([r["max_lat_error"] for r in res_l]))
        },
        "sample_trajectory": res_h[0]["trajectory"] # Show Heading Noise sample
    }

def exp_4_blindfold(env, agent, n_runs=5):
    # Partial Obs: No Speed
    print("  Running Exp 4: Blindfold...")
    env.track = env.track.__class__(track_type="figure8")
    
    results = []
    for _ in range(n_runs):
        results.append(run_episode(env, agent, config={"blindfold": True}))
        
    return {
        "metrics": {
            "survival": float(np.mean([r["completed"] for r in results])),
            "avg_speed": float(np.mean([r["mean_speed"] for r in results]))
        },
        "sample_trajectory": results[0]["trajectory"]
    }

def exp_5_wake_up_call(env, agent):
    # Adversarial Starts
    print("  Running Exp 5: Wake Up Call...")
    env.track = env.track.__class__(track_type="figure8")
    
    scenarios = {
        "wall_facer": [50.0, 30.0, np.pi/4], # Facing outside wall at start
        "mid_turn": [89.0, 31.0, np.pi/2] # Middle of turn 1
    }
    
    metrics = {}
    trajs = {}
    
    for name, pose in scenarios.items():
        runs = []
        for _ in range(3):
            # Pass start_pose to reset
            runs.append(run_episode(env, agent, config={"start_pose": pose}))
            
        metrics[name + "_success"] = float(np.mean([r["completed"] for r in runs]))
        trajs[name] = runs[0]["trajectory"]
        
    return {
        "metrics": metrics,
        "sample_trajectory": trajs["wall_facer"] # Just show one
    }


def main():
    print("="*60)
    print("PHASE 2: BEHAVIORAL EXPERIMENTS (STRESS TESTS) v3")
    print("="*60)
    
    # 1. Load Agents
    env = CarEnv(track_type="figure8")
    
    # RL Agent
    print("Loading RL Agent (rl_phase2_fixed)...")
    rl_agent = RLAgentFactory.load("models/rl_phase2_fixed", env)
    
    # ES Agent
    print("Loading ES Agent (es_phase2_strict.pkl)...")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    es_agent = ESAgent(input_dim, output_dim, hidden_dim=64)
    es_agent.load("models/es_phase2_strict.pkl")

    agents = {"RL": rl_agent, "ES": es_agent}
    final_output = {"RL": {}, "ES": {}}

    # EXPERIMENT 1: SHARED MAPS
    print("\n--- Running Exp 1: New World (Shared) ---")
    exp1_results = exp_1_new_world_shared(env, agents)
    final_output["RL"]["exp_1"] = exp1_results["RL"]
    final_output["ES"]["exp_1"] = exp1_results["ES"]

    # EXPERIMENTS 2-5: INDIVIDUAL RUNS
    for name, agent in agents.items():
        print(f"\n--- Testing Agent: {name} (Exps 2-5) ---")
        agent_results = final_output[name] 
        
        agent_results["exp_2"] = exp_2_ice_patch(env, agent)
        agent_results["exp_3"] = exp_3_foggy_sensor(env, agent)
        agent_results["exp_4"] = exp_4_blindfold(env, agent)
        agent_results["exp_5"] = exp_5_wake_up_call(env, agent)
        
    # 3. Save
    os.makedirs(os.path.dirname(EXPERIMENT_RESULTS_PATH), exist_ok=True)
    with open(EXPERIMENT_RESULTS_PATH, "w") as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\nSaved all results to {EXPERIMENT_RESULTS_PATH}")
    
if __name__ == "__main__":
    main()
