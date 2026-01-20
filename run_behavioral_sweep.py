
import numpy as np
import json
import os
import sys

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.agents.rl import RLAgentFactory
from backend.agents.es import ESAgent
from backend.rewards.control import ControlReward
from backend.utils.json_utils import NumpyEncoder

# Configuration
RESULTS_PATH = "frontend/public/gradient_results_v2.json"
MODELS_DIR = "models"
RL_MODEL = "rl_phase2_fixed"
ES_MODEL = "es_phase2_strict"

SWEEPS = {
    "friction": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
    "noise": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "delay": [0, 2, 5, 10, 15, 20],
    "mask": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

SEEDS = [3001, 3002, 3003, 3004, 3005]

def evaluate_behavioral(env, agent, sweep_type, level, seed, save_trajectory=False):
    """Evaluate using behavioral metrics instead of returns"""
    obs, _ = env.reset(seed=seed)
    done = False
    steps = 0
    
    # Tracking arrays
    speeds = []
    lat_errors = []
    steering_actions = []
    trajectory = [] if save_trajectory else None
    crashed = False
    
    while not done and steps < 1000:
        # 1. Observation Noise
        if sweep_type == "noise":
            noise = np.random.normal(0, level, size=obs.shape).astype(np.float32)
            obs_input = obs + noise
            
            # Manual Clip with Epsilon
            eps = 1e-4
            pi_safe = np.pi - eps
            obs_input[1] = np.clip(obs_input[1], -pi_safe, pi_safe)
            obs_input[2] = max(0.0, obs_input[2])
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
                action = np.array([0.0, 0.0], dtype=np.float32)
                
        # 4. Step
        step_config = {}
        if sweep_type == "friction":
            step_config["friction"] = level
            
        obs, r, term, trunc, info = env.step(action, config=step_config)
        
        # Record behavioral metrics
        speeds.append(info["speed"])
        lat_errors.append(abs(info["lateral_error"]))
        steering_actions.append(action[0])
        
        # Record trajectory if requested
        if save_trajectory:
            trajectory.append({
                "t": steps * 0.1,
                "x": float(info["x"]),
                "y": float(info["y"]),
                "speed": float(info["speed"]),
                "lat_error": float(info["lateral_error"]),
                "heading_error": float(info["heading_error"])
            })
        
        steps += 1
        done = term or trunc
        
        if info.get("off_track", False):
            crashed = True
    
    # Compute behavioral metrics
    result = {
        "mean_speed": float(np.mean(speeds)),
        "lat_error_rms": float(np.sqrt(np.mean(np.square(lat_errors)))),
        "steering_variance": float(np.var(steering_actions)),
        "time_to_crash": float(steps * 0.1) if crashed else 100.0,
        "survival_rate": 0.0 if crashed else 1.0,
        "steps": steps
    }
    
    if save_trajectory:
        result["trajectory"] = trajectory
    
    return result

def main():
    print("--- Starting Corrected Behavioral Sweep (v2) ---")
    print("Using ControlReward (matching training)")
    print("Tracking: Speed, Lat Error RMS, Steering Variance, Time-to-Crash")
    
    # 1. Load Dummy Env to Load Agents
    from backend.env.track import Track
    dummy_track = Track(track_type="figure8")
    dummy_env = CarEnv(track_type="figure8", reward_fn=ControlReward(dummy_track))
    
    print(f"\nLoading RL Agent: {RL_MODEL}...")
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
        print(f"\n🔬 Running Sweep: {sweep_name.upper()}")
        
        for level in levels:
            print(f"  Level: {level}", end="", flush=True)
            
            # Prepare Env with ControlReward
            reward_delay = level if sweep_name == "delay" else 0
            track = Track(track_type="figure8")
            env = CarEnv(track_type="figure8", reward_fn=ControlReward(track), reward_delay_steps=reward_delay)
            
            # Increase Difficulty: Stricter termination
            env.track.track_width = 6.0  # Narrower track (was 8.0)
            
            # RL
            rl_metrics = []
            rl_sample_traj = None
            for i, seed in enumerate(SEEDS):
                save_traj = (i == 0)  # Save first seed's trajectory
                m = evaluate_behavioral(env, rl_agent, sweep_name, level, seed, save_trajectory=save_traj)
                if save_traj:
                    rl_sample_traj = m.pop("trajectory", None)
                rl_metrics.append(m)
            
            # ES
            es_metrics = []
            es_sample_traj = None
            for i, seed in enumerate(SEEDS):
                save_traj = (i == 0)
                m = evaluate_behavioral(env, es_agent, sweep_name, level, seed, save_trajectory=save_traj)
                if save_traj:
                    es_sample_traj = m.pop("trajectory", None)
                es_metrics.append(m)
                
            # Aggregation
            def aggregate(ms):
                return {
                    "level": level,
                    "mean_speed": np.mean([m["mean_speed"] for m in ms]),
                    "lat_error_rms": np.mean([m["lat_error_rms"] for m in ms]),
                    "steering_variance": np.mean([m["steering_variance"] for m in ms]),
                    "time_to_crash": np.mean([m["time_to_crash"] for m in ms]),
                    "survival_rate": np.mean([m["survival_rate"] for m in ms]),
                    "std_speed": np.std([m["mean_speed"] for m in ms]),
                    "std_lat_error": np.std([m["lat_error_rms"] for m in ms])
                }
            
            rl_agg = aggregate(rl_metrics)
            es_agg = aggregate(es_metrics)
            
            # Add sample trajectories
            rl_agg["sample_trajectory"] = rl_sample_traj
            es_agg["sample_trajectory"] = es_sample_traj
                
            results[sweep_name]["RL"].append(rl_agg)
            results[sweep_name]["ES"].append(es_agg)
            
            print(" ✓")
            
    # 3. Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
        
    print(f"\n✅ Saved Corrected Gradient Results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
