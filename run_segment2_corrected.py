
import numpy as np
import json
import os
import sys

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.agents.rl import RLAgentFactory
from backend.agents.es import ESAgent
from backend.env.track import Track
from backend.rewards.control import ControlReward
from backend.utils.json_utils import NumpyEncoder

RESULTS_PATH = "frontend/public/segment2_results_v2.json"
MODELS_DIR = "models"
RL_MODEL = "rl_phase2_fixed"
ES_MODEL = "es_phase2_strict"

SEEDS = [2001, 2002, 2003]

def evaluate_behavioral(env, agent, n_runs=3, max_steps=1000):
    """Evaluate using behavioral metrics"""
    seeds = SEEDS[:n_runs]
    trajectories = []
    metrics = []
    
    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        obs, _ = env.reset(seed=seed)
        
        traj = []
        steps = 0
        crashed = False
        
        # Tracking
        speeds = []
        lat_errors = []
        steering_actions = []
        
        done = False
        while not done and steps < max_steps:
            action, _ = agent.predict(obs)
            obs, r, term, trunc, info = env.step(action)
            
            # Record trajectory for visualization
            traj.append({
                "t": steps * 0.1,
                "x": float(info["x"]),
                "y": float(info["y"]),
                "speed": float(info["speed"]),
                "lat_error": float(info["lateral_error"]),
                "heading_error": float(info["heading_error"])
            })
            
            # Record behavioral metrics
            speeds.append(info["speed"])
            lat_errors.append(abs(info["lateral_error"]))
            steering_actions.append(action[0])
            
            steps += 1
            done = term or trunc
            
            if info.get("off_track", False):
                crashed = True
        
        trajectories.append(traj)
        
        # Calculate Metrics
        metrics.append({
            "mean_speed": float(np.mean(speeds)),
            "lat_error_rms": float(np.sqrt(np.mean(np.square(lat_errors)))),
            "steering_variance": float(np.var(steering_actions)),
            "time_to_crash": float(steps * 0.1) if crashed else 100.0,
            "survival_rate": 0.0 if crashed else 1.0,
            "steps": steps
        })
    
    return {
        "metrics": metrics,
        "sample_trajectory": trajectories[0],
        "aggregated": {
            "mean_speed": float(np.mean([m["mean_speed"] for m in metrics])),
            "lat_error_rms": float(np.mean([m["lat_error_rms"] for m in metrics])),
            "survival_rate": float(np.mean([m["survival_rate"] for m in metrics]))
        }
    }

def exp_1_friction_ladder():
    """Friction Robustness: Progressive friction degradation"""
    print("\n--- Exp 1: The Friction Ladder (Physics Robustness) ---")
    results = {}
    
    levels = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    for friction in levels:
        print(f"  Friction: {friction}")
        
        track = Track(track_type="figure8")
        env = CarEnv(track_type="figure8", reward_fn=ControlReward(track), friction_scale=friction)
        env.track.track_width = 6.0  # Tighter track
        
        # Load agents
        rl_agent = RLAgentFactory.load(f"{MODELS_DIR}/{RL_MODEL}.zip", None)
        es_agent = ESAgent(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim=64)
        es_agent.load(f"{MODELS_DIR}/{ES_MODEL}.pkl")
        
        results[str(friction)] = {
            "RL": evaluate_behavioral(env, rl_agent),
            "ES": evaluate_behavioral(env, es_agent)
        }
    
    return results

def exp_2_noise_gradient():
    """Sensor Noise: Progressive noise injection"""
    print("\n--- Exp 2: The Noise Gradient (Sensor Robustness) ---")
    results = {}
    
    levels = [0.0, 0.1, 0.2, 0.3]
    
    for noise_std in levels:
        print(f"  Noise Std: {noise_std}")
        
        track = Track(track_type="figure8")
        env = CarEnv(track_type="figure8", reward_fn=ControlReward(track))
        env.track.track_width = 6.0
        
        # Load agents
        rl_agent = RLAgentFactory.load(f"{MODELS_DIR}/{RL_MODEL}.zip", None)
        es_agent = ESAgent(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim=64)
        es_agent.load(f"{MODELS_DIR}/{ES_MODEL}.pkl")
        
        # Evaluate with noise injection (handled in evaluation)
        def evaluate_with_noise(env, agent):
            seeds = SEEDS
            trajectories = []
            metrics = []
            
            for seed in seeds:
                obs, _ = env.reset(seed=seed)
                traj = []
                speeds = []
                lat_errors = []
                steering_actions = []
                crashed = False
                steps = 0
                done = False
                
                while not done and steps < 1000:
                    # Inject noise
                    if noise_std > 0:
                        noise = np.random.normal(0, noise_std, size=obs.shape).astype(np.float32)
                        obs_noisy = obs + noise
                        # Clip
                        eps = 1e-4
                        pi_safe = np.pi - eps
                        obs_noisy[1] = np.clip(obs_noisy[1], -pi_safe, pi_safe)
                        obs_noisy[2] = max(0.0, obs_noisy[2])
                        obs_noisy[3] = np.clip(obs_noisy[3], -pi_safe, pi_safe)
                        obs_noisy = obs_noisy.astype(np.float32)
                    else:
                        obs_noisy = obs
                    
                    action, _ = agent.predict(obs_noisy)
                    obs, r, term, trunc, info = env.step(action)
                    
                    traj.append({
                        "t": steps * 0.1,
                        "x": float(info["x"]),
                        "y": float(info["y"]),
                        "speed": float(info["speed"]),
                        "lat_error": float(info["lateral_error"]),
                        "heading_error": float(info["heading_error"])
                    })
                    
                    speeds.append(info["speed"])
                    lat_errors.append(abs(info["lateral_error"]))
                    steering_actions.append(action[0])
                    
                    steps += 1
                    done = term or trunc
                    if info.get("off_track", False):
                        crashed = True
                
                trajectories.append(traj)
                metrics.append({
                    "mean_speed": float(np.mean(speeds)),
                    "lat_error_rms": float(np.sqrt(np.mean(np.square(lat_errors)))),
                    "steering_variance": float(np.var(steering_actions)),
                    "time_to_crash": float(steps * 0.1) if crashed else 100.0,
                    "survival_rate": 0.0 if crashed else 1.0
                })
            
            return {
                "metrics": metrics,
                "sample_trajectory": trajectories[0],
                "aggregated": {
                    "mean_speed": float(np.mean([m["mean_speed"] for m in metrics])),
                    "lat_error_rms": float(np.mean([m["lat_error_rms"] for m in metrics])),
                    "survival_rate": float(np.mean([m["survival_rate"] for m in metrics]))
                }
            }
        
        results[str(noise_std)] = {
            "RL": evaluate_with_noise(env, rl_agent),
            "ES": evaluate_with_noise(env, es_agent)
        }
    
    return results

def main():
    print("=== Segment 2 Corrected Execution (v2) ===")
    print("Using ControlReward (matching training)")
    print("Tracking: Speed, Lat Error RMS, Steering Variance, Time-to-Crash\n")
    
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    
    final_output = {}
    
    # Exp 1: Friction Ladder
    final_output["exp_1"] = exp_1_friction_ladder()
    
    # Exp 2: Noise Gradient
    final_output["exp_2"] = exp_2_noise_gradient()
    
    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(final_output, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✅ Saved Corrected Segment 2 Results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
