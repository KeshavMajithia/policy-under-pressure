import sys
import os
import json
import numpy as np

sys.path.append(os.getcwd())
from backend.env.car_env import CarEnv

def generate_sample():
    env = CarEnv()
    obs, info = env.reset()
    
    # Metadata
    log_data = {
        "metadata": {
            "agent_type": "Sample",
            "algorithm": "Random",
            "track_name": "Oval-1",
            "track_points": env.track.centerline.tolist(),
            "total_time": 0.0,
            "total_reward": 0.0
        },
        "trajectory": []
    }
    
    # Run loop
    done = False
    step = 0
    total_reward = 0
    
    # We want a nice looking trajectory, so let's drive straight for a bit then turn
    print("Generating sample trajectory...")
    
    while not done and step < 50:
        # Action: Full throttle, slight steering
        # 0-20 steps: Straight
        # 20-50 steps: Slight left turn
        steering = 0.0
        if step > 20: 
            steering = 0.5
            
        action = np.array([steering, 0.5])
        
        obs, reward, terminated, truncated, run_info = env.step(action)
        total_reward += reward
        
        # Log state
        step_data = {
            "t": step * 0.1,
            "x": float(run_info["x"]),
            "y": float(run_info["y"]),
            "heading": float(run_info["heading"]),
            "speed": float(run_info["speed"]),
            "steering": float(steering),
            "throttle": 0.5,
            "reward": float(reward)
        }
        log_data["trajectory"].append(step_data)
        
        step += 1
        if terminated or truncated:
            done = True

    log_data["metadata"]["total_time"] = float(step * 0.1)
    log_data["metadata"]["total_reward"] = float(total_reward)
    
    # Save
    os.makedirs("logs", exist_ok=True)
    with open("logs/sample_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Sample log saved to logs/sample_log.json with {len(log_data['trajectory'])} points.")

if __name__ == "__main__":
    generate_sample()
