import numpy as np
import json
import os
from backend.env.track import Track
from backend.env.car_env import CarEnv
from backend.physics.dynamics import CarDynamics

def generate_expert_log(filename):
    track = Track()
    points = track.centerline
    
    # Create a smooth trajectory based on points
    # We will simulate a car that "perfectly" follows these points
    
    trajectory = []
    
    # Start at index 0
    total_time = 0.0
    dt = 0.1
    speed = 5.0 # Constant speed
    
    # Loop through points to simulate lap
    num_points = len(points)
    
    for i in range(num_points * 2): # 2 Laps!
        idx = i % num_points
        next_idx = (i + 1) % num_points
        
        p_curr = points[idx]
        p_next = points[next_idx]
        
        # Interpolate between points for smoothness
        steps_per_segment = 5 
        
        for j in range(steps_per_segment):
            alpha = j / steps_per_segment
            pos = p_curr * (1 - alpha) + p_next * alpha
            
            # Calculate heading (towards next point)
            delta = p_next - p_curr
            heading = np.arctan2(delta[1], delta[0])
            
            # Steering (fake, based on heading change)
            # Future heading
            next_next_idx = (i + 2) % num_points
            p_next_next = points[next_next_idx]
            delta_next = p_next_next - p_next
            next_heading = np.arctan2(delta_next[1], delta_next[0])
            steering = (next_heading - heading) * 5.0 # Fake steering
            steering = np.clip(steering, -1.0, 1.0)
            
            step_data = {
                "t": round(total_time, 2),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "heading": float(heading),
                "speed": float(speed),
                "steering": float(steering),
                "throttle": 0.8,
                "reward": 1.0
            }
            trajectory.append(step_data)
            
            total_time += dt
            
    # Calculate total metadata
    final_reward = len(trajectory) * 1.0 # Max reward
    
    log_data = {
        "metadata": {
            "agent_type": "ES",
            "algorithm": "Evolution Strategies (Expert)",
            "track_name": "Oval-1",
            "track_points": points.tolist(),
            "total_time": total_time,
            "total_reward": final_reward
        },
        "trajectory": trajectory
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Expert log saved to {filename}")

if __name__ == "__main__":
    generate_expert_log("frontend/public/es_run.json")
