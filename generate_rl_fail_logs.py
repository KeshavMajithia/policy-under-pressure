
import sys
import os
import argparse
import numpy as np

sys.path.append(os.getcwd())
from backend.env.car_env import CarEnv
from backend.agents.rl import RLAgentFactory
from backend.rewards.control import ControlReward
from generate_log import run_evaluation

def generate_rl_fail_logs():
    model_path = "models/rl_pretrained"
    output_dir = "frontend/public"
    
    # 1. Real Figure Eights
    print("Generating RL Log for Real Figure 8 (Will likely fail)...")
    run_evaluation(
        agent_type="RL", 
        model_path=model_path, 
        output_file="frontend/public/rl_real_fig8.json", 
        algorithm_name="RL (Untrained)", 
        track_type="figure8"
    )

    # 2. Random Tracks (Matching seeds from ES run)
    num_tracks = 3
    for k in range(1, num_tracks + 1):
        filename = f"rl_random_{k}.json"
        full_path = os.path.join(output_dir, filename)
        
        # Reset numpy seed to match ES generation
        np.random.seed(k * 100)
        
        print(f"Generating {filename} for RL...")
        run_evaluation(
            agent_type="RL", 
            model_path=model_path, 
            output_file=full_path, 
            algorithm_name=f"RL (Untrained)", 
            track_type="random"
        )

if __name__ == "__main__":
    generate_rl_fail_logs()
