
import sys
import os
import argparse
import numpy as np

sys.path.append(os.getcwd())
from backend.env.car_env import CarEnv
from backend.agents.es import ESAgent
from backend.rewards.control import ControlReward
from generate_log import run_evaluation

def generate_random_logs():
    model_path = "models/es_pretrained.pkl"
    output_dir = "frontend/public"
    
    # 1. Random Track 1 (Seed 100)
    print("Generating Random Track 1...")
    # Using run_evaluation directly, but we need to force the track generation to be different
    # The current CarEnv uses random.randint/uniform in _generate_random. 
    # run_evaluation resets env with seed=i.
    
    # Actually, simpler to just loop CarEnv here with manual seeds
    num_tracks = 3
    for k in range(1, num_tracks + 1):
        filename = f"es_random_{k}.json"
        full_path = os.path.join(output_dir, filename)
        
        # We need to ensure the TRACK SHAPE is different. _generate_random is called in __init__.
        # So we just init a new env each time.
        # Numpy seed affects _generate_random if we call it before init?
        # Track.__init__ calls _generate_random.
        
        # Reset numpy seed to ensure different track generation
        np.random.seed(k * 100)
        
        print(f"Generating {filename} with random track...")
        run_evaluation(
            agent_type="ES", 
            model_path=model_path, 
            output_file=full_path, 
            algorithm_name=f"ES (Random {k})", 
            track_type="random"
        )

if __name__ == "__main__":
    generate_random_logs()
