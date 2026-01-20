import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv

def test_random_agent():
    print("Initializing Environment...")
    env = CarEnv()
    
    obs, info = env.reset()
    print(f"Initial State: {obs}")
    
    done = False
    step = 0
    total_reward = 0
    
    print("Starting Random Run...")
    while not done:
        # Random action
        action = env.action_space.sample()
        # Bias throttle to actually move forward
        action[1] = 0.5 + 0.5 * np.random.random() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        if step % 20 == 0:
            print(f"Step {step}: Pos({info['x']:.2f}, {info['y']:.2f}) Speed({info['speed']:.2f}) OffTrack({info['off_track']})")
            
        if terminated or truncated:
            done = True
            print(f"Episode Finished. Reason: {'Terminated (Crash)' if terminated else 'Truncated (TimeOut)'}")
            print(f"Final Pos: ({info['x']:.2f}, {info['y']:.2f})")
            print(f"Total Steps: {step}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    test_random_agent()
