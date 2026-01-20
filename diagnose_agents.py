import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from backend.env.car_env import CarEnv
from backend.agents.rl import RLAgentFactory
from backend.agents.es import ESAgent
from backend.env.track import Track
from backend.rewards.control import ControlReward

def diagnose_agent(agent_type="RL"):
    """Diagnose agent behavior to find why RL is stationary"""
    
    # Create environment
    track = Track(track_type="figure8")
    env = CarEnv(track_type="figure8", reward_fn=ControlReward(track))
    
    # Load agents
    if agent_type == "RL":
        print("=== Diagnosing RL Agent ===")
        agent = RLAgentFactory.load("models/rl_pretrained.zip", None)
    else:
        print("=== Diagnosing ES Agent ===")
        agent = ESAgent(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim=64)
        agent.load("models/es_pretrained.pkl")
    
    # Run episode
    obs, _ = env.reset(seed=1001)
    
    print(f"\nInitial Observation: {obs}")
    print(f"Obs Shape: {obs.shape}, Dtype: {obs.dtype}")
    print(f"Obs Space: {env.observation_space}")
    
    speeds = []
    actions_steering = []
    actions_throttle = []
    rewards = []
    
    for step in range(100):
        # Get action
        action, _ = agent.predict(obs)
        
        # Record
        actions_steering.append(action[0])
        actions_throttle.append(action[1])
        
        # Step
        obs, reward, term, trunc, info = env.step(action)
        
        speeds.append(info["speed"])
        rewards.append(reward)
        
        if step < 10:
            print(f"\nStep {step}:")
            print(f"  Action: steering={action[0]:.4f}, throttle={action[1]:.4f}")
            print(f"  Speed: {info['speed']:.4f} m/s")
            print(f"  Lat Error: {info['lateral_error']:.4f}")
            print(f"  Heading Error: {info['heading_error']:.4f}")
            print(f"  Reward: {reward:.4f}")
        
        if term or trunc:
            print(f"\nEpisode ended at step {step}")
            break
    
    print(f"\n=== Summary for {agent_type} ===")
    print(f"Mean Speed: {np.mean(speeds):.4f} m/s")
    print(f"Max Speed: {np.max(speeds):.4f} m/s")
    print(f"Mean Steering: {np.mean(actions_steering):.4f}")
    print(f"Steering Variance: {np.var(actions_steering):.4f}")
    print(f"Mean Throttle: {np.mean(actions_throttle):.4f}")
    print(f"Throttle Variance: {np.var(actions_throttle):.4f}")
    print(f"Mean Reward: {np.mean(rewards):.4f}")
    print(f"Total Reward: {np.sum(rewards):.4f}")
    
    return {
        "speeds": speeds,
        "actions_steering": actions_steering,
        "actions_throttle": actions_throttle,
        "rewards": rewards
    }

if __name__ == "__main__":
    print("DIAGNOSTIC TEST: Agent Behavior Analysis\n")
    print("="*60)
    
    rl_data = diagnose_agent("RL")
    print("\n" + "="*60)
    es_data = diagnose_agent("ES")
    
    print("\n" + "="*60)
    print("COMPARISON:")
    print(f"RL Mean Speed: {np.mean(rl_data['speeds']):.4f} m/s")
    print(f"ES Mean Speed: {np.mean(es_data['speeds']):.4f} m/s")
    print(f"\nRL Mean Throttle: {np.mean(rl_data['actions_throttle']):.4f}")
    print(f"ES Mean Throttle: {np.mean(es_data['actions_throttle']):.4f}")
