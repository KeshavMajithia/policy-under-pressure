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
    """Diagnose agent behavior with CORRECT Phase 1 models"""
    
    # Create environment
    track = Track(track_type="figure8")
    env = CarEnv(track_type="figure8", reward_fn=ControlReward(track))
    
    # Load CORRECT Phase 1 agents
    if agent_type == "RL":
        print("=== Diagnosing RL Agent (rl_phase2_fixed) ===")
        agent = RLAgentFactory.load("models/rl_phase2_fixed", env)
    else:
        print("=== Diagnosing ES Agent (es_phase2_strict) ===")
        agent = ESAgent(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim=64)
        agent.load("models/es_phase2_strict.pkl")
    
    # Run episode
    obs, _ = env.reset(seed=1001)
    
    speeds = []
    actions_throttle = []
    rewards = []
    
    for step in range(100):
        action, _ = agent.predict(obs)
        actions_throttle.append(action[1])
        
        obs, reward, term, trunc, info = env.step(action)
        speeds.append(info["speed"])
        rewards.append(reward)
        
        if step < 5:
            print(f"Step {step}: throttle={action[1]:.4f}, speed={info['speed']:.4f} m/s")
        
        if term or trunc:
            break
    
    print(f"\n=== Summary for {agent_type} ===")
    print(f"Mean Speed: {np.mean(speeds):.4f} m/s")
    print(f"Mean Throttle: {np.mean(actions_throttle):.4f}")
    print(f"Mean Reward: {np.mean(rewards):.4f}")
    
    return {"speeds": speeds, "throttle": actions_throttle}

if __name__ == "__main__":
    print("DIAGNOSTIC TEST with CORRECT Phase 1 Models\n")
    print("="*60)
    
    rl_data = diagnose_agent("RL")
    print("\n" + "="*60)
    es_data = diagnose_agent("ES")
    
    print("\n" + "="*60)
    print("COMPARISON:")
    print(f"RL Mean Speed: {np.mean(rl_data['speeds']):.4f} m/s")
    print(f"ES Mean Speed: {np.mean(es_data['speeds']):.4f} m/s")
    print(f"RL Mean Throttle: {np.mean(rl_data['throttle']):.4f}")
    print(f"ES Mean Throttle: {np.mean(es_data['throttle']):.4f}")
