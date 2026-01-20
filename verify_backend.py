import sys
import os
import numpy as np
import argparse

sys.path.append(os.getcwd())
from backend.env.car_env import CarEnv
from backend.agents.rl import RLAgentFactory
from backend.agents.es import ESAgent

def evaluate_agent(agent_type, model_path, episodes=5):
    print(f"\n--- Evaluating {agent_type} Agent ---")
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}")
        return

    env = CarEnv(reward_type="progress")
    
    # Load
    agent = None
    if agent_type == "RL":
        agent = RLAgentFactory.load(model_path, env)
    elif agent_type == "ES":
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.shape[0]
        agent = ESAgent(input_dim, output_dim)
        agent.load(model_path)

    rewards = []
    steps = []
    
    for i in range(episodes):
        obs, _ = env.reset(seed=42 + i) # Fixed seed for fairness
        done = False
        total_rew = 0
        step = 0
        
        while not done:
            action, _ = agent.predict(obs)
            if isinstance(action, np.ndarray): action = action.tolist()
            
            obs, rew, term, trunc, info = env.step(action)
            total_rew += rew
            step += 1
            if term or trunc:
                done = True
        
        rewards.append(total_rew)
        steps.append(step)
        print(f"Ep {i+1}: Reward={total_rew:.2f}, Steps={step}, FinalX={info['x']:.1f}")

    print(f"{agent_type} Results: Mean Reward = {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Avg Duration: {np.mean(steps):.1f} steps")

if __name__ == "__main__":
    evaluate_agent("RL", "models/ppo_car")
    evaluate_agent("ES", "models/es_car.pkl")
