import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import gymnasium as gym

@dataclass
class ExperimentConfig:
    """Configuration for a specific research experiment."""
    name: str
    num_episodes: int = 20
    max_steps: int = 1000
    noise_level: float = 0.0
    reward_type: str = "progress"
    track_layout: str = "oval"
    wrappers: List[Any] = field(default_factory=list)
    description: str = ""

@dataclass
class ExperimentResult:
    """Structured results from an experiment."""
    config: ExperimentConfig
    episode_rewards: List[float]
    episode_lengths: List[int]
    success_rate: float
    wall_time: float
    metrics: Dict[str, Any] = field(default_factory=dict) # For custom metrics like 'entropy', 'smoothness'

class ExperimentRunner:
    """
    Standardized runner for Head-to-Head comparisons.
    Executes a specific config for a specific agent.
    """
    def __init__(self, env_factory):
        self.env_factory = env_factory

    def run(self, agent, config: ExperimentConfig) -> ExperimentResult:
        """
        Run the experiment.
        
        Args:
           agent: The agent policy (must have .predict method)
           config: Experiment settings
        """
        env = self.env_factory(config)
        
        # Apply wrappers (e.g., Noise, Delay)
        for wrapper_cls, wrapper_kwargs in config.wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)
            
        rewards = []
        lengths = []
        successes = 0
        
        start_time = time.time()
        
        print(f"Starting Experiment: {config.name} ({config.num_episodes} eps)")
        
        for i in range(config.num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            while not (done or truncated) and steps < config.max_steps:
                action, _ = agent.predict(obs)
                
                # Ensure action format
                if isinstance(action, np.ndarray):
                    action = action.tolist()
                
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
            rewards.append(total_reward)
            lengths.append(steps)
            
            # Simple success metric (e.g. if steps reached near max or track limits)
            # For now, let's assume if they ran > 90% of max steps without dying, it's a "success" 
            # OR if they triggered a specific 'lap_complete' flag (to be added)
            if steps > 50: # Minimal survival threshold
                successes += 1
                
        duration = time.time() - start_time
        
        result = ExperimentResult(
            config=config,
            episode_rewards=rewards,
            episode_lengths=lengths,
            success_rate=successes / config.num_episodes,
            wall_time=duration,
            metrics={
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "mean_length": np.mean(lengths)
            }
        )
        
        return result
