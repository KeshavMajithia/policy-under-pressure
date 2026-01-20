import gymnasium as gym
import numpy as np

class NoiseWrapper(gym.Wrapper):
    """
    Injects Gaussian noise into observations or actions.
    Used for Robustness testing (Dims 3 & 4).
    """
    def __init__(self, env, obs_noise_std=0.0, action_noise_std=0.0):
        super().__init__(env)
        self.obs_noise_std = obs_noise_std
        self.action_noise_std = action_noise_std
        
    def step(self, action):
        # Inject Action Noise (Simulating actuator failure/jitter)
        if self.action_noise_std > 0:
            noise = np.random.normal(0, self.action_noise_std, size=len(action))
            # Assuming action is list or array. Gym actions usually are.
            action = np.array(action) + noise
            # Clip to valid range if known (assuming -1 to 1 for basic continuous)
            action = np.clip(action, -1.0, 1.0)
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Inject Observation Noise (Simulating sensor failure/LIDAR noise)
        if self.obs_noise_std > 0:
            noise = np.random.normal(0, self.obs_noise_std, size=obs.shape)
            obs = obs + noise
            
        return obs, reward, terminated, truncated, info

class DelayWrapper(gym.Wrapper):
    """
    Delays reward feedback.
    Used for "Robustness to Reward Delay" testing.
    """
    def __init__(self, env, delay_steps=0):
        super().__init__(env)
        self.delay_steps = delay_steps
        self.reward_buffer = []

    def reset(self, **kwargs):
        self.reward_buffer = [0.0] * self.delay_steps
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.reward_buffer.append(reward)
        delayed_reward = self.reward_buffer.pop(0)
        
        return obs, delayed_reward, terminated, truncated, info
