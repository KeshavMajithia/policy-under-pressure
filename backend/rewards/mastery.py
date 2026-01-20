import numpy as np
from backend.rewards.definitions import RewardFunction

class MasteryReward(RewardFunction):
    """
    Reward function v2 that REQUIRES proper driving:
    1. Must maintain minimum speed (anti-camping)
    2. Stay on track (survival)
    3. Follow centerline (precision)
    4. Make progress (lap completion)
    
    Fixed: Removed exploitable survival bonus.
    """
    
    def __init__(self):
        self.prev_progress = 0.0
        self.min_speed_threshold = 0.5  # Must be moving
        
    def reset(self):
        self.prev_progress = 0.0
        
    def compute(self, env_state, action, info):
        """
        Reward breakdown (v2 - Anti-camping):
        - Speed requirement: Penalty if speed < threshold
        - Centerline adherence: -distance_from_center
        - Progress: +delta_progress (ONLY reward for this)
        - Off-track: -10.0
        
        Args:
            env_state: [x, y, heading, speed]
            action: [steering, throttle]
            info: dict with 'distance_from_center', 'progress', 'off_track', etc.
        """
        reward = 0.0
        
        x, y, heading, speed = env_state
        
        # 1. ANTI-CAMPING: Penalize standing still
        if speed < self.min_speed_threshold:
            reward -= 1.0  # Heavy penalty for not moving
        
        # 2. Centerline adherence (tight driving)
        dist_from_center = info.get('distance_from_center', 0.0)
        reward -= dist_from_center * 0.3  # Light penalty
        
        # 3. Progress reward (PRIMARY reward signal)
        current_progress = info.get('progress', 0.0)
        delta_progress = current_progress - self.prev_progress
        
        # Handle lap wrap-around
        if delta_progress < -0.5:  # Wrapped around (completed lap)
            delta_progress += 1.0
            reward += 50.0  # HUGE lap completion bonus
            
        # Main reward: make progress
        reward += delta_progress * 10.0  # Strong reward for forward movement
        self.prev_progress = current_progress
        
        # 4. Off-track catastrophic penalty
        if info.get('off_track', False):
            reward -= 20.0
            
        return reward
