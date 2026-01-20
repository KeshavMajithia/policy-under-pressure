import numpy as np
from backend.rewards.definitions import RewardFunction

class SurvivalReward(RewardFunction):
    """
    Phase 1 Reward v2: MUST complete laps to get reward.
    
    Components:
    - Progress (small): +10 per lap completion
    - Distance: Small reward for moving forward
    - Crash: -10 (terminal)
    
    NO survival bonus (agents were exploiting it by spinning/crawling).
    You MUST complete laps to succeed.
    """
    
    def __init__(self, track):
        self.track = track
        self.prev_progress = 0.0
        self.laps_completed = 0
        
    def reset(self):
        self.prev_progress = 0.0
        self.laps_completed = 0
        
    def compute(self, env_state, action, info):
        x, y, heading, speed = env_state
        reward = 0.0
        
        # 1. Progress reward (moving forward)
        current_progress = info.get('progress', 0.0)
        delta_progress = current_progress - self.prev_progress
        
        # Handle lap wrap-around (LAP COMPLETED!)
        if delta_progress < -0.5:
            delta_progress += 1.0
            self.laps_completed += 1
            reward += 100.0  # HUGE lap completion bonus
        
        # Small reward for forward movement
        reward += delta_progress * 5.0
        self.prev_progress = current_progress
        
        # 2. Speed requirement (must be moving)
        if speed < 0.3:
            reward -= 0.5  # Penalty for standing still/spinning
        
        # 3. Crash penalty (terminal)
        if info.get('off_track', False):
            reward -= 20.0
        
        return reward
