import numpy as np
from backend.rewards.definitions import RewardFunction

class DrivingReward(RewardFunction):
    """
    Ultra-simple reward: Just drive on the track.
    
    Reward = Speed * (1 - distance_penalty)
    
    Translation: Go fast when you're on the centerline, slow down when drifting.
    No tricks, no exploits, just pure driving behavior.
    """
    
    def __init__(self, track):
        self.track = track
        self.track_width = 4.0  # From track definition
        
    def reset(self):
        pass  # No state needed
        
    def compute(self, env_state, action, info):
        x, y, heading, speed = env_state
        
        # Get distance from centerline
        dist_from_center = info.get('distance_from_center', 0.0)
        
        # Normalize distance (0 = perfect, 1 = at edge)
        normalized_dist = min(dist_from_center / (self.track_width / 2.0), 1.0)
        
        # Centerline factor: 1.0 at center, 0.0 at edge
        centerline_factor = 1.0 - normalized_dist
        
        # Reward = Speed when on centerline
        # If you're at center going 5 m/s: reward = 5.0
        # If you're at edge going 5 m/s: reward = 0.0
        reward = speed * centerline_factor
        
        # Off-track termination (handled by env, but penalize here too)
        if info.get('off_track', False):
            reward = -10.0
            
        return reward
