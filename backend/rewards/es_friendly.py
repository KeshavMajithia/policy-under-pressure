import numpy as np
from backend.rewards.definitions import RewardFunction

class ESFriendlyReward(RewardFunction):
    """
    Phase 1 Reward v3: MUST stay on track to get reward.
    
    Key insight: Reward = Progress ONLY when on centerline
    
    - On centerline + moving forward = HIGH reward
    - Off centerline = ZERO reward (but don't crash, keep trying)
    - Lap completion = BIG bonus
    
    This forces agents to learn: "I must stay on the road to progress"
    """
    
    def __init__(self, track):
        self.track = track
        self.prev_progress = 0.0
        self.track_width = 4.0
        
    def reset(self):
        self.prev_progress = 0.0
        
    def compute(self, env_state, action, info):
        x, y, heading, speed = env_state
        reward = 0.0
        
        # Get track info
        dist_from_center = info.get('progress', 0.0)
        current_progress = info.get('progress', 0.0)
        
        # Calculate progress delta
        delta_progress = current_progress - self.prev_progress
        
        # Handle lap wrap
        if delta_progress < -0.5:
            delta_progress += 1.0
            reward += 200.0  # HUGE lap bonus
        
        # CORE REWARD: Progress weighted by centerline proximity
        # On centerline (dist < 1.0): Full progress reward
        # Off track (dist > 2.0): Zero progress reward
        
        dist_from_center = info.get('distance_from_center', 0.0)
        
        # Centerline factor: 1.0 at center, 0.0 at edge
        # Smooth falloff using sigmoid
        centerline_factor = max(0.0, 1.0 - (dist_from_center / 2.0))
        
        # Reward = progress * how_well_centered
        # If on centerline making progress: +5 per progress unit
        # If off centerline making progress: +0
        reward += delta_progress * 50.0 * centerline_factor
        
        # Small speed bonus (only when on track)
        if dist_from_center < 2.0:
            reward += speed * 0.1 * centerline_factor
        
        self.prev_progress = current_progress
        
        return reward
