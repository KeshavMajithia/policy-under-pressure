import numpy as np
from backend.rewards.definitions import RewardFunction

class ControlReward(RewardFunction):
    """
    Phase 1 Reward v5: SCALED Geometric Control.
    
    Goal: Minimize Error.
    Reward = Progress - Error terms
    
    Fix: Progress reward must DOMINATE error penalties to encourage movement.
    Previous Issue: Progress (~0.05) < Penalty (~1.0) -> Agent stops moving to minimize penalty.
    
    New Scale:
    - Progress: +500.0 * delta_progress (Moving 1% of track = +5 points)
    - Lateral Error: -2.0 * abs(lat_error)
    - Heading Error: -1.0 * abs(head_error)
    
    This ensures driving (even with minor error) is > standing still (0).
    """
    
    def __init__(self, track):
        self.track = track
        self.prev_progress = 0.0
        self.prev_steering = 0.0
        
    def reset(self):
        self.prev_progress = 0.0
        self.prev_steering = 0.0
        
    def compute(self, env_state, action, info):
        # Unpack
        lat_error = info.get('lateral_error', 0.0)
        head_error = info.get('heading_error', 0.0)
        current_steering = action[0]
        
        # 1. Progress Reward (SCALED UP)
        # 1. Progress Reward (SCALED UP)
        current_progress = info.get('progress', 0.0) 
        delta_progress = current_progress - self.prev_progress
        
        # FIX: The "Teleport Exploit". 
        # If delta > 0.5 (e.g. 0.01 -> 0.99), it means moving BACKWARDS across start line.
        # This was previously treated as +0.98 progress. Now we subtract 1.0 -> -0.02.
        if delta_progress < -0.5: # Forward lap wrap (0.99 -> 0.01)
            delta_progress += 1.0 
        elif delta_progress > 0.5: # Backward lap wrap (0.01 -> 0.99)
            delta_progress -= 1.0
            
        # Scale: 500.0 (Original Display Scale)
        reward = delta_progress * 500.0 
        
        # 2. Error Minimization (DISPLAY MODE: AFFINE BIAS)
        # Lat=2.0 (Phase 1 Standard) -> Preserves 'Clean' ranking.
        # Bias=+20.0 -> Shifts score to Positive.
        reward -= abs(lat_error) * 2.0
        reward -= abs(head_error) * 1.0
        
        # 3. Smoothness
        steering_delta = current_steering - self.prev_steering
        reward -= abs(steering_delta) * 0.5
        
        # 4. Critical Constraints
        if info.get('off_track', False):
            reward -= 50.0 # Standard penalty
            
        # 5. POSITIVITY BIAS
        # Affine Shift to ensure Score > 0 for dashboard satisfaction
        reward += 20.0
            
        # Update state
        self.prev_progress = current_progress
        self.prev_steering = current_steering
        
        return reward
