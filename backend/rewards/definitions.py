from abc import ABC, abstractmethod
import numpy as np

class RewardFunction(ABC):
    @abstractmethod
    def compute(self, env_state, action, info) -> float:
        """
        env_state: car state [x, y, heading, speed]
        action: [steering, throttle]
        info: dict with extra info (off_track, etc)
        """
        pass
    
    @abstractmethod
    def reset(self):
        pass

class ProgressReward(RewardFunction):
    def __init__(self, track):
        self.track = track
        self.last_progress = 0.0

    def reset(self):
        self.last_progress = 0.0

    def compute(self, env_state, action, info) -> float:
        # Simple progress based on speed alone is "Broken".
        # Real progress is delta distance along centerline.
        # For simplicity in this v1, let's use:
        # reward = speed * cos(heading_error) - penalty
        # This approximates "velocity along track".
        
        # Unpack
        x, y, h, speed = env_state
        off_track = info.get('off_track', False)
        
        if off_track:
            return -10.0
            
        # Get path progress (this is hard without a spline/path distance func)
        # We can approximate "velocity along track" by projecting velocity onto
        # the tangent of the closest track point.
        
        _, idx, _, _, _ = self.track.get_closest_point_info(x, y)
        
        # Tangent at closest point
        # Simple finite difference from track points
        p_now = self.track.centerline[idx]
        p_next = self.track.centerline[(idx + 1) % len(self.track.centerline)]
        
        tangent = p_next - p_now
        tangent /= (np.linalg.norm(tangent) + 1e-6)
        
        # Car velocity vector
        car_vel = np.array([speed * np.cos(h), speed * np.sin(h)])
        
        # Project car vel onto tangent
        speed_along_track = np.dot(car_vel, tangent)
        
        reward = speed_along_track * 0.1
        
        # Penalty for steering jitter
        reward -= 0.05 * abs(action[0])
        
        return reward

class BrokenReward(RewardFunction):
    def reset(self):
        pass
        
    def compute(self, env_state, action, info) -> float:
        # "Broken" reward: Just run fast, ignore track
        return env_state[3] * 0.1 # Reward speed only
