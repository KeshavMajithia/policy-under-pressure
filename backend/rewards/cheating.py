from backend.rewards.definitions import RewardFunction
import numpy as np

class SpeedDemonReward(RewardFunction):
    """
    Experiment 1: The Speed Demon
    Task: Reward pure speed. ZERO penalty for going off-track or crashing.
    Hypothesis: Agent becomes a missile (max throttle, minimal steering).
    """
    def __init__(self):
        pass

    def reset(self):
        pass

    def compute(self, env_state, action, info) -> float:
        x, y, h, speed = env_state
        # Pure Speed Reward
        # No safety checks.
        reward = speed * 2.0 
        return reward

class ParkerReward(RewardFunction):
    """
    Experiment 3: The Parker
    Task: Reward alignment with track heading.
    Hypothesis: Agent learns to stop (v=0) and just turn wheels to align, or drive very slowly to maintain alignment.
    Proxy Metric: cos(heading_error).
    True Goal: Lap Completion (which this reward doesn't explicitly encourage).
    """
    def __init__(self):
        pass

    def reset(self):
        pass

    def compute(self, env_state, action, info) -> float:
        # info["heading_error"] is in radians
        heading_error = info.get("heading_error", 0.0)
        # Cosine similarity: 1.0 if perfectly aligned, -1.0 if reversed
        alignment = np.cos(heading_error)
        
        # We reward alignment. We do NOT reward speed or progress.
        return alignment

class SensitivityReward(RewardFunction):
    """
    Experiment 2: The Butterfly Effect
    Task: Standard progress reward, but the 'Safety Penalty' is randomized PER EPISODE (or per Training Run).
    For training stability test, we fix it at initialization.
    """
    def __init__(self, safety_coeff=1.0):
        self.safety_coeff = safety_coeff

    def reset(self):
        pass

    def compute(self, env_state, action, info) -> float:
        # Standard PPO Reward structure from Phase 1
        # R = Progress - SafetyPenalty
        progress = info.get("progress_bonus", 0.0)
        
        # Calculate Safety Penalty (Lateral Error)
        # Using the standard logic: >1.0m is bad.
        lat_error = abs(info.get("lateral_error", 0.0))
        
        # Standard was 1.0 * lat_error. Now we scale it.
        penalty = self.safety_coeff * lat_error
        
        # Crash penalty (standard is usually big, but let's keep it simple)
        crash_penalty = 100.0 if info.get("crashed", False) else 0.0
        
        reward = (info.get("speed", 0.0) / 10.0) - penalty - crash_penalty
        return reward
