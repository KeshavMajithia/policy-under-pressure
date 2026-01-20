import gymnasium as gym
from gymnasium import spaces
import numpy as np
from backend.physics.dynamics import CarDynamics
from backend.env.track import Track

class CarEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, reward_type="progress", track_type="oval", reward_fn=None, friction_scale=1.0, reward_delay_steps=0):
        super(CarEnv, self).__init__()
        
        self.dynamics = CarDynamics(dt=0.1, friction_scale=friction_scale)
        self.track = Track(track_type=track_type)
        self.reward_delay_steps = reward_delay_steps
        self.reward_buffer = []
        
        # Initialize Reward Function
        if reward_fn is not None:
             self.reward_fn = reward_fn
        else:
            from backend.rewards.definitions import ProgressReward, BrokenReward
            if reward_type == "broken":
                self.reward_fn = BrokenReward()
            else:
                self.reward_fn = ProgressReward(self.track)
        
        # Action: [steering, throttle]
        # Steering: -1.0 (Left) to 1.0 (Right)
        # Throttle: 0.0 (Coast) to 1.0 (Max Accel)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # New Observation Space (Control Theory focused):
        # 1. Lateral Error (Signed distance from centerline)
        # 2. Heading Error (Relative to track tangent)
        # 3. Speed
        # 4. Curvature (Lookahead)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.pi, 0.0, -np.pi]), 
            high=np.array([np.inf, np.pi, np.inf, np.pi]), 
            dtype=np.float32
        )
        
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Phase 2: Adversarial Starts
        if options and "start_pose" in options:
            start_x, start_y, start_h = options["start_pose"]
        else:
            # Start at the beginning of the track
            start_x, start_y, start_h = self.track.get_start_pose()
        
        self.dynamics.reset(start_x, start_y, start_h)
        self.reward_fn.reset()
        self.current_step = 0
        self.reward_buffer = [0.0] * self.reward_delay_steps
        
        return self._get_obs(), {}

    def regenerate_track(self):
        self.track.regenerate()
        # Reset dynamics to new start pose
        self.reset()

    def step(self, action, config=None):
        """
        config: {
            "friction": float, # For Ice Patch
            "noise": [float, float, ...], # For Foggy Sensor
            "mask": [int, int, ...] # For Blindfold
        }
        """
        self.current_step += 1
        
        # Unpack and Apply
        steering = float(action[0])
        throttle = float(action[1])
        
        # Phase 2: Variable Friction
        friction = config.get("friction", None) if config else None
        
        x, y, h, s = self.dynamics.step(steering, throttle, friction_override=friction)
        
        # Check Constraints
        dist, closest_idx, _, _, _ = self.track.get_closest_point_info(x, y)
        off_track = abs(dist) > (self.track.track_width / 2.0)
        
        # Calculate Progress (Normalized 0-1)
        progress = closest_idx / len(self.track.centerline)
        
        terminated = False
        truncated = False
        
        # Get Obs
        obs = self._get_obs()
        
        # Phase 2: Sensor Corruption (Noise & Masking)
        if config:
            if "noise" in config:
                obs += np.array(config["noise"], dtype=np.float32)
            if "mask" in config:
                for idx in config["mask"]:
                    obs[idx] = 0.0
        
        lat_error = obs[0]
        head_error = obs[1]
        
        info = {
            "x": x,
            "y": y,
            "heading": h,
            "speed": s,
            "off_track": off_track,
            "lateral_error": lat_error,
            "heading_error": head_error,
            "progress": progress
        }

        # Calculate Reward using swappable module
        env_state = [x, y, h, s]
        raw_reward = self.reward_fn.compute(env_state, [steering, throttle], info)
        
        # REWARD DELAY LOGIC (Exp 4)
        if self.reward_delay_steps > 0:
            self.reward_buffer.append(raw_reward)
            if len(self.reward_buffer) > self.reward_delay_steps:
                reward = self.reward_buffer.pop(0)
            else:
                reward = 0.0 # Return 0 until buffer fills
        else:
            reward = raw_reward
        
        # ES-FRIENDLY: Soft crash (slow down instead of terminate)
        if off_track:
            self.dynamics.speed *= 0.2
            
        if self.current_step >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        s = self.dynamics.get_state()
        x, y, h, speed = s
        
        # Get geometric errors from Track
        lat_error, _, _, track_angle, curvature = self.track.get_closest_point_info(x, y)
        
        # Heading error: Agent heading - Track heading
        heading_error = h - track_angle
        # Wrap to [-pi, pi]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([
            lat_error,
            heading_error,
            speed,
            curvature
        ], dtype=np.float32)
