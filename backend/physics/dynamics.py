import numpy as np

class CarDynamics:
    def __init__(self, dt=0.1, friction_scale=1.0):
        self.dt = dt
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.speed = 0.0
        
        # Physics Constants
        self.MAX_SPEED = 20.0
        self.MAX_STEERING_ANGLE = 1.0  # Max radians to turn per second approx
        self.ACCELERATION = 5.0
        self.FRICTION = 1.0 * friction_scale # Drag coefficient (Scaled)

    def reset(self, x=0.0, y=0.0, heading=0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = 0.0
        return self.get_state()

    def step(self, steering, throttle, friction_override=None):
        """
        Update the car state based on actions.
        """
        dt = self.dt
        friction = friction_override if friction_override is not None else self.FRICTION
        
        self.x, self.y, self.heading, self.speed = self._calculate_next_state(
            self.x, self.y, self.heading, self.speed, steering, throttle, dt, friction
        )
        return self.get_state()

    def peek_step(self, steering, throttle):
        """
        Predicts the next state without updating the internal state.
        """
        return self._calculate_next_state(
            self.x, self.y, self.heading, self.speed, steering, throttle, self.dt, self.FRICTION
        )

    def _calculate_next_state(self, x, y, h, s, steering, throttle, dt, friction):
        # Clip actions
        steering = np.clip(steering, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)

        # Update Speed
        accel = throttle * self.ACCELERATION
        new_speed = s + (accel - friction * s) * dt
        new_speed = np.clip(new_speed, 0.0, self.MAX_SPEED)

        # Update Heading
        turn_rate = steering * self.MAX_STEERING_ANGLE
        new_heading = h + turn_rate * dt
        new_heading = (new_heading + np.pi) % (2 * np.pi) - np.pi

        # Update Position
        new_x = x + new_speed * np.cos(new_heading) * dt
        new_y = y + new_speed * np.sin(new_heading) * dt

        return new_x, new_y, new_heading, new_speed

    def get_state(self):
        return np.array([self.x, self.y, self.heading, self.speed], dtype=np.float32)
