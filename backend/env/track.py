import numpy as np

class Track:
    def __init__(self, track_type="oval"):
        self.track_type = track_type
        if track_type == "oval":
            self.centerline = self._generate_oval()
        elif track_type == "figure8":
            self.centerline = self._generate_figure8()
        elif track_type == "random":
            self.centerline = self._generate_random()
        else:
            raise ValueError(f"Unknown track_type: {track_type}")
            
        self.track_width = 8.0 # meters (4.0 either side of center)

    def regenerate(self):
        if self.track_type == "random":
            self.centerline = self._generate_random()
        else:
            print(f"Warning: Cannot regenerate fixed track type {self.track_type}")

    def _generate_oval(self):
        points = []
        # Straight 1 (Bottom): x=0 to x=50, y=0
        xs = np.linspace(0, 50, 50)
        for x in xs: points.append([x, 0.0])
        
        # Turn 1 (Right): Semicircle center=(50, 10), radius=10
        thetas = np.linspace(-np.pi/2, np.pi/2, 30)
        for t in thetas:
            if t == -np.pi/2: continue
            px = 50 + 10 * np.cos(t)
            py = 10 + 10 * np.sin(t)
            points.append([px, py])
            
        # Straight 2 (Top): x=50 to x=0, y=20
        xs = np.linspace(50, 0, 50)
        for x in xs:
            if x == 50: continue 
            points.append([x, 20.0])
            
        # Turn 2 (Left): Semicircle center=(0, 10), radius=10
        thetas = np.linspace(np.pi/2, 3*np.pi/2, 30)
        for t in thetas:
            if t == np.pi/2: continue
            px = 0 + 10 * np.cos(t)
            py = 10 + 10 * np.sin(t)
            points.append([px, py])
            
        return np.array(points, dtype=np.float32)

    def _generate_figure8(self):
        # Parametric Figure 8 (Lemniscate-ish)
        # Using two touching circles for easier driving logic or parametric eq.
        # Let's use parametric for smoothness: 
        # x = a * cos(t) / (1 + sin^2(t))
        # y = a * sin(t) * cos(t) / (1 + sin^2(t))
        
        points = []
        scale = 40.0 
        num_points = 200
        
        # Full loop 0 -> 2pi
        ts = np.linspace(0, 2*np.pi, num_points)
        
        for t in ts:
            denom = 1 + np.sin(t)**2
            x = scale * np.cos(t) / denom
            y = scale * np.sin(t) * np.cos(t) / denom
            
            # Shift to be positive (approx bounds are -scale to +scale)
            # x range: [-40, 40] -> shift by 50 -> [10, 90]
            # y range: [-20, 20] -> shift by 30 -> [10, 50]
            points.append([x + 50.0, y + 30.0])
            
        return np.array(points, dtype=np.float32)

    def _generate_random(self):
        # COMPLEX GENERATION v2: Rugged Terrain
        # 1. Generate random anchor points
        num_anchors = 24  # High frequency
        center = np.array([50.0, 35.0])
        radius_x = 45.0
        radius_y = 28.0
        
        anchors = []
        # sort by angle to ensure a closed loop without self-intersection
        angles = np.linspace(0, 2*np.pi, num_anchors, endpoint=False)
        # Add strong random jitter to angles
        angles += np.random.uniform(-0.15, 0.15, size=num_anchors)
        angles = np.sort(angles)
        
        for theta in angles:
            # Vary radius significantly for "insets" and "outsets"
            # Perlin-noise-ish variation
            r_scale = np.random.uniform(0.4, 1.1)
            x = center[0] + radius_x * r_scale * np.cos(theta)
            y = center[1] + radius_y * r_scale * np.sin(theta)
            anchors.append([x, y])
            
        anchors = np.array(anchors)
        
        # 2. Chaikin Smoothing (Corner Cutting) to make it drivable
        # Run 3 iterations (keeps it organic but drivable)
        points = anchors
        for _ in range(3):
            new_points = []
            num = len(points)
            for i in range(num):
                p0 = points[i]
                p1 = points[(i + 1) % num]
                # Cut at 25% and 75%
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_points.append(q)
                new_points.append(r)
            points = np.array(new_points)
            
        return points.astype(np.float32)

    def get_closest_point_info(self, x, y):
        """
        Returns (distance_to_center, closest_point_index, closest_point_coords, tangent_angle)
        """
        # Vectorized distance to all points
        # shape (N, 2)
        diffs = self.centerline - np.array([x, y], dtype=np.float32)
        dists_sq = np.sum(diffs**2, axis=1)
        closest_idx = np.argmin(dists_sq)
        min_dist = np.sqrt(dists_sq[closest_idx])
        
        # Calculate tangent at this point
        # Use next point to determine direction (cyclic)
        next_idx = (closest_idx + 1) % len(self.centerline)
        p_curr = self.centerline[closest_idx]
        p_next = self.centerline[next_idx]
        
        dx = p_next[0] - p_curr[0]
        dy = p_next[1] - p_curr[1]
        tangent_angle = np.arctan2(dy, dx)
        

        
        # Calculate curvature (change in tangent)
        # Angle difference between this tangent and next tangent
        next_next_idx = (closest_idx + 2) % len(self.centerline)
        p_next_next = self.centerline[next_next_idx]
        dx2 = p_next_next[0] - p_next[0]
        dy2 = p_next_next[1] - p_next[1]
        next_tangent_angle = np.arctan2(dy2, dx2)
        
        curvature = next_tangent_angle - tangent_angle
        # Wrap to [-pi, pi]
        curvature = (curvature + np.pi) % (2 * np.pi) - np.pi

        # Calculate signed distance (Cross product)
        # Vector from track to car
        dx_car = x - p_curr[0]
        dy_car = y - p_curr[1]
        
        # Cross product of tangent and car vector (2D cross output is scalar z)
        # tangent x car_vector = dx*dy_car - dy*dx_car
        # If positive, car is to the left (assuming CCW track).
        # We want simple signed distance.
        cross = dx * dy_car - dy * dx_car
        
        # Sign: Left is positive, Right is negative
        signed_dist = min_dist * np.sign(cross)
        
        return signed_dist, closest_idx, p_curr, tangent_angle, curvature

    def is_off_track(self, x, y):
        dist, _, _, _, _ = self.get_closest_point_info(x, y)
        return abs(dist) > (self.track_width / 2.0)

    def get_start_pose(self):
        # Start at index 0, pointing +x
        p = self.centerline[0]
        # Heading 0.0 (East)
        return p[0], p[1], 0.0
