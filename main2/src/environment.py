import numpy as np


class Environment:
    def __init__(self, width=100, height=100, target=(80, 60), seed=None):
        self.width = width
        self.height = height
        self.target = np.array(target, dtype=float)
        self.rng = np.random.default_rng(seed)

    def get_target_direction(self, agent_position, agent_heading):
        direction = self.target - agent_position
        target_angle = np.arctan2(direction[1], direction[0])

        angle_difference = target_angle - agent_heading
        angle_difference = np.arctan2(
            np.sin(angle_difference),
            np.cos(angle_difference)
        )

        center_angle = np.deg2rad(15)

        if angle_difference > center_angle:
            return "LEFT"
        elif angle_difference < -center_angle:
            return "RIGHT"
        return "CENTER"

    def get_boundary_direction(self, position, margin=10):
        x, y = position
        return {
            "left": x < margin,
            "right": x > self.width - margin,
            "bottom": y < margin,
            "top": y > self.height - margin
        }

    def reached_target(self, position, distance=3.0):
        return np.linalg.norm(self.target - position) <= distance

    def randomize_target(self, agent_position, min_distance=25, margin=5):
        """Move the food to a random in-bounds position, away from the agent."""
        for _ in range(1000):
            candidate = np.array([
                self.rng.uniform(margin, self.width - margin),
                self.rng.uniform(margin, self.height - margin)
            ])
            if np.linalg.norm(candidate - agent_position) >= min_distance:
                self.target = candidate
                return self.target.copy()

        # Fallback for unusually small worlds or restrictive settings.
        self.target = np.array([
            self.rng.uniform(margin, self.width - margin),
            self.rng.uniform(margin, self.height - margin)
        ])
        return self.target.copy()

    def keep_inside(self, position):
        position[0] = np.clip(position[0], 0, self.width)
        position[1] = np.clip(position[1], 0, self.height)
        return position
