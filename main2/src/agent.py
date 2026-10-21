import numpy as np


class Agent:
    def __init__(self, network, start=(20, 50), heading=0.0):
        self.network = network

        self.position = np.array(start, dtype=float)
        self.heading = heading

        self.speed = 0.25 #OLD VAL: 0.4
        self.turn_speed = 0.04 # OLD VAL: 0.02

    def get_direction(self):

        left = []
        right = []

        for neuron_type in ["PFL1", "PFL2", "PFL3"]:
            left.append(
                self.network.get_side_mean_activity(
                    neuron_type, "L"
                )
            )

            right.append(
                self.network.get_side_mean_activity(
                    neuron_type, "R"
                )
            )

        left_activity = np.mean(left)
        right_activity = np.mean(right)

        difference = left_activity - right_activity

        threshold = 0.002

        if difference > threshold:
            return 1
        elif difference < -threshold:
            return -1

        return 0            # STRAIGHT

    def avoid_boundary(self, environment):
        margin = 8

        x, y = self.position

        if x < margin:
            self.heading = 0

        elif x > environment.width - margin:
            self.heading = np.pi

        elif y < margin:
            self.heading = np.pi / 2

        elif y > environment.height - margin:
            self.heading = -np.pi / 2

    def update(self, turn=None):
        if turn is not None:
            self.heading += self.turn_speed * turn

        dx = np.cos(self.heading) * self.speed
        dy = np.sin(self.heading) * self.speed

        self.position += np.array([dx, dy])

    def reset(self, position=(20, 50), heading=0.0):
        self.position = np.array(position, dtype=float)
        self.heading = heading