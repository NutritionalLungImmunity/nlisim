import numpy as np

from nlisim.coordinates import Point


class Sphere:
    def __init__(self, center: np.ndarray, radius: float, type: str):
        self.center = Point(x=center[0], y=center[1], z=center[2])
        self.radius = radius
        self.type = type

    def scale(self, n):
        self.radius *= n
        self.center.x *= n
        self.center.y *= n
        self.center.z *= n


class Cylinder:
    def __init__(
        self, center: np.ndarray, direction: np.ndarray, radius: float, length: float, type: str
    ):
        self.center = Point(x=center[0], y=center[1], z=center[2])
        self.direction = np.asarray(direction)
        self.radius = radius
        self.length = length
        self.type = type

    def scale(self, n):
        self.radius *= n
        self.center.x *= n
        self.center.y *= n
        self.center.z *= n
        self.length *= n
