import numpy as np


class Point(np.ndarray):
    """An array subclass representing a point or vector in 3D space."""
    dtype = np.dtype((np.dtype('<f8'), (3,)))

    def __new__(cls, x: float = 0, y: float = 0, z: float = 0) -> 'Point':
        return np.asarray([z, y, x], dtype=np.float64).view(cls)

    @classmethod
    def from_array(cls, array: np.ndarray) -> 'Point':
        if array.shape != (3,):
            raise ValueError('Invalid shape for point object')
        return array.view(cls)

    @property
    def x(self) -> float:
        return self[2]

    @x.setter
    def x(self, value: float):
        self[2] = value

    @property
    def y(self) -> float:
        return self[1]

    @y.setter
    def y(self, value: float):
        self[1] = value

    @property
    def z(self) -> float:
        return self[0]

    @z.setter
    def z(self, value: float):
        self[0] = value

    def norm(self, ord: float = 2):
        return np.linalg.norm(self, ord=ord)
