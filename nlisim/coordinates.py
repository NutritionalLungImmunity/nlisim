import numpy as np


class Coordinate(np.ndarray):
    """A base class for representing 3D vectors as numpy arrays."""

    def __new__(cls, *, x: float = 0.0, y: float = 0.0, z: float = 0.0, **kwargs) -> 'Coordinate':
        return np.asarray([(z, y, x)], dtype=cls.dtype).reshape((3,)).view(cls)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.x}, {self.y}, {self.z})'

    @classmethod
    def from_array(cls, array: np.ndarray) -> 'Coordinate':
        """Generate a `Coordinate` vector from a 3-element array."""
        if array.shape != (3,):
            raise ValueError('Invalid shape for a coordinate object')
        return array.view(cls)

    @property
    def x(self) -> float:
        """Return the `x`-coordinate of the vector."""
        return self[2]

    @x.setter
    def x(self, value: float):
        self[2] = value

    @property
    def y(self) -> float:
        """Return the `y`-coordinate of the vector."""
        return self[1]

    @y.setter
    def y(self, value: float):
        self[1] = value

    @property
    def z(self) -> float:
        """Return the `z`-coordinate of the vector."""
        return self[0]

    @z.setter
    def z(self, value: float):
        self[0] = value

    def norm(self):
        """Return the euclidean norm of the vector."""
        return np.linalg.norm(self, ord=2)


class Point(Coordinate):
    """An array subclass representing a point or vector in 3D space."""

    dtype = np.dtype((np.dtype('<f8'), (3,)))


class Voxel(Coordinate):
    """An array subclass representing the coordinates of a voxel in a 3D array."""

    dtype = np.dtype((np.dtype('i4'), (3,)))

    def __hash__(self):
        return hash(','.join([str(i) for i in self]))

    def __eq__(self, other):
        value = super().__eq__(other)
        if isinstance(value, np.ndarray):
            value = bool(value.all())
        return value

    def __ne__(self, other):
        return not self.__eq__(other)
