from functools import reduce
from io import BytesIO
from pathlib import PurePath
from tempfile import NamedTemporaryFile
from typing import Any, cast, Dict, IO, List, Tuple, TYPE_CHECKING, Union

import attr
from h5py import File as H5File
import numpy as np

from simulation.validation import context as validation_context

if TYPE_CHECKING:  # prevent circular imports for type checking
    from simulation.config import SimulationConfig  # noqa
    from simulation.module import ModuleState  # noqa

ShapeType = Tuple[int, int, int]
SpacingType = Tuple[float, float, float]


@attr.s(auto_attribs=True, repr=False)
class RectangularGrid(object):
    """A class representation of a rectangular grid."""
    # cell centered coordinates (1-d arrays)
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    # vertex coordinates (1-d arrays)
    xv: np.ndarray
    yv: np.ndarray
    zv: np.ndarray

    @classmethod
    def _make_coordinate_arrays(cls, size: int, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
        vertex = np.arange(size + 1) * spacing
        cell = spacing / 2 + vertex[:-1]
        vertex.flags['WRITEABLE'] = False
        cell.flags['WRITEABLE'] = False
        return cell, vertex

    @classmethod
    def construct_uniform(cls, shape: ShapeType, spacing: SpacingType) -> 'RectangularGrid':
        """Create a rectangular grid with uniform spacing in each axis."""
        nz, ny, nx = shape
        dz, dy, dx = spacing
        x, xv = cls._make_coordinate_arrays(nx, dx)
        y, yv = cls._make_coordinate_arrays(ny, dy)
        z, zv = cls._make_coordinate_arrays(nz, dz)
        return cls(x=x, y=y, z=z, xv=xv, yv=yv, zv=zv)

    @property
    def meshgrid(self) -> List[np.ndarray]:
        """Return the coordinate grid representation.

        This returns three 3D arrays containing the z, y, x coordinates
        respectively.  For example,

        >>> Z, Y, X = grid.meshgrid()

        X[zi, yi, xi] is is the x-coordinate of the point at indices (xi, yi,
        zi).  The data returned is a read-only view into the coordinate arrays
        and is efficient to compute on demand.
        """
        return np.meshgrid(self.z, self.y, self.x, indexing='ij', copy=False)

    def delta(self, axis: int) -> np.ndarray:
        """Return grid spacing along the given axis."""
        if axis == 0:
            meshgrid = np.meshgrid(self.zv, self.y, self.x, indexing='ij', copy=False)[axis]
        elif axis == 1:
            meshgrid = np.meshgrid(self.z, self.yv, self.x, indexing='ij', copy=False)[axis]
        elif axis == 2:
            meshgrid = np.meshgrid(self.z, self.y, self.xv, indexing='ij', copy=False)[axis]
        else:
            raise ValueError('Invalid axis provided')

        return np.diff(meshgrid, axis=axis)

    @property
    def shape(self) -> ShapeType:
        return (len(self.z), len(self.y), len(self.x))

    def __len__(self):
        return reduce(lambda x, y: x * y, self.shape, 1)

    def allocate_variable(self, dtype: np.dtype = np.dtype('float64')) -> np.ndarray:
        """Allocate a numpy array defined over this grid."""
        return np.zeros(self.shape, dtype=dtype)

    def __repr__(self):
        shp = self.shape
        return f'RectangularGrid(nx={shp[2]}, ny={shp[1]}, nz={shp[0]})'

    def save(self, file: H5File) -> None:
        """Save the grid state into an HDF5 file."""
        for dim in ('x', 'xv', 'y', 'yv', 'z', 'zv'):
            d = file.create_dataset(dim, data=getattr(self, dim))
            d.make_scale(dim)

    @classmethod
    def load(cls, file: H5File) -> 'RectangularGrid':
        """Generate a grid object from an existing HDF5 file."""
        kwargs = {}
        for dim in ('x', 'xv', 'y', 'yv', 'z', 'zv'):
            kwargs[dim] = file[dim][:]
        return cls(**kwargs)


@attr.s(auto_attribs=True, repr=False)
class State(object):
    """A container for storing the simulation state at a single time step."""
    time: float
    grid: RectangularGrid

    # simulation configuration
    config: 'SimulationConfig'

    # a private container for module state, users of this class should use the
    # public API instead
    _extra: Dict[str, 'ModuleState'] = attr.ib(factory=dict)

    @classmethod
    def load(cls, arg: Union[str, bytes, PurePath, IO[bytes]]) -> 'State':
        """Load a pickled state from either a path, a file, or blob of bytes."""
        from simulation.config import SimulationConfig  # prevent circular imports

        if isinstance(arg, bytes):
            arg = BytesIO(arg)

        with H5File(arg, 'r') as hf:
            time = hf.attrs['time']
            grid = RectangularGrid.load(hf)

            with NamedTemporaryFile('r+') as cf:
                cf.write(hf.attrs['config'])
                cf.flush()
                cf.seek(0)
                config = SimulationConfig(file=cf.name)

            state = cls(time=time, grid=grid, config=config)

            for module in config.modules:
                group = hf.get(module.name)
                if group is None:
                    raise ValueError(
                        f'File contains no group for {module.name}'
                    )
                try:
                    module_state = module.StateClass.load_state(state, group)
                except Exception:
                    print(f'Error loading state for {module.name}')
                    raise

                state._extra[module.name] = module_state

        return state

    def save(self, arg: Union[str, PurePath, IO[bytes]]) -> None:
        """Save the current state to the file system."""
        with H5File(arg, 'w') as hf:
            hf.attrs['time'] = self.time
            hf.attrs['config'] = str(self.config)  # TODO: save this in a different format
            self.grid.save(hf)

            for module in self.config.modules:
                module_state = cast('ModuleState', getattr(self, module.name))
                group = hf.create_group(module.name)
                try:
                    module_state.save_state(group)
                except Exception:
                    print(f'Error serializing {module.name}')
                    raise

    def serialize(self) -> bytes:
        """Return a serialized representation of the current state."""
        f = BytesIO()
        self.save(f)
        return f.getvalue()

    @classmethod
    def create(cls, config: 'SimulationConfig'):
        """Generate a new state object from a config."""
        shape = (
            config.getint('simulation', 'nz'),
            config.getint('simulation', 'ny'),
            config.getint('simulation', 'nx')
        )
        spacing = (
            config.getfloat('simulation', 'dz'),
            config.getfloat('simulation', 'dy'),
            config.getfloat('simulation', 'dx')
        )
        grid = RectangularGrid.construct_uniform(shape, spacing)
        state = State(time=0.0, grid=grid, config=config)

        for module in state.config.modules:
            if hasattr(state, module.name):
                # prevent modules from overriding existing class attributes
                raise ValueError(f'The name "{module.name}" is a reserved token.')

            with validation_context(f'{module.name} (construction)'):
                state._extra[module.name] = module.StateClass(global_state=state)
                module.construct(state)

        return state

    def __repr__(self):
        modules = [m.name for m in self.config.modules]
        return f'State(time={self.time}, grid={repr(self.grid)}, modules={modules})'

    # expose module state as attributes on the global state object
    def __getattr__(self, module_name: str) -> Any:
        if module_name != '_extra' and module_name in self._extra:
            return self._extra[module_name]
        return super().__getattribute__(module_name)

    def __dir__(self):
        return sorted(super().__dir__() + list(self._extra.keys()))


def grid_variable(dtype: np.dtype = np.dtype('float')) -> np.ndarray:
    """Return an "attr.ib" object defining a gridded state variable.

    A "gridded" variable is one that is discretized on the primary grid.  The
    attribute returned by this method contains a factory function for
    initialization and a default validation that checks for NaN's.
    """
    from simulation.module import ModuleState  # noqa prevent circular imports
    from simulation.validation import ValidationError  # prevent circular imports

    def factory(self: 'ModuleState') -> np.ndarray:
        return self.global_state.grid.allocate_variable(dtype)

    def validate_numeric(self: 'ModuleState', attribute: attr.Attribute, value: np.ndarray) -> None:
        grid = self.global_state.grid
        if value.shape != grid.shape:
            raise ValidationError(f'Invalid shape for gridded variable {attribute.name}')
        if not np.isfinite(value).all():
            raise ValidationError(f'Invalid value in gridded variable {attribute.name}')

    metadata = {
        'grid': True
    }
    return attr.ib(default=attr.Factory(factory, takes_self=True),
                   validator=validate_numeric, metadata=metadata)
