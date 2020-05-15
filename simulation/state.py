from io import BytesIO, StringIO
from pathlib import PurePath
from typing import Any, cast, Dict, IO, Type, TYPE_CHECKING, Union

import attr
from h5py import File as H5File
import numpy as np

from simulation.grid import RectangularGrid
from simulation.validation import context as validation_context

if TYPE_CHECKING:  # prevent circular imports for type checking
    from simulation.cell import CellList
    from simulation.config import SimulationConfig  # noqa
    from simulation.module import ModuleState  # noqa

_dtype_float = np.dtype('float')
_dtype_float64 = np.dtype('float64')


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

            with StringIO(hf.attrs['config']) as cf:
                config = SimulationConfig(cf)

            state = cls(time=time, grid=grid, config=config)

            for module in config.modules:
                group = hf.get(module.name)
                if group is None:
                    raise ValueError(f'File contains no group for {module.name}')
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
            config.getint('simulation', 'nx'),
        )
        spacing = (
            config.getfloat('simulation', 'dz'),
            config.getfloat('simulation', 'dy'),
            config.getfloat('simulation', 'dx'),
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


def grid_variable(dtype: np.dtype = _dtype_float) -> np.ndarray:
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
        if value.dtype.names:
            for name in value.dtype.names:
                if not np.isfinite(value[name]).all():
                    raise ValidationError(f'Invalid value in gridded variable {attribute.name}')
        else:
            if not np.isfinite(value).all():
                raise ValidationError(f'Invalid value in gridded variable {attribute.name}')

    metadata = {'grid': True}
    return attr.ib(
        default=attr.Factory(factory, takes_self=True),
        validator=validate_numeric,
        metadata=metadata,
    )


def cell_list(list_class: Type['CellList']) -> 'CellList':
    def factory(self: 'ModuleState'):
        return list_class(grid=self.global_state.grid)

    metadata = {'cell_list': True}
    return attr.ib(default=attr.Factory(factory, takes_self=True), metadata=metadata)


def get_class_path(instance: Any) -> str:
    class_ = instance.__class__
    return f'{class_.__module__}:{class_.__name__}'
