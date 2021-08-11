from importlib import import_module
from typing import Any, Dict, Optional, Tuple, Type, Union

import attr
from h5py import Dataset, Group
import numpy as np

from nlisim.cell import CellList
from nlisim.config import SimulationConfig
from nlisim.state import State

AttrValue = Union[float, str, bool, np.ndarray]


@attr.s(auto_attribs=True, kw_only=True)
class ModuleState(object):
    """
    Base type intended to store the state for simulation modules.

    This class contains serialization support for basic types (float, int, str,
    bool) and numpy arrays of those types.  Modules containing more complicated
    state must override the serialization mechanism with custom behavior.
    """

    global_state: 'State'

    def save_state(self, group: Group) -> None:
        """Save the module state into an HDF5 group."""
        for field in attr.fields(type(self)):
            name = field.name
            if name == 'global_state':
                continue
            value = getattr(self, name)
            self.save_attribute(group, name, value, field.metadata)

    @classmethod
    def load_state(cls, global_state: 'State', group: Group) -> 'ModuleState':
        """Load this module's state from an HDF5 group."""
        kwargs: Dict[str, Any] = {'global_state': global_state}
        for field in attr.fields(cls):
            name = field.name
            if name == 'global_state':
                continue
            metadata = field.metadata or {}

            group_object = group.get(name, None)
            if group_object is None:
                raise ValueError(f'Could not read {name} from file.')

            if isinstance(group_object, Group):
                # TODO: break this out into helper methods so subclasses can customize
                class_name = group_object.attrs.get('class')
                if class_name is None:
                    raise ValueError(f'Field {name} contained an invalid composite type.')

                module_name, class_name = class_name.split(':')
                try:
                    module = import_module(module_name)
                except ImportError:
                    raise TypeError(f'File references unknown module {module_name} in {name}')

                class_ = getattr(module, class_name, None)
                if class_ is None:
                    raise TypeError(f'File references invalid class for {name}')

                kwargs[name] = class_.load(global_state, group, name, metadata)

            else:
                kwargs[name] = cls.load_attribute(global_state, group, name, metadata)
        return cls(**kwargs)

    @classmethod
    def save_attribute(
        cls, group: Group, name: str, value: AttrValue, metadata: dict
    ) -> Union[Dataset, Group]:
        """Save an attribute into an HDF5 group."""
        metadata = metadata or {}
        if isinstance(value, (float, int, str, bool, np.ndarray)):
            return cls.save_simple_type(group, name, value, metadata)
        elif hasattr(value, 'save'):
            return value.save(group, name, metadata)
        else:
            # modules must define serializers for unsupported types
            raise TypeError(f'Attribute {name} in {group.name} contains an unsupported datatype.')

    @classmethod
    def save_simple_type(cls, group: Group, name: str, value: AttrValue, metadata: dict) -> Dataset:
        kwargs: Dict[str, Any] = {}
        if metadata.get('grid'):
            kwargs = dict(
                compression='gzip',  # transparent compression
                shuffle=True,  # improve compressiblity
                fletcher32=True,  # checksum
            )

        var = group.create_dataset(name=name, data=np.asarray(value), **kwargs)

        # mark the original attribute as a scalar to unwrap the numpy array when loading
        var.attrs['scalar'] = not isinstance(value, np.ndarray)

        if metadata.get('grid'):
            # attach grid scales to dataset dimensions
            var.dims[0].attach_scale(var.file['z'])
            var.dims[1].attach_scale(var.file['y'])
            var.dims[2].attach_scale(var.file['x'])

        return var

    @classmethod
    def load_attribute(
        cls, global_state: State, group: Group, name: str, metadata: Optional[dict] = None
    ) -> AttrValue:
        """Load a raw value from an HDF5 file group."""
        dataset = group[name]
        if dataset.attrs.get('scalar', False):
            value = dataset[()]
        else:
            value = dataset[:]
        return value


class ModuleModel(object):
    name: str = ''
    """A unique name for this module used for namespacing"""

    StateClass: Type[ModuleState] = ModuleState
    """Container for extra state required by this module."""

    def __init__(self, config: SimulationConfig):
        if not config.has_section(self.section):
            config.add_section(self.section)

        self.config = config[self.section]
        values = dict(**self.config)
        self.config.update(values)

    @property
    def time_step(self) -> float:
        """Return the period over which the module is updated."""
        assert 'time_step' in self.config.keys(), (
            f'Module {self.name} has no time step configured! '
            f'Configuration file must have a time_step field under section [{self.name}] '
            'Enter zero or negative values for modules which do not update.'
        )
        return float(self.config['time_step'])

    @property
    def section(self):
        """Return the section in the configuration object used by this module."""
        return self.name

    # The following are no-op hooks that a module can define to customize the
    # behavior.  A module can override any of these methods to execute
    # arbitrary code during the simulation lifetime.
    def construct(self, state: State) -> None:
        """Run after state construction."""

    def initialize(self, state: State) -> State:
        """Run after state initialization."""
        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Run after advancing the simulation state in time."""
        return state

    def finalize(self, state: State) -> State:
        """Run after the last time step."""
        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        """Run to provide informative statistics based on the module's current state.

        Note: Json encoder does not support numpy data-types (e.g. np.float32) so it
        is important to cast these values to standard python types.
        """
        return dict()

    def visualization_data(self, state: State) -> Tuple[str, Optional[Union[CellList, np.ndarray]]]:
        """Return any 3d visualization data associated with the module.

        This is returned as a tuple of the form (datatype, data) where datatype should be one of
        'molecule', 'cells', or ''. The last, empty string, should be reported by modules which do
        not report data.
        """
        return '', None
