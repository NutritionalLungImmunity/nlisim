from importlib import import_module
from typing import Any, cast, Dict, Optional, Type, Union

import attr
from h5py import Dataset, Group
import numpy as np
from scipy.sparse import coo_matrix

from simulation.cell import CellTree, MAX_CELL_TREE_SIZE
from simulation.config import SimulationConfig
from simulation.state import State

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
            metadata = field.metadata
            if metadata.get('cell_tree'):
                kwargs[name] = cls.load_cell_tree(global_state, group, name, metadata)
            else:
                kwargs[name] = cls.load_attribute(global_state, group, name, metadata)
        return cls(**kwargs)

    @classmethod
    def save_attribute(cls, group: Group,
                       name: str, value: AttrValue,
                       metadata: dict = None) -> Union[Dataset, Group]:
        """Save an attribute into an HDF5 group."""

        metadata = metadata or {}
        if isinstance(value, (float, int, str, bool, np.ndarray)):
            return cls.save_simple_type(group, name, value, metadata)
        elif isinstance(value, CellTree):
            return cls.save_cell_tree(group, name, value, metadata)
        else:
            # modules must define serializers for unsupported types
            raise TypeError(
                f'Attribute {name} in {group.name} contains an unsupported datatype.'
            )

    @classmethod
    def save_simple_type(cls, group: Group,
                         name: str, value: AttrValue,
                         metadata: dict) -> Dataset:
        kwargs: Dict[str, Any] = {}
        if metadata.get('grid'):
            kwargs = dict(
               compression='gzip',  # transparent compression
               shuffle=True,        # improve compressiblity
               fletcher32=True      # checksum
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
    def save_cell_tree(cls, group: Group,
                       name: str, value: CellTree,
                       metadata: dict) -> Group:
        composite_group = group.create_group(name)

        class_ = value.__class__
        composite_group.attrs['type'] = 'CellTree'
        composite_group.attrs['class'] = f'{class_.__module__}:{class_.__name__}'
        composite_group.attrs['max_size'] = value.max_cells

        composite_group.create_dataset(name='cells', data=value.cells)

        sp = value.adjacency.tocoo()
        composite_group.create_dataset(name='row', data=sp.row)
        composite_group.create_dataset(name='col', data=sp.col)
        composite_group.create_dataset(name='value', data=sp.data)
        return composite_group

    @classmethod
    def load_attribute(cls, global_state: State, group: Group, name: str,
                       metadata: Optional[dict] = None) -> AttrValue:
        """Load a raw value from an HDF5 file group."""
        dataset = group[name]
        if dataset.attrs.get('scalar', False):
            value = dataset[()]
        else:
            value = dataset[:]
        return value

    @classmethod
    def load_cell_tree(cls, global_state: State, group: Group, name: str,
                       metadata: Optional[dict] = None) -> CellTree:
        composite_dataset = group[name]

        attrs = composite_dataset.attrs
        if attrs.get('type') != 'CellTree' or attrs.get('class') is None:
            raise TypeError(f'File contains invalid type for {name}')

        module_name, class_name = attrs['class'].split(':')
        try:
            module = import_module(module_name)
        except ImportError:
            raise TypeError(f'File references unknown module {module_name} in {name}')

        class_ = getattr(module, class_name, None)
        if class_ is None:
            raise TypeError(f'File references invalid class for {name}')

        class_ = cast(Type[CellTree], class_)
        max_size = attrs.get('max_size', MAX_CELL_TREE_SIZE)

        adjacency = coo_matrix((
            composite_dataset['value'],
            (composite_dataset['row'], composite_dataset['col'])),
            shape=(max_size, max_size)
        ).todok()
        cells = composite_dataset['cells'][:].view(class_.CellArrayClass)

        return class_(grid=global_state.grid, cells=cells, adjacency=adjacency)


class Module(object):
    #: a unique name for this module used for namespacing
    name: str = ''

    #: default values for all config options
    defaults: Dict[str, str] = {}

    #: container for extra state required by this module
    StateClass: Type[ModuleState] = ModuleState

    def __init__(self, config: SimulationConfig):
        if not config.has_section(self.section):
            config.add_section(self.section)

        self.config = config[self.section]
        values = dict(self.defaults, **self.config)
        self.config.update(values)

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
