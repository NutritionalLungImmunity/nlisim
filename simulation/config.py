from collections import OrderedDict
from configparser import ConfigParser
from functools import lru_cache
from importlib import import_module
import logging
from pathlib import PurePath
import re
from typing import Callable, cast, List, TYPE_CHECKING, Union

from pkg_resources import iter_entry_points

from simulation.validator import Validator

if TYPE_CHECKING:
    from simulation.state import State  # noqa


DEFAULTS = {
    'time_step': 0.05,

    'nx': 100,
    'ny': 80,
    'dx': 1.0,
    'dy': 1.0,

    'validate': True,
    'verbosity': logging.WARNING,

    'state_class': 'simulation.state.State',
    'boundary_conditions': 'simulation.boundary.Dirichlet',

    'initialization': 'simulation.contrib.mutation.constant',
    'iteration': 'simulation.contrib.mutation.point_source',
    'validation': ''
}


@lru_cache()
def get_entry_point(name):
    return [
        handler.load() for handler in iter_entry_points(name)
    ]


class SimulationConfig(ConfigParser):
    """
    Internal representation of a simulation config based on the standard python
    ConfigParser class.
    """

    def __init__(self, file: Union[str, PurePath, None] = None) -> None:
        super().__init__()

        # set built-in defaults
        self.read_dict({
            'simulation': DEFAULTS
        })

        # if provided, read the config file
        if file is not None:
            self.read(file)

        # The remaining code dereferences the import paths for the extensions
        # described in the config.  This is done here rather than lazily to ensure
        # import errors occur early.

        # set the class used to store the system state... a downstream
        # plugin could append new variables to the state, but the mechanics
        # of this are not fully developed.
        self.StateClass = self.load_import_path(
            self.get('simulation', 'state_class')
        )

        # import the boundary condition class in use
        self.boundary_conditions = self.load_import_path(
            self.get('simulation', 'boundary_conditions')
        )

        # load all of functions that run during initialization
        self.initialization_plugins = OrderedDict([
            (p, self.load_import_path(p)) for p in self.getlist('simulation', 'initialization')
        ])

        # load all of the function that run during solver iterations
        self.iteration_plugins = OrderedDict([
            (p, self.load_import_path(p)) for p in self.getlist('simulation', 'iteration')
        ])

        # generate the state validator class
        validators = [
            self.load_import_path(p) for p in self.getlist('simulation', 'validation')
        ]
        self.validate = Validator(
            validators, skip=not self.getboolean('simulation', 'validate', fallback=True)
        )

    @classmethod
    def load_import_path(cls, path: Union[str, Callable]) -> Callable:
        """Import a module path to a callable function.

        This is a generic utility method used by the configuration to load extensions
        to the solution.  For example, calling with ``my_package.module_1.module_2.method``
        would return the result of `from my_package.module_1.module_2 import method``.
        """
        if not isinstance(path, str):
            return path

        module_path, func_name = path.rsplit('.', 1)
        module = import_module(module_path, 'simulation')
        func = getattr(module, func_name, None)
        if not callable(func):
            raise Exception(f'{path} is not a callable object')
        return cast(Callable, func)

    def getlist(self, section: str, option: str) -> List[str]:
        """Return a list of strings from a configuration value.

        This method reads a string value from the configuration object and splits
        it by: new lines, spaces, and commas.  The values in the returned list are
        stripped of all white space and removed if empty.

        For example, the following values all parse as ``['a', 'b', 'c']``:

        value = a,b,c
        value = a b c
        value = a, b, c
        value = a
                b
                c
        value = a,
                b,
                c
        value = a b
                ,c
        """
        values = []
        for value in re.split('[\n ,]+', self.get(section, option, fallback='')):
            stripped = value.strip()
            if stripped:
                values.append(stripped)
        return values
