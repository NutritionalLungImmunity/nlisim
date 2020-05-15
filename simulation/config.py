from collections import OrderedDict
from configparser import ConfigParser
from importlib import import_module
from io import StringIO
import logging
from pathlib import PurePath
import re
from typing import List, Type, TYPE_CHECKING, Union


if TYPE_CHECKING:
    from simulation.module import Module  # noqa
    from simulation.state import State  # noqa


DEFAULTS = {
    'time_step': 0.05,
    'nx': 100,
    'ny': 80,
    'nz': 60,
    'dx': 1.0,
    'dy': 1.0,
    'dz': 1.0,
    'validate': True,
    'verbosity': logging.WARNING,
    'modules': '',
}


class SimulationConfig(ConfigParser):
    """
    Internal representation of a simulation config.

    This is based on the standard Python ConfigParser class.
    """

    def __init__(self, *config_sources: Union[str, PurePath, dict]) -> None:
        super().__init__()

        # set built-in defaults
        self.read_dict({'simulation': DEFAULTS})

        for config_source in config_sources:
            if isinstance(config_source, dict):
                self.read_dict(config_source)
            else:
                self.read(config_source)

        self._modules: OrderedDict[str, 'Module'] = OrderedDict()
        for module_path in self.getlist('simulation', 'modules'):
            module = self.load_module(module_path)(self)
            if module.name in self._modules:
                raise ValueError(f'A module named {module.name} already exists')
            self._modules[module.name] = module

    @property
    def modules(self) -> List['Module']:
        """Return a list of instantiated modules connected to this config."""
        return list(self._modules.values())

    @classmethod
    def load_module(cls, path: Union[str, Type['Module']]) -> Type['Module']:
        """Load and validate a module class returning the class constructor."""
        from simulation.module import Module  # noqa avoid circular imports

        if isinstance(path, str):
            module_path, func_name = path.rsplit('.', 1)
            module = import_module(module_path, 'simulation')
            func = getattr(module, func_name, None)
        else:
            func = path

        if not issubclass(func, Module):
            raise TypeError(f'Invalid module class for "{path}"')
        if not func.name.isidentifier() or func.name.startswith('_'):
            raise ValueError(f'Invalid module name "{func.name}" for "{path}')
        return func

    def getlist(self, section: str, option: str) -> List[str]:
        """
        Return a list of strings from a configuration value.

        This method reads a string value from the configuration object and splits
        it by: new lines, spaces, and commas.  The values in the returned list are
        stripped of all white space and removed if empty.

        For example, the following values all parse as <code>`['a', 'b', 'c']`</code>:

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
        return self.parselist(self.get(section, option, fallback=''))

    @classmethod
    def parselist(cls, value: str) -> List[str]:
        """
        Return a list of strings from a configuration value.

        This is a helper method for `getlist`, split out for code sharing.
        """
        values = []
        for value in re.split('[\n ,]+', value):
            stripped = value.strip()
            if stripped:
                values.append(stripped)
        return values

    def __str__(self):
        f = StringIO()
        self.write(f)
        return f.getvalue()
