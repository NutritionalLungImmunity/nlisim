from collections import OrderedDict
from configparser import ConfigParser
from functools import lru_cache
from importlib import import_module
import logging
from pathlib import PurePath
import re
from typing import List, Type, TYPE_CHECKING, Union

from pkg_resources import iter_entry_points


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

    'modules': ''
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

        self._modules: OrderedDict[str, 'Module'] = OrderedDict()
        for module_path in self.getlist('simulation', 'modules'):
            module = self.load_module(module_path)(self)
            if module.name in self._modules:
                raise ValueError(f'A module named {module.name} already exists')
            self._modules[module.name] = module

    @property
    def modules(self) -> List['Module']:
        return list(self._modules.values())

    @classmethod
    def load_module(cls, path: Union[str, Type['Module']]) -> Type['Module']:
        from simulation.module import Module  # noqa avoid circular imports

        if isinstance(path, str):
            module_path, func_name = path.rsplit('.', 1)
            module = import_module(module_path, 'simulation')
            func = getattr(module, func_name, None)
        else:
            func = path

        if not issubclass(func, Module):
            raise TypeError(f'Invalid module class for "{path}"')
        if not func.name.isidentifier():
            raise ValueError(f'Invalid module name "{func.name}" for "{path}')
        return func

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
