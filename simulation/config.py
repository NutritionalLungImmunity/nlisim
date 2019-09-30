from configparser import ConfigParser
from functools import lru_cache
from importlib import import_module
import logging
from pathlib import PurePath
from pkg_resources import iter_entry_points
import re
from typing import Callable, cast, List, TYPE_CHECKING, Union

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

    'initialization': 'simulation.contrib.constant',
    'iteration': 'simulation.contrib.point_source'
}


@lru_cache()
def get_entry_point(name):
    return [
        handler.load() for handler in iter_entry_points(name)
    ]


class SimulationConfig(ConfigParser):
    def __init__(self, file: Union[str, PurePath]='config.ini') -> None:
        super().__init__()

        # set defaults
        self.read_dict({
            'simulation': DEFAULTS
        })
        self.read(file)

        self.StateClass = self.load_import_path(
            self.get('simulation', 'state_class')
        )
        self.boundary_conditions = self.load_import_path(
            self.get('simulation', 'boundary_conditions')
        )

        self.initialization_plugins = [
            self.load_import_path(p) for p in self.getlist('simulation', 'initialization')
        ]
        self.iteration_plugins = [
            self.load_import_path(p) for p in self.getlist('simulation', 'iteration')
        ]

    @classmethod
    def load_import_path(cls, path: Union[str, Callable]) -> Callable:
        if not isinstance(path, str):
            return path

        module_path, func_name = path.rsplit('.', 1)
        module = import_module(module_path, 'simulation')
        func = getattr(module, func_name, None)
        if not callable(func):
            raise Exception(f'{path} is not a callable object')
        return cast(Callable, func)

    def getlist(self, section: str, option: str) -> List[str]:
        values = []
        for value in re.split('[\n ,]+', self.get(section, option, fallback='')):
            stripped = value.strip()
            if stripped:
                values.append(stripped)
        return values
