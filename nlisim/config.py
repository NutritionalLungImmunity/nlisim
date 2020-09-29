import logging
import re
from collections import OrderedDict
from configparser import ConfigParser
from importlib import import_module
from io import StringIO, TextIOBase
from pathlib import PurePath
from typing import List, Optional, TextIO, Type, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from nlisim.module import Module  # noqa
    from nlisim.state import State  # noqa

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

    def __init__(self, *config_sources: Union[str, PurePath, TextIO, dict]) -> None:
        super().__init__(allow_no_value=False,
                         inline_comment_prefixes=("#",))
        self._modules: OrderedDict[str, 'Module'] = OrderedDict()

        # set built-in defaults
        self.read_dict({'simulation': DEFAULTS})

        for config_source in config_sources:
            if isinstance(config_source, dict):
                self.read_dict(config_source)
            elif isinstance(config_source, TextIOBase):
                self.read_file(config_source)
            else:
                self.read(config_source)

        for module_path in self.getlist('simulation', 'modules'):
            self.add_module(module_path)

    @property
    def modules(self) -> List['Module']:
        """Return a list of instantiated modules connected to this config."""
        return list(self._modules.values())

    def add_module(self, module_ref: Union[str, Type['Module']]):
        if isinstance(module_ref, str):
            module_func = self.load_module(module_ref)
        else:
            module_func = module_ref
            self.validate_module(module_func)

        module = module_func(self)

        if module.name in self._modules:
            raise ValueError(f'A module named {module.name} already exists')
        self._modules[module.name] = module

    @classmethod
    def load_module(cls, path: str) -> Type['Module']:
        """Load a module class, returning the class constructor."""
        module_path, func_name = path.rsplit('.', 1)
        module = import_module(module_path, 'simulation')
        func = getattr(module, func_name, None)

        cls.validate_module(func, path)

        return func

    @classmethod
    def validate_module(cls, func: Type['Module'], path: Optional[str] = None) -> None:
        """Validate basic aspects of a module class."""
        from nlisim.module import Module  # noqa avoid circular imports

        if path is None:
            path = repr(func)

        if not issubclass(func, Module):
            raise TypeError(f'Invalid module class for "{path}"')
        if not func.name.isidentifier() or func.name.startswith('_'):
            raise ValueError(f'Invalid module name "{func.name}" for "{path}')

    # Wrapper so that this fails when a parameter is missing
    def getint(self, section: str, option: str, *,
               raw: bool = False, vars=None, fallback=None, **kwargs) -> int:
        result = self._get_conv(section, option, int, raw=raw, vars=vars,
                                fallback=fallback, **kwargs)
        assert result is not None, f"Missing parameter {option} in section {section}"
        return result

    # Wrapper so that this fails when a parameter is missing
    def getfloat(self, section, option, *, raw=False, vars=None,
                 fallback=None, **kwargs) -> float:
        result = self._get_conv(section, option, float, raw=raw, vars=vars,
                                fallback=fallback, **kwargs)
        assert result is not None, f"Missing parameter {option} in section {section}"
        return result

    # Wrapper so that this fails when a parameter is missing
    def getboolean(self, section, option, *, raw=False, vars=None,
                   fallback=None, **kwargs) -> bool:
        result = self._get_conv(section, option, self._convert_to_boolean,
                                raw=raw, vars=vars, fallback=fallback, **kwargs)
        assert result is not None, f"Missing parameter {option} in section {section}"
        return result

    # TODO: see if there is a slicker way to do these gettype wrappers.
    # TODO: do checking on 'type-less' get (or implement one for strings)

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
