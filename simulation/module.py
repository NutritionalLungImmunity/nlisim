from typing import Dict, Type

import attr
from typing_extensions import Final

from simulation.config import SimulationConfig
from simulation.state import State


@attr.s(auto_attribs=True, kw_only=True)
class ModuleState(object):
    global_state: 'State'


class Module(object):
    #: a unique name for this module used for namespacing
    name: Final[str] = ''

    #: default values for all config options
    defaults: Final[Dict[str, str]] = {}

    #: container for extra state required by this module
    StateClass: Final[Type[ModuleState]] = ModuleState

    def __init__(self, config: SimulationConfig):
        if not config.has_section(self.section):
            config.add_section(self.section)

        self.config = config[self.section]
        self.config.update(self.defaults)

    @property
    def section(self):
        """Return the section in the configuration object used by this module."""
        return self.name

    # The following are no-op hooks that a module can define to customize the
    # behavior.  A module can override any of these methods to execute
    # arbitrary code during the simulation lifetime.
    def construct(self, state: State) -> State:
        """Run after state construction."""
        return state

    def initialize(self, state: State) -> State:
        """Run after state initialization."""
        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Run after advancing the simulation state in time."""
        return state
