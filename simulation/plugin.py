from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.config import SimulationConfig  # noqa
    from simulation.state import State  # noqa


class IterationHandler(object):
    def __init__(self, config: 'SimulationConfig') -> None:
        pass

    def __call__(self, state: 'State') -> 'State':
        raise NotImplementedError('IterationHandler must implement __call__')
