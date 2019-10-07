from pytest import fixture

from simulation.config import SimulationConfig
from simulation.initialization import create_state


@fixture
def config():
    return SimulationConfig()


@fixture
def state(config):
    return create_state(config)
