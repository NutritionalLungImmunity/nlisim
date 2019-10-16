from pytest import fixture

from simulation.config import SimulationConfig
from simulation.state import State


@fixture
def base_config():
    yield SimulationConfig()


@fixture
def config():
    yield SimulationConfig(defaults={
        'simulation': {'modules': 'simulation.modules.advection.Advection'}
    })


@fixture
def state(config):
    yield State.create(config)
