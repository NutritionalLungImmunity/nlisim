from pytest import fixture

from simulation.config import SimulationConfig
from simulation.coordinates import Point
from simulation.state import RectangularGrid, State


@fixture
def base_config():
    yield SimulationConfig()


@fixture
def config():
    yield SimulationConfig(
        defaults={'simulation': {'modules': 'simulation.modules.advection.Advection'}}
    )


@fixture
def state(config):
    yield State.create(config)


@fixture
def grid():
    # a 100 x 100 x 100 unit grid
    yield RectangularGrid.construct_uniform((10, 10, 10), (10, 10, 10))


@fixture
def point():
    yield Point(50, 50, 50)
