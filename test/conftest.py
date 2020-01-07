from io import BytesIO

from h5py import File
from pytest import fixture

from simulation.config import SimulationConfig
from simulation.coordinates import Point
from simulation.grid import RectangularGrid
from simulation.state import State


@fixture
def base_config():
    yield SimulationConfig()


@fixture
def config():
    yield SimulationConfig(
        defaults={'simulation': {'modules': 'simulation.modules.afumigatus.Afumigatus'}}
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
    yield Point(x=50, y=50, z=50)


@fixture
def hdf5_file():
    yield File(BytesIO(), 'w')


@fixture
def hdf5_group(hdf5_file):
    yield hdf5_file.create_group('test-group')
