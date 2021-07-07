from io import BytesIO

from h5py import File
import numpy as np
from pytest import fixture

from nlisim.config import SimulationConfig
from nlisim.coordinates import Point
from nlisim.grid import RectangularGrid
from nlisim.state import State


@fixture
def base_config():
    yield SimulationConfig()


@fixture
def config():
    yield SimulationConfig(
        {
            'simulation': {
                'modules': 'nlisim.oldmodules.fungus.Fungus',
                'nx': 20,
                'ny': 40,
                'nz': 20,
                'dx': 20,
                'dy': 40,
                'dz': 20,
                'voxel_volume': 6.4e-11,
                'space_volume': 6.4e-11 * 20 * 40 * 20,
                'geometry_path': 'geometry.hdf5',
                'validate': True,
            }
        }
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


@fixture
def epi_geometry():
    # a 10 x 10 x 10 grid with epithelium
    tissue = np.empty((10, 10, 10))
    tissue.fill(3)
    yield tissue


@fixture
def air_geometry():
    # a 10 x 10 x 10 grid with air
    tissue = np.empty((10, 10, 10))
    tissue.fill(0)
    yield tissue
