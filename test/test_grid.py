import pytest

from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid


def v(x: int, y: int, z: int) -> Voxel:
    return Voxel(x=x, y=y, z=z)


def p(x: float, y: float, z: float) -> Point:
    return Point(x=x, y=y, z=z)


@pytest.fixture
def grid():
    shape = [10, 20, 30]
    spacing = [1, 1, 1]
    yield RectangularGrid.construct_uniform(shape=shape, spacing=spacing)


@pytest.mark.parametrize(
    'point,voxel',
    [
        (p(0.5, 0.5, 0.5), v(0, 0, 0)),
        (p(1.5, 1.5, 1.5), v(1, 1, 1)),
        (p(0.9, 0.9, 0.9), v(0, 0, 0)),
        (p(1.9, 1.1, 0.1), v(1, 1, 0)),
        (p(-0.1, -0.4, 0.4), v(-1, -1, 0)),
    ],
)
def test_get_voxel(grid: RectangularGrid, point, voxel):
    assert grid.get_voxel(point) == voxel


@pytest.mark.parametrize(
    'voxel,valid',
    [
        (v(0, 0, 0), True),
        (v(9, 0, 4), True),
        (v(-1, 0, 0), False),
        (v(25, 0, 0), True),
        (v(0, 0, 25), False),
        (v(-1, -1, 100), False),
    ],
)
def test_valid_voxel(grid: RectangularGrid, voxel, valid):
    assert grid.is_valid_voxel(voxel) == valid


# fmt: off
@pytest.mark.parametrize(
    'voxel,neighbors',
    [
        (
            v(5, 5, 5),
            {
                v(4, 5, 5),
                v(6, 5, 5),
                v(5, 4, 5),
                v(5, 6, 5),
                v(5, 5, 4),
                v(5, 5, 6),
            }
        ),
        (
            v(0, 5, 5),
            {
                v(1, 5, 5),
                v(0, 4, 5),
                v(0, 6, 5),
                v(0, 5, 4),
                v(0, 5, 6),
            }
        ),
        (
            v(0, 0, 5),
            {
                v(1, 0, 5),
                v(0, 1, 5),
                v(0, 0, 4),
                v(0, 0, 6),
            }
        ),
        (
            v(0, 0, 0),
            {
                v(1, 0, 0),
                v(0, 1, 0),
                v(0, 0, 1),
            }
        )
    ]
)
# fmt: on
def test_get_adjacent_voxels(grid: RectangularGrid, voxel, neighbors):
    assert set(grid.get_adjacent_voxels(voxel)) == neighbors


# fmt: off
@pytest.mark.parametrize(
    'voxel,neighbors',
    [
        (
            v(5, 5, 5),
            26
        ),
        (
            v(0, 5, 5),
            17
        ),
        (
            v(0, 0, 5),
            11
        ),
        (
            v(0, 0, 0),
            7
        )
    ]
)
# fmt: on
def test_get_corner_adjacent_voxels(grid: RectangularGrid, voxel, neighbors):
    assert len(list(grid.get_adjacent_voxels(voxel, corners=True))) == neighbors


@pytest.mark.parametrize(
    'point,in_domain',
    [
        (p(0.1, 0.1, 0.1), True),
        (p(0, 0, 0), True),
        (p(-1, 0, 0), False),
        (p(100, 0, 0), False),
        (p(100, 10, -100), False),
    ],
)
def test_point_in_domain(grid, point, in_domain):
    assert grid.is_point_in_domain(point) == in_domain


@pytest.mark.parametrize(
    'point,voxel',
    [
        (p(0.5, 0.5, 0.5), v(0, 0, 0)),
        (p(1.5, 0.5, 0.5), v(1, 0, 0)),
        (p(1.1, 5.5, 1.9), v(1, 5, 1)),
    ],
)
def test_get_nearest_voxel(grid, point, voxel):
    assert grid.get_nearest_voxel(point) == voxel


# fmt: off
@pytest.mark.parametrize(
    'point,distance,voxels',
    [
        (
            p(5.5, 5.5, 5.5),
            1,
            {
                (v(5, 5, 5), 0),
                (v(4, 5, 5), 1),
                (v(6, 5, 5), 1),
                (v(5, 4, 5), 1),
                (v(5, 6, 5), 1),
                (v(5, 5, 4), 1),
                (v(5, 5, 6), 1),
            }
        ),
        (
            p(0.5, 5.5, 5.5),
            1,
            {
                (v(0, 5, 5), 0),
                (v(1, 5, 5), 1),
                (v(0, 4, 5), 1),
                (v(0, 6, 5), 1),
                (v(0, 5, 4), 1),
                (v(0, 5, 6), 1),
            }
        ),
        (
            p(0.5, 0.5, 5.5),
            1,
            {
                (v(0, 0, 5), 0),
                (v(1, 0, 5), 1),
                (v(0, 1, 5), 1),
                (v(0, 0, 4), 1),
                (v(0, 0, 6), 1),
            }
        ),
        (
            p(0.5, 0.5, 0.5),
            1,
            {
                (v(0, 0, 0), 0),
                (v(1, 0, 0), 1),
                (v(0, 1, 0), 1),
                (v(0, 0, 1), 1),
            }
        )
    ]
)
# fmt: on
def test_get_voxels_in_range(grid: RectangularGrid, point, distance, voxels):
    assert set(grid.get_voxels_in_range(point, distance)) == voxels


def test_get_voxels_in_large_range(grid: RectangularGrid):
    for voxel, distance in grid.get_voxels_in_range(Point(x=4, y=4, z=4), 10):
        assert distance <= 10
        assert grid.is_valid_voxel(voxel)


@pytest.mark.parametrize(
    'voxel,index',
    [(v(0, 0, 0), 0), (v(1, 0, 0), 1), (v(0, 1, 0), 30), (v(0, 0, 1), 600), (v(1, 1, 1), 631)],
)
def test_get_flattened_index(voxel, index, grid: RectangularGrid):
    assert grid.get_flattened_index(voxel) == index
    assert grid.voxel_from_flattened_index(index) == voxel
