from enum import IntEnum
import random

import attr
import numpy as np

from simulation.cell import CellData, CellList
from simulation.coordinates import Point
from simulation.grid import RectangularGrid
from simulation.module import Module, ModuleState
from simulation.modules.geometry import GeometryState, TissueTypes
from simulation.state import State


class MacrophageCellData(CellData):
    BOOLEAN_NETWORK_LENGTH = 3  # place holder for now

    class Status(IntEnum):
        RESTING = 0
        ACTIVE = 1
        INTERACTING = 2
        SECRETING = 3
        SYNERGIC = 4
        APOPTOTIC = 5
        NECROTIC = 6
        DEAD = 7

    MACROPHAGE_FIELDS = [
        ('boolean_network', 'b1', BOOLEAN_NETWORK_LENGTH),
        ('status', 'u1'),
        ('iron_pool', 'f8'),
        ('iteration', 'i4'),
    ]

    dtype = np.dtype(CellData.FIELDS + MACROPHAGE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls, *, iron_pool: float = 0, status: Status = Status.RESTING, **kwargs,
    ) -> np.record:

        network = cls.initial_boolean_network()
        iteration = 0

        return CellData.create_cell_tuple(**kwargs) + (network, status, iron_pool, iteration,)

    @classmethod
    def initial_boolean_network(cls) -> np.ndarray:
        return np.asarray([True, False, True, ])


@attr.s(kw_only=True, frozen=True, repr=False)
class MacrophageCellList(CellList):
    CellDataClass = MacrophageCellData

    def is_moveable(self, grid: RectangularGrid):
        cells = self.cell_data
        return self.alive(
            (cells['status'] == MacrophageCellData.Status.RESTING)
            & cells.point_mask(cells['point'], grid)
        )


def cell_list_factory(self: 'MacrophageState'):
    return MacrophageCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class MacrophageState(ModuleState):
    cells: MacrophageCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    init_num: int


class Macrophage(Module):
    name = 'macrophage'
    defaults = {
        'cells': '',
        'init_num': '0',
    }
    StateClass = MacrophageState

    def initialize(self, state: State):
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid
        tissue: GeometryState = state.geometry.lung_tissue

        macrophage.init_num = self.config.getint('init_num')
        macrophage.cells = MacrophageCellList(grid=grid)

        if macrophage.init_num > 0:
            # initialize the surfactant layer with some macrophage in random locations
            indices = np.argwhere(tissue == TissueTypes.SURFACTANT.value)
            np.random.shuffle(indices)

            for i in range(0, macrophage.init_num):
                x = indices[i][2] + (random.uniform(0, 1))
                y = indices[i][1] + (random.uniform(0, 1))
                z = indices[i][0] + (random.uniform(0, 1))

                point = Point(x=x, y=y, z=z)
                status = MacrophageCellData.Status.RESTING

                macrophage.cells.append(MacrophageCellData.create_cell(point=point, status=status))

        return state

    def advance(self, state: State, previous_time: float):
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid
        tissue: GeometryState = state.geometry.lung_tissue

        drift(macrophage.cells, tissue, grid)
        # TODO - add recruitment
        # indices = np.argwhere(molecule_to_recruit >= threshold_value)
        # then for each index create a cell with prob 'rec_rate'
        # TODO - add leaving
        # indices = np.argwhere(molecule_to_leave <= threshold_value)
        # then for each index kill a cell with prob 'leave_rate'
        # TODO - add boolena network update
        # for index in cells.alive:
        #   cells[index].update_boolean network
        # TODO - add interaction?

        return state


def drift(cells: MacrophageCellList, tissue: GeometryState, grid: RectangularGrid):
    # currently randomly drifts resting macrophages, need to add molecule dependence
    def valid_point(point: Point, grid: RectangularGrid):
        return (
            (grid.xv[0] <= point.x)
            & (point.x <= grid.xv[-1])
            & (grid.yv[0] <= point.y)
            & (point.y <= grid.yv[-1])
            & (grid.zv[0] <= point.z)
            & (point.z <= grid.zv[-1])
        )

    # TODO - replace random 'delta_point' with logic based on chemokines
    delta_point = Point(x=random.uniform(-1, 1), y=random.uniform(-1, 1), z=random.uniform(-1, 1))
    # TODO - use a more numpy solution to move more efficiently
    for index in cells.is_moveable(grid):
        cell = cells[index]
        new_point = cell['point'] + delta_point
        vox = grid.get_voxel(new_point)
        while not (
            valid_point(new_point, grid)
            and tissue[vox.z, vox.y, vox.x] == TissueTypes.SURFACTANT.value
        ):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            new_point = cell['point'] + Point(x=x, y=y, z=z)
            vox = grid.get_voxel(new_point)

        cell['point'] = new_point
        cells.update_voxel_index([index])

    return cells
