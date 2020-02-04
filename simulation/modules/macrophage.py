import random

import attr
import numpy as np

from simulation.cell import CellData
from simulation.coordinates import Point
from simulation.grid import RectangularGrid
from simulation.module import Module, ModuleState
from simulation.modules.geometry import TissueTypes
from simulation.modules.phagocyte import PhagocyteCellData, PhagocyteCellList
from simulation.state import State


class MacrophageCellData(PhagocyteCellData):
    BOOLEAN_NETWORK_LENGTH = 3  # place holder for now
    MACROPHAGE_FIELDS = [
        ('boolean_network', 'b1', BOOLEAN_NETWORK_LENGTH),
    ]

    dtype = np.dtype(
        CellData.FIELDS + PhagocyteCellData.PHAGOCYTE_FIELDS + MACROPHAGE_FIELDS, align=True
    )  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs,) -> np.record:
        network = cls.initial_boolean_network()
        return PhagocyteCellData.create_cell_tuple(**kwargs) + (network,)

    @classmethod
    def initial_boolean_network(cls) -> np.ndarray:
        return np.asarray([True, False, True])


@attr.s(kw_only=True, frozen=True, repr=False)
class MacrophageCellList(PhagocyteCellList):
    CellDataClass = MacrophageCellData

    def update(self, tissue, grid: RectangularGrid):
        # TODO - add boolena network update
        # for index in self.alive:
        #   self[index].update_boolean network
        return


def cell_list_factory(self: 'MacrophageState'):
    return MacrophageCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class MacrophageState(ModuleState):
    cells: MacrophageCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    init_num: int = 0
    MPH_UPTAKE_QTTY: float = 0.1
    TF_ENHANCE: float = 1
    DRIFT_LAMBDA: float = 10
    DRIFT_BIAS: float = 0.9


class Macrophage(Module):
    name = 'macrophage'
    defaults = {
        'cells': '',
        'init_num': '0',
        'MPH_UPTAKE_QTTY': '0.1',
        'TF_ENHANCE': '1',
        'DRIFT_LAMBDA': '10',
        'DRIFT_BIAS': '0.9',
    }
    StateClass = MacrophageState

    def initialize(self, state: State):
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        macrophage.init_num = self.config.getint('init_num')
        macrophage.MPH_UPTAKE_QTTY = self.config.getfloat('MPH_UPTAKE_QTTY')
        macrophage.TF_ENHANCE = self.config.getfloat('TF_ENHANCE')
        macrophage.DRIFT_LAMBDA = self.config.getfloat('DRIFT_LAMBDA')
        macrophage.DRIFT_BIAS = self.config.getfloat('DRIFT_BIAS')
        MacrophageCellData.LEAVE_RATE = self.config.getfloat('LEAVE_RATE')
        MacrophageCellData.RECRUIT_RATE = self.config.getfloat('RECRUIT_RATE')
        macrophage.cells = MacrophageCellList(grid=grid)

        if macrophage.init_num > 0:
            # initialize the surfactant layer with some macrophage in random locations
            indices = np.argwhere(tissue == TissueTypes.SURFACTANT.value)
            np.random.shuffle(indices)

            for i in range(0, macrophage.init_num):
                x = random.uniform(grid.xv[indices[i][2]], grid.xv[indices[i][2] + 1])
                y = random.uniform(grid.yv[indices[i][1]], grid.yv[indices[i][1] + 1])
                z = random.uniform(grid.zv[indices[i][0]], grid.zv[indices[i][0] + 1])

                point = Point(x=x, y=y, z=z)
                status = MacrophageCellData.Status.RESTING

                macrophage.cells.append(MacrophageCellData.create_cell(point=point, status=status))

        return state

    def advance(self, state: State, previous_time: float):
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue
        cells = macrophage.cells
        iron = state.molecules.grid.concentrations.iron
        # drift(macrophage.cells, tissue, grid)
        interact(state)

        cells.recruit(MacrophageCellData.RECRUIT_RATE, tissue, grid)

        cells.remove(MacrophageCellData.LEAVE_RATE, tissue, grid)

        cells.update(tissue, grid)

        cells.chemotaxis(
            iron, macrophage.DRIFT_LAMBDA, macrophage.DRIFT_BIAS, tissue, grid,
        )

        return state


def hill_probability(substract, km=10):
    return substract * substract / (substract * substract + km * km)


def interact(state: State):
    # get molecules in voxel
    iron = state.molecules.grid.concentrations.iron
    cells = state.macrophage.cells
    grid = state.grid

    uptake = state.macrophage.MPH_UPTAKE_QTTY
    enhance = state.macrophage.TF_ENHANCE

    # 1. Get cells that are alive
    for index in cells.alive():

        # 2. Get voxel for each cell to get agents in that voxel
        cell = cells[index]
        vox = grid.get_voxel(cell['point'])

        # 3. Interact with all molecules

        #  Iron -----------------------------------------------------
        iron_amount = iron[vox.z, vox.y, vox.x]
        if hill_probability(iron_amount) > random.random():
            qtty = uptake * iron_amount
            iron[vox.z, vox.y, vox.x] -= qtty
            cell['iron_pool'] += qtty
            # print(qtty)
            # print(iron[vox.z, vox.y, vox.x])
            # print(cell['iron_pool'])

        fpn = 0  # TODO replace with actual boolean network
        tf = 1  # TODO replace with actual boolean network
        cell['boolean_network'][fpn] = True
        cell['boolean_network'][tf] = False

        if cell['boolean_network'][fpn]:
            enhancer = enhance if cell['boolean_network'][tf] else 1
            qtty = cell['iron_pool'] * (1 if uptake * enhancer > 1 else uptake * enhancer)
            iron[vox.z, vox.y, vox.x] += qtty
            cell['iron_pool'] -= qtty
            # print(qtty)
            # print(iron[vox.z, vox.y, vox.x])
            # print(cell['iron_pool'])

        #  Next_Mol -----------------------------------------------------
        #    next_mol_amount = iron[vox.z, vox.y, vox.x] ...
