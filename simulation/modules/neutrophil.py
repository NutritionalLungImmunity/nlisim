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


class NeutrophilCellData(PhagocyteCellData):
    NEUTROPHIL_FIELDS = [
        ('name', 'U10'),
    ]
    dtype = np.dtype(
        CellData.FIELDS + PhagocyteCellData.PHAGOCYTE_FIELDS + NEUTROPHIL_FIELDS, align=True
    )  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs,) -> np.record:
        name = 'neutrophil'
        return PhagocyteCellData.create_cell_tuple(**kwargs) + (name,)


@attr.s(kw_only=True, frozen=True, repr=False)
class NeutrophilCellList(PhagocyteCellList):
    CellDataClass = NeutrophilCellData


def cell_list_factory(self: 'NeutrophilState'):
    return NeutrophilCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class NeutrophilState(ModuleState):
    cells: NeutrophilCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    init_num: int = 0
    DRIFT_LAMBDA: float = 10
    DRIFT_BIAS: float = 0.9


class Neutrophil(Module):
    name = 'neutrophil'
    defaults = {
        'cells': '',
        'init_num': '0',
        'DRIFT_LAMBDA': '10',
        'DRIFT_BIAS': '0.9',
    }
    StateClass = NeutrophilState

    def initialize(self, state: State):
        neutrophil: NeutrophilState = state.neutrophil
        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        neutrophil.init_num = self.config.getint('init_num')
        neutrophil.DRIFT_LAMBDA = self.config.getfloat('DRIFT_LAMBDA')
        neutrophil.DRIFT_BIAS = self.config.getfloat('DRIFT_BIAS')
        NeutrophilCellData.RECRUIT_RATE = self.config.getfloat('RECRUIT_RATE')
        NeutrophilCellData.LEAVE_RATE = self.config.getfloat('LEAVE_RATE')

        neutrophil.cells = NeutrophilCellList(grid=grid)

        if neutrophil.init_num > 0:
            # initialize the surfactant layer with some neutrophil in random locations
            indices = np.argwhere(tissue == TissueTypes.SURFACTANT.value)

            for i in range(0, neutrophil.init_num):
                j = random.randint(0, len(indices)-1)
                x = random.uniform(grid.xv[indices[j][2]], grid.xv[indices[j][2] + 1])
                y = random.uniform(grid.yv[indices[j][1]], grid.yv[indices[j][1] + 1])
                z = random.uniform(grid.zv[indices[j][0]], grid.zv[indices[j][0] + 1])

                point = Point(x=x, y=y, z=z)
                status = NeutrophilCellData.Status.RESTING

                neutrophil.cells.append(NeutrophilCellData.create_cell(point=point, status=status))

        return state

    def advance(self, state: State, previous_time: float):
        neutrophil: NeutrophilState = state.neutrophil
        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        cells = neutrophil.cells
        iron = state.molecules.grid.concentrations.iron
        # drift(neutrophil.cells, tissue, grid)
        interact(state)

        cells.recruit(NeutrophilCellData.RECRUIT_RATE, tissue, grid)
        cells.remove(NeutrophilCellData.LEAVE_RATE, tissue, grid)
        # cells.update
        cells.chemotaxis(
            iron, neutrophil.DRIFT_LAMBDA, neutrophil.DRIFT_BIAS, tissue, grid,
        )

        # print(neutrophil.cells.cell_data['point'])

        return state


def interact(state: State):
    # get molecules in voxel
    iron = state.molecules.grid.concentrations.iron  # TODO implement real behavior
    cells = state.neutrophil.cells
    grid = state.grid

    # 1. Get cells that are alive
    for index in cells.alive():

        # 2. Get voxel for each cell to get agents in that voxel
        cell = cells[index]
        vox = grid.get_voxel(cell['point'])

        # 3. Interact with all molecules

        #  Iron -----------------------------------------------------
        iron_amount = iron[vox.z, vox.y, vox.x]
        qtty = 0.5 * iron_amount
        iron[vox.z, vox.y, vox.x] -= qtty
        cell['iron_pool'] += qtty

        #  Next_Mol -----------------------------------------------------
        #    next_mol_amount = iron[vox.z, vox.y, vox.x] ...
