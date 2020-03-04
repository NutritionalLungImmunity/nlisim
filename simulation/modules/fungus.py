from enum import IntEnum
import math
import random

import attr
import numpy as np

from simulation.cell import CellData, CellList
from simulation.coordinates import Point, Voxel
from simulation.grid import RectangularGrid
from simulation.module import Module, ModuleState
from simulation.modules.geometry import TissueTypes
from simulation.state import State


class FungusCellData(CellData):
    
    class Status(IntEnum):
        DRIFTING = -1
        RESTING = 0 # CONIDIA relevant
        SWOLLEN = 1 # CONIDIA relevant
        GERMINATED = 2 # CONIDIA relevant
        GROWABLE = 3 # HYPHAE relevant
        GROWN = 4 # HYPHAE relevant
        DEAD = 5

    class Form(IntEnum):
        CONIDIA = 0
        HYPHAE = 1
    
    FUNGUS_FIELDS = [
        ('form', 'u1'),
        ('status', 'u1'),
        ('iteration', 'i4'),
        ('mobile', 'b1'),
        ('internalized', 'b1'),
        ('iron', 'f8'),
        ('health', 'f8'),
    ]

    dtype = np.dtype(CellData.FIELDS + FUNGUS_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        *,
        iron: float = 0,
        status: Status = Status.DRIFTING,
        form: Form = Form.CONIDIA,
        **kwargs,
    ) -> np.record:

        iteration = 0
        mobile = True
        internalized = False
        health = 100
        
        return CellData.create_cell_tuple(**kwargs) + (
            form, 
            status, 
            iteration, 
            mobile, 
            internalized, 
            iron,
            health)


@attr.s(kw_only=True, frozen=True, repr=False)
class FungusCellList(CellList):
    CellDataClass = FungusCellData

def cell_list_factory(self: 'FungusState'):
    return FungusCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class FungusState(ModuleState):
    cells: FungusCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    init_num: int = 0
    p_lodge: float = 0.1
    p_internal_swell: float = 0.05
    ITER_TO_CHANGE_STATUS: int = 1


class Fungus(Module):
    name = 'fungus'
    StateClass = FungusState

    def initialize(self, state: State):
        fungus: FungusState = state.fungus
        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        fungus.init_num = self.config.getint('init_num')
        fungus.p_lodge = self.config.getfloat('p_lodge')
        fungus.p_internal_swell = self.config.getfloat('p_internal_swell')
        fungus.ITER_TO_CHANGE_STATUS = self.config.getint('ITER_TO_CHANGE_STATUS')
        fungus.iron_min = self.config.getint('iron_min')
        fungus.iron_max = self.config.getfloat('iron_max')
        fungus.iron_absorb = self.config.getfloat('iron_absorb')

        fungus.cells = FungusCellList(grid=grid)
        if fungus.init_num > 0:
            # initialize the surfactant layer with some fungus in random locations
            indices = np.argwhere(tissue == TissueTypes.SURFACTANT.value)
            np.random.shuffle(indices)

            for i in range(0, fungus.init_num):
                x = random.uniform(grid.xv[indices[i][2]], grid.xv[indices[i][2] + 1])
                y = random.uniform(grid.yv[indices[i][1]], grid.yv[indices[i][1] + 1])
                z = random.uniform(grid.zv[indices[i][0]], grid.zv[indices[i][0] + 1])

                point = Point(x=x, y=y, z=z)
                status = FungusCellData.Status.RESTING

                fungus.cells.append(FungusCellData.create_cell(point=point, status=status))

        return state

    def advance(self, state: State, previous_time: float):
        iron = state.molecules.grid.concentrations.iron

        update(state)       
        iron_uptake(state)
        #grow_hyphae(state)

        return state

def update(state):
    fungus: FungusState = state.fungus
    grid: RectangularGrid = state.grid
    tissue = state.geometry.lung_tissue
    cells = fungus.cells

    for index in cells.alive(
        cells.cell_data['form'] != FungusCellData.Form.HYPHAE
    ):
        cell = cells[index]
        vox = grid.get_voxel(cell['point'])

        if (cell['health'] <= 0):
                cell['status'] = FungusCellData.Status.DEAD
                cell['dead'] = True

        if cell['mobile']:
            # check if at the edge of space
            if (vox.x > Voxel(x=grid.xv[-1], y=grid.yv[0], z=grid.zv[0]).x):
                cell['status'] = FungusCellData.Status.DEAD
                cell['dead'] = True
            else:
                #move
                print('cells move')
            
            # check contact and lodging with epithelium
            if (tissue[vox.z, vox.y, vox.x] == TissueTypes.EPITHELIUM.value):
                if(fungus.p_lodge > random.random()):
                    cell['mobile'] = False
                    cell['status'] = FungusCellData.Status.RESTING
                    cell['iteration'] += 1
        else:
            cell['iteration'] += 1

        # change status
        if cell['iteration'] >= fungus.ITER_TO_CHANGE_STATUS:
            if cell['internalized']:
                print('internal')
                if cell['status'] == FungusCellData.Status.RESTING:
                    if (fungus.p_internal_swell > random.random()):
                        cell['status'] = FungusCellData.Status.SWOLLEN
                elif cell['status'] == FungusCellData.Status.SWOLLEN:
                    cell['status'] = FungusCellData.Status.GERMINATED   
            else:
                print('free')
                if cell['status'] == FungusCellData.Status.RESTING:
                    cell['status'] = FungusCellData.Status.SWOLLEN
                elif cell['status'] == FungusCellData.Status.SWOLLEN:
                    cell['status'] = FungusCellData.Status.GERMINATED
                    #cell['form'] = HYphae?
                
            # TODO - check ODD protocol
            if cell['status'] == FungusCellData.Status.SWOLLEN:
                cell['iteration'] = 0


def iron_uptake(state):
    fungus: FungusState = state.fungus
    grid: RectangularGrid = state.grid
    tissue = state.geometry.lung_tissue
    cells = fungus.cells
    iron = state.molecules.grid['iron']

    hyphae = FungusCellList(grid=grid)
    mask = (
        cells.cell_data['form'] == FungusCellData.Form.HYPHAE and
        cells.cell_data['internalized'] == False and
        cells.cell_data['iron'] < fungus.iron_max
    )
    hyphae.extend(cells[mask])

    for vox_index in np.argwhere(iron > fungus.iron_min):
        vox = Voxel(x=vox_index[2],y=vox_index[1],z=vox_index[0])

        cells_here = hyphae.get_cells_in_voxel(vox)
        iron_split = fungus.iron_absorb * (iron[vox.z, vox.y, vox.x] / len(cells_here))
        for cell_index in cells_here:
            cells[cell_index]['iron'] += iron_split
            if cells[cell_index]['iron'] > fungus.iron_max:
                cells[cell_index]['iron'] = fungus.iron_max
        
        iron[vox.z, vox.y, vox.x] = (1 - fungus.iron_absorb) * iron[vox.z, vox.y, vox.x]

