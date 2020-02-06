from random import shuffle
from typing import List

import attr
import numpy as np

from simulation.coordinates import Voxel
from simulation.module import Module, ModuleState
from simulation.state import State


@attr.s(kw_only=True)
class CellCellInteractState(ModuleState):
    names: List[str] = []
    PR_INT_RESTIN: float = 1.0
    PR_INT_ACTIVE: float = 1.0


class CellCellInteract(Module):
    name = 'cell_cell_interact'
    defaults = {
        'names': '',
        'PR_INT_RESTIN': '1.0',
        'PR_INT_ACTIVE': '1.0',
    }
    StateClass = CellCellInteractState

    def initialize(self, state: State):
        cell_cell_interact = state.cell_cell_interact

        cell_cell_interact.PR_INT_RESTIN = self.config.getfloat('PR_INT_RESTIN')
        cell_cell_interact.PR_INT_ACTIVE = self.config.getfloat('PR_INT_ACTIVE')
        cell_cell_interact.names = state.config.getlist('cell_cell_interact', 'names')
        
        return state

    def advance(self, state: State, previous_time: float):
        cell_cell_interact: CellCellInteractState = state.cell_cell_interact
        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        macrophage_cells = state.macrophage.cells
        neutrophil_cells = state.neutrophil.cells
        afumigatus_cells = state.afumigatus.tree.cells

        for x in range(0, int(grid.xv[-1] + 1)):
            for y in range(0, int(grid.yv[-1] + 1)):
                for z in range(0, int(grid.zv[-1] + 1)):
                    vox = Voxel(x=x,y=y,z=z)
                    cells_in_voxel = []
                    if(len(macrophage_cells.get_cells_in_voxel(vox)) > 0):
                        cells_in_voxel.extend(macrophage_cells.cell_data[macrophage_cells.get_cells_in_voxel(vox)])
                    if(len(neutrophil_cells.get_cells_in_voxel(vox)) > 0):
                        cells_in_voxel.extend(neutrophil_cells.cell_data[neutrophil_cells.get_cells_in_voxel(vox)])
                    if(len(afumigatus_cells.get_cells_in_voxel(vox)) > 0):
                        cells_in_voxel.extend(afumigatus_cells.cell_data[afumigatus_cells.get_cells_in_voxel(vox)])

                    size = len(cells_in_voxel)
                    if(size > 1):
                        shuffle(cells_in_voxel)
                        for i in range(size):
                            for j in range(i, size):
                                interact(cells_in_voxel[i], cells_in_voxel[j])

        return state

def interact(cell_1, cell_2):
    # macrophage
    # macrophage
    # 
    # afumigatus
    # afumigatus
    # 
    # afumigatus
    # macrophage
    # 
    # neutrophil
    # neutrophil
    # 
    # neutrophil
    # afumigatus
    # 
    # neutrophil
    # macrophage
    # --------------------------------

    # macrophage
    # macrophage
    


    # afumigatus
    # afumigatus
    # 
    # afumigatus
    # macrophage
    # 
    # neutrophil
    # neutrophil
    # 
    # neutrophil
    # afumigatus
    # 
    # neutrophil
    # macrophage
    return True
