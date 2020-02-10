from random import shuffle
from typing import List

import attr
import numpy as np

from simulation.coordinates import Voxel
from simulation.module import Module, ModuleState
from simulation.modules.geometry import TissueTypes
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
        start_time = time.time()
        cell_cell_interact: CellCellInteractState = state.cell_cell_interact
        tissue = state.geometry.lung_tissue

        macrophage_cells = state.macrophage.cells
        neutrophil_cells = state.neutrophil.cells
        afumigatus_cells = state.afumigatus.tree.cells

        a = np.argwhere(tissue == TissueTypes.SURFACTANT.value)
        b = np.argwhere(tissue == TissueTypes.EPITHELIUM.value)
        c = np.argwhere(tissue == TissueTypes.BLOOD.value)
        d = np.argwhere(tissue == TissueTypes.PORE.value)
        
        for index in np.concatenate((a,b,c,d)):
            vox = Voxel(x=index[2],y=index[1],z=index[0])
            cells_in_voxel = []

            for cell_index in macrophage_cells.get_cells_in_voxel(vox):
                cells_in_voxel.append(macrophage_cells.cell_data[cell_index])

            for cell_index in neutrophil_cells.get_cells_in_voxel(vox):
                cells_in_voxel.append(neutrophil_cells.cell_data[cell_index])
            
            for cell_index in afumigatus_cells.get_cells_in_voxel(vox):
                cells_in_voxel.append(afumigatus_cells.cell_data[cell_index])  
            
            size = len(cells_in_voxel)
            if(size > 1):
                for i in range(size):
                    for j in range(i+1, size):
                        interact(cells_in_voxel[i], cells_in_voxel[j])

        return state

def interact(cell_1, cell_2):
    # macrophage
    #       macrophage
    # 
    # afumigatus
    #       afumigatus
    #       macrophage
    # 
    # neutrophil
    #       neutrophil
    #       afumigatus
    #       macrophage

    # ---------------------------------------------------------
    # 1. macrophage
    
    if(cell_1['name'] == 'macrophage' and cell_2['name'] == 'macrophage'):
        #print('mac - mac')
        return

    # ---------------------------------------------------------
    # 2. afumigatus

    # afumigatus
    if(cell_1['name'] == 'afumigatus' and cell_2['name'] == 'afumigatus'):
        #print('af - af')
        return

    # macrophage
    if(cell_2['name'] == 'afumigatus' and cell_1['name'] == 'macrophage'):
        temp_1 = cell_1
        cell_1 = cell_2
        cell_2 = temp_1
    if(cell_1['name'] == 'afumigatus' and cell_2['name'] == 'macrophage'):
        #print('af - mac')
        return

    # ---------------------------------------------------------
    # 3 .neutrophil
    #
    # neutrophil
    if(cell_1['name'] == 'neutrophil' and cell_2['name'] == 'neutrophil'):
        #print('neu - neu')
        return
     
    # afumigatus
    if(cell_2['name'] == 'neutrophil' and cell_1['name'] == 'afumigatus'):
        temp_1 = cell_1
        cell_1 = cell_2
        cell_2 = temp_1
    if(cell_1['name'] == 'neutrophil' and cell_2['name'] == 'afumigatus'):
        #print('af - neu')
        return
     
    # macrophage
    if(cell_2['name'] == 'neutrophil' and cell_1['name'] == 'macrophage'):
        temp_1 = cell_1
        cell_1 = cell_2
        cell_2 = temp_1
    if(cell_1['name'] == 'neutrophil' and cell_2['name'] == 'macrophage'):
        #print('neu - mac')
        return

    # ---------------------------------------------------------

    return False
