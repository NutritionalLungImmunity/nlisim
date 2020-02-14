from random import shuffle
from random import random
from typing import List

import attr
import numpy as np

from simulation.coordinates import Voxel
from simulation.module import Module, ModuleState
from simulation.modules.afumigatus import AfumigatusCellData as acd
from simulation.modules.geometry import TissueTypes
from simulation.modules.phagocyte import PhagocyteCellData as pcd
from simulation.state import State


@attr.s(kw_only=True)
class CellCellInteractState(ModuleState):
    names: List[str] = []
    PR_INT_RESTIN: float = 1.0
    PR_INT_ACTIVE: float = 1.0
    PR_N_HYPHAE: float = 1.0
    PR_MA_PHAG: float = 1.0


class CellCellInteract(Module):
    name = 'cell_cell_interact'
    defaults = {
        'names': '',
        'PR_INT_RESTIN': '1.0',
        'PR_INT_ACTIVE': '1.0',
        'PR_N_HYPHAE': '1.0',
        'PR_MA_PHAG': '1.0',
    }
    StateClass = CellCellInteractState

    def initialize(self, state: State):
        cell_cell_interact = state.cell_cell_interact

        cell_cell_interact.PR_INT_RESTIN = self.config.getfloat('PR_INT_RESTIN')
        cell_cell_interact.PR_INT_ACTIVE = self.config.getfloat('PR_INT_ACTIVE')
        cell_cell_interact.PR_N_HYPHAE = self.config.getfloat('PR_N_HYPHAE')
        cell_cell_interact.PR_MA_PHAG = self.config.getfloat('PR_MA_PHAG')
        cell_cell_interact.names = state.config.getlist('cell_cell_interact', 'names')

        return state

    def advance(self, state: State, previous_time: float):
        #start_time = time.time()
        cell_cell_interact: CellCellInteractState = state.cell_cell_interact
        tissue = state.geometry.lung_tissue

        macrophage_cells = state.macrophage.cells
        neutrophil_cells = state.neutrophil.cells
        afumigatus_cells = state.afumigatus.tree.cells

        a = np.argwhere(tissue == TissueTypes.SURFACTANT.value)
        b = np.argwhere(tissue == TissueTypes.EPITHELIUM.value)
        c = np.argwhere(tissue == TissueTypes.BLOOD.value)
        d = np.argwhere(tissue == TissueTypes.PORE.value)

        for index in [[3,3,3]]:#np.concatenate((a, b, c, d)):
            vox = Voxel(x=index[2], y=index[1], z=index[0])
            cells_in_voxel = []

            for cell_index in macrophage_cells.get_cells_in_voxel(vox):
                cells_in_voxel.append(macrophage_cells.cell_data[cell_index])

            for cell_index in neutrophil_cells.get_cells_in_voxel(vox):
                cells_in_voxel.append(neutrophil_cells.cell_data[cell_index])

            for cell_index in afumigatus_cells.get_cells_in_voxel(vox):
                cells_in_voxel.append(afumigatus_cells.cell_data[cell_index])

            size = len(cells_in_voxel)
            if size > 1:
                shuffle(cells_in_voxel)
                for i in range(size):
                    for j in range(i + 1, size):
                        interact(cells_in_voxel[i], cells_in_voxel[j], state)

        return state


def int_afumigatus(phagocyte, afumigatus, phagocytize=False):
        if afumigatus['state'] == acd.State.FREE:
            if ((afumigatus['status'] == acd.Status.RESTING_CONIDIA
                or afumigatus['status'] == acd.Status.SWELLING_CONIDIA) 
                or phagocytize):
                #if len(phagocyte.phagosome.agents) < phagocyte._get_max_conidia():
                #    phagocyte.phagosome.has_conidia = True
                #    afumigatus['state'] = acd.INTERNALIZING
                #    phagocyte.phagosome.agents[afumigatus.id] = afumigatus
                afumigatus['state'] = acd.State.INTERNALIZING
            if afumigatus['status'] != acd.Status.RESTING_CONIDIA:
                phagocyte['status'] = pcd.Status.INTERACTING
                if phagocyte['status'] != pcd.Status.ACTIVE:
                    phagocyte['status'] = pcd.Status.ACTIVATING
                else:
                    print('handle: phagocyte.status_iteration = 0')

def interact(cell_1, cell_2, state: State):
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

    constants: CellCellInteractState = state.cell_cell_interact

    # ---------------------------------------------------------
    # 1. macrophage

    if cell_1['name'] == 'macrophage' and cell_2['name'] == 'macrophage':
        return

    # ---------------------------------------------------------
    # 2. afumigatus

    # afumigatus
    if cell_1['name'] == 'afumigatus' and cell_2['name'] == 'afumigatus':
        return

    # macrophage
    if cell_2['name'] == 'afumigatus' and cell_1['name'] == 'macrophage':
        temp_1 = cell_1
        cell_1 = cell_2
        cell_2 = temp_1
    if cell_1['name'] == 'afumigatus' and cell_2['name'] == 'macrophage':
        if (cell_2['status'] != pcd.Status.APOPTOTIC
            and cell_2['status'] != pcd.Status.NECROTIC
            and cell_2['status'] != pcd.Status.DEAD):
            pr_interact = (
                constants.PR_N_HYPHAE
                if cell_1['status'] == acd.Status.HYPHAE
                else constants.PR_MA_PHAG)
            print('        pr_int=', pr_interact)
            if random() < pr_interact:
                print('             start int_af')
                int_afumigatus(cell_2, cell_1, cell_1['status'] != acd.Status.HYPHAE)
                if (cell_1['status'] == acd.Status.HYPHAE
                    and cell_2['status'] == pcd.Status.ACTIVE):
                    cell_1['status'] = acd.Status.DYING
        return

    # ---------------------------------------------------------
    # 3 .neutrophil
    #
    # neutrophil
    if cell_1['name'] == 'neutrophil' and cell_2['name'] == 'neutrophil':
        return

    # afumigatus
    if cell_2['name'] == 'neutrophil' and cell_1['name'] == 'afumigatus':
        temp_1 = cell_1
        cell_1 = cell_2
        cell_2 = temp_1
    if cell_1['name'] == 'neutrophil' and cell_2['name'] == 'afumigatus':
        return

    # macrophage
    if cell_2['name'] == 'neutrophil' and cell_1['name'] == 'macrophage':
        temp_1 = cell_1
        cell_1 = cell_2
        cell_2 = temp_1
    if cell_1['name'] == 'neutrophil' and cell_2['name'] == 'macrophage':
        return

    # ---------------------------------------------------------

    return False
