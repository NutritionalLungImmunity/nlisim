from enum import IntEnum
import random

import attr
import numpy as np

from simulation.cell import CellData, CellList
from simulation.coordinates import Point
from simulation.grid import RectangularGrid
from simulation.module import Module, ModuleState
from simulation.modules.geometry import TissueTypes
from simulation.state import State


class NeutrophilCellData(CellData):

    class Status(IntEnum):
        INACTIVE = 0
        INACTIVATING = 1
        RESTING = 2
        ACTIVATING = 3
        ACTIVE = 4
        APOPTOTIC = 5
        NECROTIC = 6
        DEAD = 7

    NEUTROPHIL_FIELDS = [
        ('status', 'u1'),
        ('iteration', 'i4'),
    ]

    dtype = np.dtype(
        CellData.FIELDS + NEUTROPHIL_FIELDS, align=True
    )  # type: ignore

    @classmethod
    def create_cell_tuple(cls, status, **kwargs,) -> np.record:
        status = status
        iteration = 0
        return CellData.create_cell_tuple(**kwargs) + (status,iteration,)


@attr.s(kw_only=True, frozen=True, repr=False)
class NeutrophilCellList(CellList):
    CellDataClass = NeutrophilCellData


def cell_list_factory(self: 'NeutrophilState'):
    return NeutrophilCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class NeutrophilState(ModuleState):
    cells: NeutrophilCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    neutropenic: bool = False
    rec_rate_ph: int = 6
    n_absorb: float = 0.9


class Neutrophil(Module):
    name = 'neutrophil'
    defaults = {
        'cells': '',
        'neutropenic': 'False',
        'rec_rate_ph': '10',
        'DRIFT_BIAS': '0.9',
    }
    StateClass = NeutrophilState

    def initialize(self, state: State):
        neutrophil: NeutrophilState = state.neutrophil
        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        neutrophil.neutropenic = self.config.getboolean('neutropenic')
        neutrophil.rec_rate_ph = self.config.getint('rec_rate_ph')
        neutrophil.rec_r = self.config.getfloat('rec_r')
        neutrophil.n_absorb = self.config.getfloat('n_absorb')
        #NeutrophilCellData.LEAVE_RATE = self.config.getfloat('LEAVE_RATE')

        neutrophil.cells = NeutrophilCellList(grid=grid)

        return state

    def advance(self, state: State, previous_time: float):

        recruit_new(state, previous_time)
        absorb_cytokines(state)
        produce_cytokines(state)
        move(state)
        damage_hyphae(state)

        return state


def recruit_new(state, time):
    neutrophil: NeutrophilState = state.neutrophil
    n_cells = neutrophil.cells
    tissue = state.geometry.lung_tissue
    grid = state.grid
    cyto = state.molecules.grid['n_cyto']

    num_reps = neutrophil.rec_rate_ph # number of neutrophils recruited per time step
    if (neutrophil.neutropenic and time >= 48 and time <= 96):
        num_reps = (time - 48) / 8
    
    blood_index = np.argwhere(tissue == TissueTypes.BLOOD.value)
    blood_index = np.transpose(blood_index)
    mask = cyto[blood_index[2], blood_index[1], blood_index[0]] >= neutrophil.rec_r
    blood_index = np.transpose(blood_index)
    cyto_index = blood_index[mask]
    np.random.shuffle(cyto_index)

    for i in range(0, num_reps):
        if(len(cyto_index) > 0):
            ii = random.randint(0, len(cyto_index) - 1)
            point = Point(
                x = grid.x[cyto_index[ii][2]], 
                y = grid.x[cyto_index[ii][1]], 
                z = grid.x[cyto_index[ii][0]])

            status = NeutrophilCellData.Status.RESTING
            n_cells.append(NeutrophilCellData.create_cell(point=point, status=status))
            

def absorb_cytokines(state):
    neutrophil: NeutrophilState = state.neutrophil
    n_cells = neutrophil.cells
    cyto = state.molecules.grid['n_cyto']
    grid = state.grid

    for index in n_cells.alive():
        vox = grid.get_voxel(n_cells[index]['point'])
        x = vox.x
        y = vox.y
        z = vox.z
        cyto[z,y,x] = (1 - neutrophil.n_absorb) * cyto[z,y,x]

    return state


def produce_cytokines(state):
    neutrophil: NeutrophilState = state.neutrophil
    n_cells = neutrophil.cells

    tissue = state.geometry.lung_tissue
    grid = state.grid

    cyto = state.molecules.grid.concentration.n_cyto

    return state


def move(state):
    neutrophil: NeutrophilState = state.neutrophil
    n_cells = neutrophil.cells

    tissue = state.geometry.lung_tissue
    grid = state.grid

    cyto = state.molecules.grid.concentration.n_cyto

    return state


def damage_hyphae(state):
    neutrophil: NeutrophilState = state.neutrophil
    n_cells = neutrophil.cells

    tissue = state.geometry.lung_tissue
    grid = state.grid

    cyto = state.molecules.grid.concentration.n_cyto

    return state
