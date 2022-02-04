from enum import IntEnum
from typing import Tuple

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellFields, CellList
from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State


class EndothelialSpecies(IntEnum):
    L = 0
    Heme = 1
    TNF = 2
    Shear = 3
    VEGF = 4
    TGF = 5
    BMP2 = 6
    BMP9 = 7
    ICAM = 8
    TNFR2 = 9
    VEGFR2 = 10
    ALK5 = 11
    ALK2 = 12
    ALK1 = 13
    Src = 14
    Smad2 = 15
    Smad1 = 16
    STAT3 = 17
    ERK = 18
    Akt = 19
    AJ = 20
    Ca = 21
    NFkB = 22
    eNOS = 23
    Rac1 = 24
    NOX = 25
    p38 = 26
    HIF1a = 27
    CO = 28
    HO1 = 29
    SF = 30
    RhoA = 31
    AP1 = 32
    HR = 33
    Dep1 = 34


def network_init():
    network = np.zeros(len(EndothelialSpecies), dtype=bool)

    network[EndothelialSpecies.AJ] = True
    # receptors
    network[EndothelialSpecies.TNFR2] = True
    network[EndothelialSpecies.VEGFR2] = True
    network[EndothelialSpecies.ALK5] = True
    network[EndothelialSpecies.ALK2] = True
    network[EndothelialSpecies.ALK1] = True
    # network[EndothelialSpecies.ICAM] = False

    return network


class EndothelialCellData(CellData):
    ENDOTHELIAL_FIELDS: CellFields = [
        ('network', bool, len(EndothelialSpecies)),
        ('pSelectinBloodSide', np.float64), # pico-grams/mL
        ('pSelectinLungSide', np.float64) # pico-grams/mL
    ]

    dtype = np.dtype(CellData.FIELDS + ENDOTHELIAL_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        initializer = {
            'network': kwargs.get('network', network_init()),
        }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + tuple(
            [initializer[key] for key, *_ in EndothelialCellData.ENDOTHELIAL_FIELDS]
        )


@attrs(kw_only=True, frozen=True, repr=False)
class EndothelialCellList(CellList):
    CellDataClass = EndothelialCellData


def cell_list_factory(self: 'EndothelialState') -> EndothelialCellList:
    return EndothelialCellList(grid=self.global_state.grid)


@attrs(kw_only=True)
class EndothelialState(ModuleState):
    cells: EndothelialCellList = attrib(default=attr.Factory(cell_list_factory, takes_self=True))
    pSelectin_heme_threshold: float

class Endothelial(ModuleModel):
    name = 'endothelial'
    StateClass = EndothelialState

    def initialize(self, state: State):
        endothelial: EndothelialState = state.endothelial

        endothelial.pSelectin_heme_threshold = self.config.getfloat('pSelectin_heme_threshold')

        return state

    def advance(self, state: State, previous_time: float) -> State:
        endothelial: EndothelialState = state.endothelial
        heme: HemeState = state.heme
        grid: RectangularGrid = state.grid

        for cell_idx in endothelial.cells.alive():
            cell: EndothelialCellData = endothelial.cells[cell_idx]
            self.boolean_network_update(cell)

            endothelial_voxel: Voxel = grid.get_voxel(cell['point'])
            if heme.grid[tuple(endothelial_voxel)] > endothelial.pSelectin_heme_threshold:
                ... # express pSelectin


        return state

    @staticmethod
    def boolean_network_update(cell: EndothelialCellData):
        temp = np.zeros(len(EndothelialSpecies), dtype=bool)
        network = cell['network']

        # bn[L]=BN[L]
        temp[EndothelialSpecies.L] = False  # sayeth Henrique on Dec 6, 2021

        # bn[Heme]=FALSE
        temp[EndothelialSpecies.Heme] = False

        # bn[TNF]=(BN[AP1] | BN[NFkB]) &!BN[STAT3]
        temp[EndothelialSpecies.TNF] = (
            network[EndothelialSpecies.AP1] or network[EndothelialSpecies.NFkB]
        ) and not network[EndothelialSpecies.STAT3]

        # TODO: finish conversion
        # bn[Shear]=FALSE
        # bn[VEGF]=BN[STAT3] | BN[HIF1a]
        # bn[TGF]=FALSE
        # bn[BMP2]=BN[BMP2]#FALSE
        # bn[BMP9]=BN[BMP9]
        # bn[ICAM]=(BN[AP1] | BN[NFkB]) &! (BN[CO] | BN[STAT3])
        # bn[TNFR2]=TRUE
        # bn[VEGFR2]=!(BN[Dep1] | BN[Smad1])
        # bn[ALK5]=TRUE
        # bn[ALK2]=TRUE
        # bn[ALK1]=TRUE
        # bn[Src]=(BN[TNFR2] & BN[TNF]) | (BN[ICAM] & BN[L]) | (BN[VEGFR2] & BN[VEGF]) | (BN[ALK2] & BN[BMP2])
        # bn[Smad2]=(BN[ALK5] & BN[TGF]) & !BN[Smad1]
        # bn[Smad1]=(BN[ALK2] & BN[BMP2]) | (BN[ALK1] & BN[BMP9])# | BN[Smad1]
        # bn[STAT3]=BN[Src]
        # bn[ERK]=((BN[ALK2] & BN[BMP2]) | (BN[VEGF] & BN[VEGFR2])) &!(BN[Akt] | (BN[ALK1] & BN[BMP9]))
        # bn[Akt]=BN[Src] & !(BN[ALK1] & BN[BMP9])
        # bn[AJ]=!(BN[eNOS] | BN[NOX] | BN[Src] | BN[Smad2] | BN[ERK])
        # bn[Ca]=(BN[VEGFR2] & BN[VEGF]) | (BN[ICAM] & BN[L]) | BN[Shear]
        # bn[NFkB]=BN[Akt] | BN[NOX]
        # bn[eNOS]=BN[Akt] | BN[Ca] | BN[Shear]
        # bn[Rac1]=((BN[ICAM] & BN[L]) | BN[Akt]) & !BN[RhoA]
        # bn[NOX]=(((BN[Ca] | BN[Akt]) & BN[Rac1]) | BN[HR])# &!BN[STAT3]
        # bn[p38]=BN[eNOS] | BN[NOX] | BN[Smad1] | BN[Smad2] | (BN[TNFR2] & BN[TNF])
        # bn[HIF1a]=BN[Smad2] | BN[eNOS] | BN[NOX]
        # bn[CO]=BN[Heme] & BN[HO1]
        # bn[HO1]=BN[NFkB] | BN[HIF1a]
        # bn[SF]=BN[eNOS] | BN[RhoA]
        # bn[RhoA]=(BN[p38] | BN[Rac1]) & !BN[NOX]
        # bn[AP1]=BN[ERK] & BN[p38]
        # bn[HR]=BN[HR]
        # bn[Dep1]=!BN[AJ]

        cell['network'][:] = temp[:]
