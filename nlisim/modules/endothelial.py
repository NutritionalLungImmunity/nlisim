from enum import IntEnum
from typing import Tuple

from attr import attr, attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellFields, CellList
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

class EndothelialCellData(CellData):
    ENDOTHELIAL_FIELDS: CellFields = [
        ('network', bool, len(EndothelialSpecies)),
    ]

    dtype = np.dtype(CellData.FIELDS + ENDOTHELIAL_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        initializer = {
            'phagosome': kwargs.get('phagosome', -1 * np.ones(MAX_CONIDIA, dtype=np.int64)),
        }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + tuple(
            [initializer[key] for key, *_ in PhagocyteCellData.ENDOTHELIAL_FIELDS]
        )


@attrs(kw_only=True, frozen=True, repr=False)
class EndothelialCellList(CellList):
    CellDataClass = EndothelialCellData

def cell_list_factory(self: 'EndothelialState') -> EndothelialCellList:
    return EndothelialCellList(grid=self.global_state.grid)

@attrs(kw_only=True)
class EndothelialState(ModuleState):
    cells: EndothelialCellList = attrib(default=attr.Factory(cell_list_factory, takes_self=True))


class Endothelial(ModuleModel):
    name = 'endothelial'
    StateClass = EndothelialState

    def initialize(self, state: State):
        return state

    def advance(self, state: State, previous_time: float) -> State:
        return state

# bn[L]=BN[L]
# bn[Heme]=FALSE
# bn[TNF]=(BN[AP1] | BN[NFkB]) &!BN[STAT3]
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
