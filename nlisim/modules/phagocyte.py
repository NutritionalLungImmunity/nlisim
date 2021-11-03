from abc import abstractmethod
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Tuple

from attr import attrs
import numpy as np

from nlisim.cell import CellData, CellFields, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State

if TYPE_CHECKING:  # prevent circular imports for type checking
    from nlisim.modules.afumigatus import AfumigatusCellData

MAX_CONIDIA = (
    30  # note: this the max that we can set the max to. i.e. not an actual model parameter
)


class PhagocyteCellData(CellData):
    PHAGOCYTE_FIELDS: CellFields = [
        ('phagosome', np.int64, MAX_CONIDIA),
    ]

    dtype = np.dtype(CellData.FIELDS + PHAGOCYTE_FIELDS, align=True)  # type: ignore

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
            [initializer[key] for key, *_ in PhagocyteCellData.PHAGOCYTE_FIELDS]
        )


@attrs(kw_only=True)
class PhagocyteModuleState(ModuleState):
    max_conidia: int  # units: count


class PhagocyteModel(ModuleModel):
    def single_step_move(
        self, state: State, cell: PhagocyteCellData, cell_index: int, cell_list: CellList
    ) -> None:
        """
        Move the phagocyte one 1 µm, probabilistically.

        depending on single_step_probabilistic_drift

        Parameters
        ----------
        state : State
            the global simulation state
        cell : PhagocyteCellData
            the cell to move
        cell_index : int
            index of cell in cell_list
        cell_list : CellList
            the CellList for the cell-type (macrophage/neutrophil/etc) of cell


        Returns
        -------
        nothing
        """
        grid: RectangularGrid = state.grid

        # At this moment, there is no inter-voxel geometry.
        cell_voxel: Voxel = grid.get_voxel(cell['point'])
        new_point: Point = self.single_step_probabilistic_drift(state, cell, cell_voxel)
        cell['point'] = new_point
        cell_list.update_voxel_index([cell_index])

    @abstractmethod
    def single_step_probabilistic_drift(
        self, state: State, cell: PhagocyteCellData, voxel: Voxel
    ) -> Point:
        ...

    @staticmethod
    def release_phagosome(state: State, phagocyte_cell: PhagocyteCellData) -> None:
        """
        Release afumigatus cells in the phagosome

        Parameters
        ----------
        state : State
            global simulation state
        phagocyte_cell : PhagocyteCellData


        Returns
        -------
        Nothing
        """
        from nlisim.modules.afumigatus import AfumigatusCellState, AfumigatusState

        afumigatus: AfumigatusState = state.afumigatus

        for fungal_cell_index in phagocyte_cell['phagosome']:
            if fungal_cell_index == -1:
                continue
            afumigatus.cells[fungal_cell_index]['state'] = AfumigatusCellState.RELEASING
        phagocyte_cell['phagosome'][:] = -1


# TODO: better name
@unique
class PhagocyteState(IntEnum):
    FREE = 0
    INTERACTING = 1


# TODO: name
@unique
class PhagocyteStatus(IntEnum):
    INACTIVE = 0
    INACTIVATING = 1
    RESTING = 2
    ACTIVATING = 3
    ACTIVE = 4
    APOPTOTIC = 5
    NECROTIC = 6
    DEAD = 7
    ANERGIC = 8
    INTERACTING = 9


# noinspection PyUnresolvedReferences
def interact_with_aspergillus(
    *,
    phagocyte_cell: PhagocyteCellData,
    phagocyte_cell_index: int,
    phagocyte_cells: CellList,
    aspergillus_cell: 'AfumigatusCellData',
    aspergillus_cell_index: int,
    phagocyte: PhagocyteModuleState,
    phagocytize: bool = False,
) -> None:
    """
    Possibly have a phagocyte phagocytize a fungal cell.

    Parameters
    ----------
    phagocyte_cell : PhagocyteCellData
    phagocyte_cell_index: int
    aspergillus_cell : AfumigatusCellData
    aspergillus_cell_index : int
    phagocyte : PhagocyteState
    phagocytize : bool
    """
    from nlisim.modules.afumigatus import AfumigatusCellState, AfumigatusCellStatus

    # We cannot internalize an already internalized fungal cell
    if aspergillus_cell['state'] != AfumigatusCellState.FREE:
        return

    # internalize conidia
    if phagocytize or aspergillus_cell['status'] in {
        AfumigatusCellStatus.RESTING_CONIDIA,
        AfumigatusCellStatus.SWELLING_CONIDIA,
        AfumigatusCellStatus.STERILE_CONIDIA,
    }:
        if phagocyte_cell['status'] not in {
            PhagocyteStatus.NECROTIC,
            PhagocyteStatus.APOPTOTIC,
            PhagocyteStatus.DEAD,
        }:
            # check to see if we have room before we add in another cell to the phagosome
            num_cells_in_phagosome = np.sum(phagocyte_cell['phagosome'] >= 0)
            if num_cells_in_phagosome < phagocyte.max_conidia:
                aspergillus_cell['state'] = AfumigatusCellState.INTERNALIZING
                # place the fungal cell in the phagosome,
                # sorting makes sure that an 'empty' i.e. -1 slot is first
                phagocyte_cell['phagosome'].sort()
                phagocyte_cell['phagosome'][0] = aspergillus_cell_index
                # move the phagocyte to the location of the aspergillus
                phagocyte_cell['point'] = aspergillus_cell['point']
                phagocyte_cells.update_voxel_index([phagocyte_cell_index])

    # All phagocytes are activated by their interaction, except with resting conidia
    if aspergillus_cell['status'] == AfumigatusCellStatus.RESTING_CONIDIA:
        return
    phagocyte_cell['state'] = PhagocyteStatus.INTERACTING
    if phagocyte_cell['status'] != PhagocyteStatus.ACTIVE:
        # non-active phagocytes begin the activation stage
        if phagocyte_cell['status'] != PhagocyteStatus.ACTIVATING:
            # reset the counter, first time only
            phagocyte_cell['status_iteration'] = 0
        phagocyte_cell['status'] = PhagocyteStatus.ACTIVATING
    else:
        # active phagocytes are kept active by resetting their iteration counter
        phagocyte_cell['status_iteration'] = 0
