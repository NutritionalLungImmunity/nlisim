from abc import abstractmethod
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Tuple

from attr import attrs
import numpy as np

from nlisim.cell import CellData
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
    PHAGOCYTE_FIELDS = [
        ('phagosome', (np.int64, MAX_CONIDIA)),
        ('has_conidia', bool),
    ]

    dtype = np.dtype(CellData.FIELDS + PHAGOCYTE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        initializer = {
            'phagosome': kwargs.get('phagosome', -1 * np.ones(MAX_CONIDIA, dtype=np.int64)),
            'has_conidia': kwargs.get('has_conidia', False),
        }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + tuple(
            [initializer[key] for key, *_ in PhagocyteCellData.PHAGOCYTE_FIELDS]
        )


@attrs(kw_only=True)
class PhagocyteModuleState(ModuleState):
    max_conidia: int


class PhagocyteModel(ModuleModel):

    # def move(self, old_voxel, steps):
    #     if steps < self.get_max_move_steps():
    #         calc_drift_probability(old_voxel, self)
    #         new_voxel = get_voxel(old_voxel, random())
    #
    #         old_voxel.remove_cell(self.id)
    #         new_voxel.set_cell(self)
    #         steps += 1
    #
    #         for _, a in self.phagosome.items():
    #             a.x = new_voxel.x + random()
    #             a.y = new_voxel.y + random()
    #             a.z = new_voxel.z + random()
    #
    #         return self.move(new_voxel, steps)

    def single_step_move(self, state: State, cell: PhagocyteCellData) -> None:
        """
        Move the phagocyte one step (voxel) probabilistically.

        depending on single_step_probabilistic_drift

        Parameters
        ----------
        state : State
            the global simulation state
        cell : PhagocyteCellData
            the cell to move

        Returns
        -------
        nothing
        """
        grid: RectangularGrid = state.grid

        # At this moment, there is no inter-voxel geometry, but I'm keeping the offset around
        # just in case.
        cell_point: Point = cell['point']
        cell_voxel: Voxel = grid.get_voxel(cell['point'])
        offset: np.ndarray = cell_point - cell_voxel
        new_voxel: Voxel = self.single_step_probabilistic_drift(state, cell, cell_voxel)
        cell['point'] = new_voxel + offset

    @abstractmethod
    def single_step_probabilistic_drift(
        self, state: State, cell: PhagocyteCellData, voxel: Voxel
    ) -> Voxel:
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
        phagocyte_cell['phagosome'].fill(-1)


# TODO: name
@unique
class PhagocyteState(IntEnum):
    FREE = 0
    INTERACTING = 1  # TODO: is this dead code?


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
    INTERACTING = 9  # TODO: check


# noinspection PyUnresolvedReferences
def internalize_aspergillus(
    phagocyte_cell: PhagocyteCellData,
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
    aspergillus_cell : AfumigatusCellData
    aspergillus_cell_index : int
    phagocyte : PhagocyteState
    phagocytize : bool

    Returns
    -------
    Nothing
    """
    from nlisim.modules.afumigatus import AfumigatusCellState, AfumigatusCellStatus

    # We cannot internalize an already internalized fungal cell
    if aspergillus_cell['state'] != AfumigatusCellState.FREE:
        return

    # deal with conidia
    if (
        aspergillus_cell['status']
        in {
            AfumigatusCellStatus.RESTING_CONIDIA,
            AfumigatusCellStatus.SWELLING_CONIDIA,
            AfumigatusCellStatus.STERILE_CONIDIA,
        }
        or phagocytize
    ):
        if phagocyte_cell['status'] not in {
            PhagocyteStatus.NECROTIC,
            PhagocyteStatus.APOPTOTIC,
            PhagocyteStatus.DEAD,
        }:
            # check to see if we have room before we add in another cell to the phagosome
            num_cells_in_phagosome = np.sum(phagocyte_cell['phagosome'] >= 0)
            if num_cells_in_phagosome < phagocyte.max_conidia:
                phagocyte_cell['has_conidia'] = True
                aspergillus_cell['state'] = AfumigatusCellState.INTERNALIZING
                # place the fungal cell in the phagosome,
                # sorting makes sure that an 'empty' i.e. -1 slot is first
                phagocyte_cell['phagosome'].sort()
                phagocyte_cell['phagosome'][0] = aspergillus_cell_index

    # TODO: what is going on here? is the if too loose?
    if aspergillus_cell['status'] != AfumigatusCellStatus.RESTING_CONIDIA:
        phagocyte_cell['state'] = PhagocyteStatus.INTERACTING
        if phagocyte_cell['status'] != PhagocyteStatus.ACTIVE:
            phagocyte_cell['status'] = PhagocyteStatus.ACTIVATING
        else:
            phagocyte_cell['status_iteration'] = 0
