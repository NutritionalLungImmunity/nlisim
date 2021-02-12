from enum import auto, IntEnum, unique

from attr import attrs
import numpy as np

from nlisim.cell import CellData
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State

MAX_CONIDIA = 30  # note: this the max that we can set the max to. i.e. not an actual model parameter


class PhagocyteCellData(CellData):
    PHAGOCYTE_FIELDS = [
        ('phagosome', (np.int64, MAX_CONIDIA)),
        ('has_conidia', bool),
        ]

    dtype = np.dtype(CellData.FIELDS + PHAGOCYTE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs, ) -> np.record:
        initializer = {
            'phagosome':   kwargs.get('phagosome',
                                      -1 * np.ones(MAX_CONIDIA, dtype=np.int64)),
            'has_conidia': kwargs.get('has_conidia',
                                      False),
            }

        # ensure that these come in the correct order
        return \
            CellData.create_cell_tuple(**kwargs) + \
            tuple([initializer[key] for key, *_ in PhagocyteCellData.PHAGOCYTE_FIELDS])


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

    # def calc_drift_probability(voxel, agent):
    #     from edu.uchc.geometry.Voxel import Voxel
    #     if agent.attracted_by() is None:
    #         return
    #
    #     chemokine = voxel.molecules[agent.attracted_by()]
    #     chemoattraction = chemokine.chemoattract(voxel.x, voxel.y, voxel.z)
    #
    #     voxel.p = chemoattraction + (0.0 if len(voxel.infectious_agent) > 0 else 0.0)
    #
    #     cum_p = voxel.p
    #     for v in voxel.neighbors:
    #         chemokine = v.molecules[agent.attracted_by()]
    #         chemoattraction = chemokine.chemoattract(voxel.x, voxel.y, voxel.z)
    #         v.p = (chemoattraction + (
    #             0.0 if len(voxel.infectious_agent) > 0 else 0.0)) if v.tissue_type != Voxel.AIR else 0.0
    #         cum_p = cum_p + v.p
    #     voxel.p = voxel.p / cum_p
    #     for v in voxel.neighbors:
    #         v.p = v.p / cum_p

    def release_phagosome(self, state: State, phagocyte_cell: PhagocyteCellData) -> None:
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
        from nlisim.modulesv2.afumigatus import AfumigatusCellState, AfumigatusState

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
    INTERACTING = auto()  # TODO: is this dead code?


# TODO: name
@unique
class PhagocyteStatus(IntEnum):
    INACTIVE = 0
    INACTIVATING = auto()
    RESTING = auto()
    ACTIVATING = auto()
    ACTIVE = auto()
    APOPTOTIC = auto()
    NECROTIC = auto()
    DEAD = auto()
    ANERGIC = auto()


# noinspection PyUnresolvedReferences
def internalize_aspergillus(phagocyte_cell: PhagocyteCellData,
                            aspergillus_cell: 'AfumigatusCellData',
                            aspergillus_cell_index: int,
                            phagocyte: PhagocyteModuleState,
                            phagocytize: bool = False) -> None:
    """
    Possibly have a phagocyte phagocytize a fungal cell

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
    from nlisim.modulesv2.afumigatus import AfumigatusCellStatus, AfumigatusCellState

    # We cannot internalize an already internalized fungal cell
    if aspergillus_cell['state'] != AfumigatusCellState.FREE:
        return

    # deal with conidia
    if (aspergillus_cell['status'] in {AfumigatusCellStatus.RESTING_CONIDIA,
                                       AfumigatusCellStatus.SWELLING_CONIDIA,
                                       AfumigatusCellStatus.STERILE_CONIDIA} or phagocytize):
        if (phagocyte_cell['status'] not in {PhagocyteStatus.NECROTIC,
                                             PhagocyteStatus.APOPTOTIC,
                                             PhagocyteStatus.DEAD}):
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
