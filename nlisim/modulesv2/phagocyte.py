from enum import auto, IntEnum, unique

from attr import attrs
import numpy as np

from nlisim.cell import CellData
from nlisim.module import ModuleModel, ModuleState
from nlisim.modulesv2.afumigatus import AfumigatusCellData, FungalForm, FungalState

MAX_CONIDIA = 50  # note: this the max that we can set the max to. i.e. not an actual model parameter


class PhagocyteCellData(CellData):
    PHAGOCYTE_FIELDS = [
        ('phagosome', (np.int32, MAX_CONIDIA)),
        ('has_conidia', np.bool),
        ]

    dtype = np.dtype(CellData.FIELDS + PHAGOCYTE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(cls, **kwargs, ) -> np.record:
        initializer = {
            'phagosome':   kwargs.get('phagosome',
                                      -1 * np.ones(MAX_CONIDIA, dtype=np.int)),
            'has_conidia': kwargs.get('has_conidia',
                                      False),
            }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + \
               [initializer[key] for key, _ in PhagocyteCellData.PHAGOCYTE_FIELDS]


@attrs(kw_only=True)
class PhagocyteState(ModuleState):
    max_conidia: int


class PhagocyteModel(ModuleModel):
    pass


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


def internalize_aspergillus(phagocyte_cell: PhagocyteCellData,
                            aspergillus_cell: AfumigatusCellData,
                            phagocyte: PhagocyteState,
                            phagocytize: bool = False):
    """
    Possibly have a phagocyte phagocytize a fungal cell

    Parameters
    ----------
    phagocyte_cell : PhagoCyteCellData
    aspergillus_cell : AfumigatusCellData
    phagocyte : PhagocyteState
    phagocytize : bool

    Returns
    -------

    """

    # We cannot internalize an already internalized fungal cell
    if aspergillus_cell['state'] != FungalState.FREE:
        return

    # deal with conidia
    if (aspergillus_cell['status'] in {FungalForm.RESTING_CONIDIA,
                                       FungalForm.SWELLING_CONIDIA,
                                       FungalForm.STERILE_CONIDIA} or phagocytize):
        if (phagocyte_cell['status'] not in {PhagocyteStatus.NECROTIC,
                                             PhagocyteStatus.APOPTOTIC,
                                             PhagocyteStatus.DEAD}):
            # check to see if we have room before we add in another cell to the phagosome
            num_cells_in_phagosome = np.sum(phagocyte_cell['phagosome'] >= 0)
            if num_cells_in_phagosome < phagocyte.max_conidia:
                phagocyte_cell['has_conidia'] = True
                aspergillus_cell['state'] = FungalState.INTERNALIZING
                # place the fungal cell in the phagosome,
                # sorting makes sure that an 'empty' i.e. -1 slot is first
                phagocyte_cell['phagosome'].sort()
                phagocyte_cell['phagosome'][0] = aspergillus_cell['id']

    # TODO: what is going on here? is the if too loose?
    if aspergillus_cell['status'] != FungalForm.RESTING_CONIDIA:
        phagocyte_cell['state'] = PhagocyteStatus.INTERACTING
        if phagocyte_cell['status'] != PhagocyteStatus.ACTIVE:
            phagocyte_cell['status'] = PhagocyteStatus.ACTIVATING
        else:
            phagocyte_cell['status_iteration'] = 0
