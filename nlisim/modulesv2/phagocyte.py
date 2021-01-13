from enum import unique, IntEnum, auto

from nlisim.module import ModuleModel
from nlisim.modulesv2.afumigatus import FungalForm


class PhagocyteModel(ModuleModel):

    @staticmethod
    def int_aspergillus(phagocyte, aspergillus, phagocytize=False):
        if aspergillus.state == aspergillus.FREE:
            if aspergillus.status in {FungalForm.RESTING_CONIDIA,
                                      FungalForm.SWELLING_CONIDIA,
                                      FungalForm.STERILE_CONIDIA} or phagocytize:
                if phagocyte.status not in {PhagocyteStatus.NECROTIC,
                                            PhagocyteStatus.APOPTOTIC,
                                            PhagocyteStatus.DEAD}:
                    if len(phagocyte.phagosome.agents) < phagocyte._get_max_conidia():
                        phagocyte.phagosome.has_conidia = True
                        aspergillus.state = aspergillus.INTERNALIZING
                        phagocyte.phagosome.agents[aspergillus.id] = aspergillus
            if aspergillus.status != aspergillus.RESTING_CONIDIA:
                phagocyte.state = PhagocyteStatus.INTERACTING
                if phagocyte.status != PhagocyteStatus.ACTIVE:
                    phagocyte.status = PhagocyteStatus.ACTIVATING
                else:
                    phagocyte.status_iteration = 0


# TODO: name
@unique
class PhagocyteState(IntEnum):
    FREE = 0
    INTERACTING = auto()


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
