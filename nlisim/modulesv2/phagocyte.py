from enum import auto, IntEnum, unique

from nlisim.module import ModuleModel
from nlisim.modulesv2.afumigatus import FungalForm


class PhagocyteModel(ModuleModel):

    pass


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
