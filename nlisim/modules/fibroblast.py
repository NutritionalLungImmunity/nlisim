#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:36:15 2021

@author: wheeler
"""

import math
from enum import IntEnum
from typing import Any, Dict, Tuple

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellFields
from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.afumigatus import AfumigatusCellStatus, AfumigatusState
from nlisim.modules.hemoglobin import HemoglobinState
from nlisim.modules.hemolysin import HemolysinState
from nlisim.modules.macrophage import MacrophageState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import TissueType, activation_function


class FibroblastStatus(IntEnum):
    RESTING = 0
    ACTIVE = 1


class FibroblastCellData(CellData):
    FIBROBLAST_FIELDS: CellFields = [
        ('status', np.uint8),
        ('fpn', bool),
        ('fpn_iteration', np.int64),
        ('tf', bool),  # TODO: descriptive name, transferrin?
        ('tnfa', bool),
        ('iron_pool', np.float64),
        ('status_iteration', np.uint64),
        ('velocity', np.float64, 3),
        ('color', 'red', 'puce', 'green'),
    ]

    dtype = np.dtype(CellData.FIELDS + FIBROBLAST_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        initializer = {
            'status': kwargs.get('status', FibroblastStatus.RESTING),
            'fpn': kwargs.get('fpn', True),
            'fpn_iteration': kwargs.get('fpn_iteration', 0),
            'tf': kwargs.get('tf', False),
            'tnfa': kwargs.get('tnfa', False),
            'iron_pool': kwargs.get('iron_pool', 0.0),
            'status_iteration': kwargs.get('status_iteration', 0),
            'velocity': kwargs.get('root', np.zeros(3, dtype=np.float64)),
        }

        # ensure that these come in the correct order
        return CellData.create_cell_tuple(**kwargs) + tuple(
            [initializer[key] for key, *_ in FibroblastCellData.FIBROBLAST_FIELDS]
        )
