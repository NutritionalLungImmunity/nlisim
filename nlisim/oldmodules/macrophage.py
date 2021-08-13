import itertools
from random import choice, shuffle
from typing import Any, Dict, Tuple

import attr
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.oldmodules.fungus import FungusCellData, FungusCellList
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import TissueType

MAX_CONIDIA = 100

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


class MacrophageCellData(CellData):
    MACROPHAGE_FIELDS = [
        ('iteration', 'i4'),
        ('phagosome', (np.int32, (MAX_CONIDIA))),
    ]

    dtype = np.dtype(CellData.FIELDS + MACROPHAGE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        iteration = 0
        phagosome = np.empty(MAX_CONIDIA)
        phagosome.fill(-1)
        return CellData.create_cell_tuple(**kwargs) + (
            iteration,
            phagosome,
        )


@attr.s(kw_only=True, frozen=True, repr=False)
class MacrophageCellList(CellList):
    CellDataClass = MacrophageCellData

    def len_phagosome(self, index):
        cell = self[index]
        return len(np.argwhere(cell['phagosome'] != -1))

    def append_to_phagosome(self, index, pathogen_index, max_size):
        cell = self[index]
        index_to_append = MacrophageCellList.len_phagosome(self, index)
        if (
            index_to_append < MAX_CONIDIA
            and index_to_append < max_size
            and pathogen_index not in cell['phagosome']
        ):
            cell['phagosome'][index_to_append] = pathogen_index
            return True
        else:
            return False

    def remove_from_phagosome(self, index, pathogen_index):
        phagosome = self[index]['phagosome']
        if pathogen_index in phagosome:
            itemindex = np.argwhere(phagosome == pathogen_index)[0][0]
            size = MacrophageCellList.len_phagosome(self, index)
            if itemindex == size - 1:
                # full phagosome
                phagosome[itemindex] = -1
                return True
            else:
                phagosome[itemindex:-1] = phagosome[itemindex + 1 :]
                phagosome[-1] = -1
                return True
        else:
            return False

    def clear_all_phagosome(self, index, fungus: FungusCellList):
        for i in range(0, self.len_phagosome(index)):
            index = self[index]['phagosome'][i]
            fungus[index]['internalized'] = False
        self[index]['phagosome'].fill(-1)

    def recruit_new(self, rec_rate_ph, rec_r, p_rec_r, tissue, grid, cyto):
        num_reps = rec_rate_ph  # maximum number of macrophages recruited per time step

        cyto_index = np.argwhere(np.logical_and(tissue == TissueType.BLOOD.value, cyto >= rec_r))
        if len(cyto_index) == 0:
            # nowhere to place cells
            return

        for _ in range(num_reps):
            if p_rec_r > rg.random():
                ii = rg.integers(cyto_index.shape[0])
                point = Point(
                    x=grid.x[cyto_index[ii, 2]],
                    y=grid.y[cyto_index[ii, 1]],
                    z=grid.z[cyto_index[ii, 0]],
                )
                # Do we really want these things to always be in the exact center of the voxel?
                # No we do not. Should not have any effect on model, but maybe some on
                # visualization.
                perturbation = rg.multivariate_normal(
                    mean=[0.0, 0.0, 0.0], cov=[[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]
                )
                perturbation_magnitude = np.linalg.norm(perturbation)
                perturbation /= max(1.0, perturbation_magnitude)
                point += perturbation
                self.append(MacrophageCellData.create_cell(point=point))

    def absorb_cytokines(self, m_abs, cyto, grid):
        for index in self.alive():
            vox = grid.get_voxel(self[index]['point'])
            x = vox.x
            y = vox.y
            z = vox.z
            cyto[z, y, x] = (1 - m_abs) * cyto[z, y, x]

    def produce_cytokines(self, m_det, m_n, grid, fungus: FungusCellList, cyto):
        for i in self.alive():
            vox = grid.get_voxel(self[i]['point'])

            hyphae_count = 0

            # Moore neighborhood
            neighborhood = tuple(itertools.product(tuple(range(-1 * m_det, m_det + 1)), repeat=3))

            for dx, dy, dz in neighborhood:
                zi = vox.z + dz
                yj = vox.y + dy
                xk = vox.x + dx
                if grid.is_valid_voxel(Voxel(x=xk, y=yj, z=zi)):
                    index_arr = fungus.get_cells_in_voxel(Voxel(x=xk, y=yj, z=zi))
                    for index in index_arr:
                        if fungus[index]['form'] == FungusCellData.Form.HYPHAE:
                            hyphae_count += 1

            cyto[vox.z, vox.y, vox.x] = cyto[vox.z, vox.y, vox.x] + m_n * hyphae_count

    def move(self, rec_r, grid, cyto, tissue, fungus: FungusCellList):
        for cell_index in self.alive():
            cell = self[cell_index]
            cell_voxel = grid.get_voxel(cell['point'])

            valid_voxel_offsets = []
            above_threshold_voxel_offsets = []

            # iterate over nearby voxels, recording the cytokine levels
            for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3):
                zi = cell_voxel.z + dz
                yj = cell_voxel.y + dy
                xk = cell_voxel.x + dx
                if grid.is_valid_voxel(Voxel(x=xk, y=yj, z=zi)):
                    if tissue[zi, yj, xk] != TissueType.AIR.value:
                        valid_voxel_offsets.append((dx, dy, dz))
                        if cyto[zi, yj, xk] >= rec_r:
                            above_threshold_voxel_offsets.append((cyto[zi, yj, xk], (dx, dy, dz)))

            # pick a target for the move
            if len(above_threshold_voxel_offsets) > 0:
                # shuffle + sort (with _only_ 0-key, not lexicographic as tuples) ensures
                # randomization when there are equal top cytokine levels
                # note that numpy's shuffle will complain about ragged arrays
                shuffle(above_threshold_voxel_offsets)
                above_threshold_voxel_offsets = sorted(
                    above_threshold_voxel_offsets, key=lambda x: x[0], reverse=True
                )
                _, target_voxel_offset = above_threshold_voxel_offsets[0]
            elif len(valid_voxel_offsets) > 0:
                target_voxel_offset = choice(valid_voxel_offsets)
            else:
                raise AssertionError(
                    'This cell has no valid voxel to move to, including the one that it is in!'
                )

            # Some nonsense here, b/c jump is happening at the voxel level, not the point level
            starting_cell_point = Point(x=cell['point'][2], y=cell['point'][1], z=cell['point'][0])
            starting_cell_voxel = grid.get_voxel(starting_cell_point)
            ending_cell_voxel = grid.get_voxel(
                Point(
                    x=grid.x[cell_voxel.x + target_voxel_offset[0]],
                    y=grid.y[cell_voxel.y + target_voxel_offset[1]],
                    z=grid.z[cell_voxel.z + target_voxel_offset[2]],
                )
            )
            ending_cell_point = (
                starting_cell_point
                + grid.get_voxel_center(ending_cell_voxel)
                - grid.get_voxel_center(starting_cell_voxel)
            )

            cell['point'] = ending_cell_point
            self.update_voxel_index([cell_index])

            for i in range(0, self.len_phagosome(cell_index)):
                f_index = cell['phagosome'][i]
                fungus[f_index]['point'] = ending_cell_point
                fungus.update_voxel_index([f_index])

    def internalize_conidia(self, m_det, max_spores, p_in, grid, fungus: FungusCellList):
        for i in self.alive():
            cell = self[i]
            vox = grid.get_voxel(cell['point'])

            # Moore neighborhood, but order partially randomized. Closest to furthest order, but
            # the order of any set of points of equal distance is random
            neighborhood = list(itertools.product(tuple(range(-1 * m_det, m_det + 1)), repeat=3))
            shuffle(neighborhood)
            neighborhood = sorted(neighborhood, key=lambda v: v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

            for dx, dy, dz in neighborhood:
                zi = vox.z + dz
                yj = vox.y + dy
                xk = vox.x + dx
                if grid.is_valid_voxel(Voxel(x=xk, y=yj, z=zi)):
                    index_arr = fungus.get_cells_in_voxel(Voxel(x=xk, y=yj, z=zi))
                    for index in index_arr:
                        if (
                            fungus[index]['form'] == FungusCellData.Form.CONIDIA
                            and not fungus[index]['internalized']
                            and p_in > rg.random()
                        ):
                            fungus[index]['internalized'] = True
                            self.append_to_phagosome(i, index, max_spores)

    def damage_conidia(self, kill, t, health, fungus):
        for i in self.alive():
            cell = self[i]
            for ii in range(0, self.len_phagosome(i)):
                index = cell['phagosome'][ii]
                fungus[index]['health'] = fungus[index]['health'] - (health * (t / kill))
                if fungus[index]['dead']:
                    self.remove_from_phagosome(i, index)

    def remove_if_sporeless(self, val):
        living = self.alive()
        living_len = len(living)
        num = int(val * living_len)
        if num == 0 and living_len > 0:
            num = 1
        for _ in range(num):
            r = rg.integers(living_len)
            self.cell_data[living[r]]['dead'] = True


def cell_list_factory(self: 'MacrophageState'):
    return MacrophageCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class MacrophageState(ModuleState):
    cells: MacrophageCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    rec_r: float
    p_rec_r: float
    m_abs: float
    m_n: float
    kill: float
    m_det: int
    rec_rate_ph: int
    time_m: float
    max_conidia_in_phag: int
    p_internalization: float
    rm: float


class Macrophage(ModuleModel):
    name = 'macrophage'
    StateClass = MacrophageState

    def initialize(self, state: State):
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid

        macrophage.rec_r = self.config.getfloat('rec_r')
        macrophage.p_rec_r = self.config.getfloat('p_rec_r')
        macrophage.m_abs = self.config.getfloat('m_abs')
        macrophage.m_n = self.config.getfloat('Mn')
        macrophage.kill = self.config.getfloat('kill')
        macrophage.m_det = self.config.getint('m_det')  # radius
        macrophage.rec_rate_ph = self.config.getint('rec_rate_ph')
        macrophage.time_m = self.config.getfloat('time_m')
        macrophage.max_conidia_in_phag = self.config.getint('max_conidia_in_phag')
        macrophage.rm = self.config.getfloat('rm')
        macrophage.p_internalization = self.config.getfloat('p_internalization')
        macrophage.cells = MacrophageCellList(grid=grid)

        return state

    def advance(self, state: State, previous_time: float):
        macrophage: MacrophageState = state.macrophage
        m_cells: MacrophageCellList = macrophage.cells
        tissue = state.geometry.lung_tissue
        grid = state.grid
        cyto = state.molecules.grid['m_cyto']
        n_cyto = state.molecules.grid['n_cyto']
        fungus: FungusCellList = state.fungus.cells
        health = state.fungus.health

        # recruit new
        m_cells.recruit_new(
            macrophage.rec_rate_ph, macrophage.rec_r, macrophage.p_rec_r, tissue, grid, cyto
        )

        # absorb cytokines
        m_cells.absorb_cytokines(macrophage.m_abs, cyto, grid)

        # produce cytokines
        m_cells.produce_cytokines(macrophage.m_det, macrophage.m_n, grid, fungus, n_cyto)

        # move
        m_cells.move(macrophage.rec_r, grid, cyto, tissue, fungus)

        # internalize
        m_cells.internalize_conidia(
            macrophage.m_det,
            macrophage.max_conidia_in_phag,
            macrophage.p_internalization,
            grid,
            fungus,
        )

        # damage conidia
        m_cells.damage_conidia(macrophage.kill, macrophage.time_m, health, fungus)

        if len(fungus.alive(fungus.cell_data['form'] == FungusCellData.Form.CONIDIA)) == 0:
            m_cells.remove_if_sporeless(macrophage.rm)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        macrophage: MacrophageState = state.macrophage

        num_phagosome: int = 0
        for cell_index in macrophage.cells.alive():
            cell: MacrophageCellData = macrophage.cells[cell_index]
            num_phagosome += np.sum(cell['phagosome'] >= 0)

        return {
            'count': len(macrophage.cells.alive()),
            'phagosome': int(num_phagosome),
        }

    def visualization_data(self, state: State) -> Tuple[str, Any]:
        return 'cells', state.macrophage.cells
