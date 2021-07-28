from enum import IntEnum
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


class NeutrophilCellData(CellData):
    class Status(IntEnum):
        NONGRANULATING = 0
        GRANULATING = 1

    NEUTROPHIL_FIELDS = [('status', 'u1'), ('iteration', 'i4'), ('granule_count', 'i4')]

    dtype = np.dtype(CellData.FIELDS + NEUTROPHIL_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        *,
        status=Status.NONGRANULATING,
        granule_count=0,
        **kwargs,
    ) -> Tuple:
        iteration = 0
        return CellData.create_cell_tuple(**kwargs) + (
            status,
            iteration,
            granule_count,
        )


@attr.s(kw_only=True, frozen=True, repr=False)
class NeutrophilCellList(CellList):
    CellDataClass = NeutrophilCellData

    def recruit_new(self, rec_rate_ph, rec_r, granule_count, neutropenic, time, grid, tissue, cyto):
        num_reps = 0
        if not neutropenic:
            num_reps = rec_rate_ph  # number of neutrophils recruited per time step
        elif neutropenic and 48 <= time <= 96:
            # TODO: relate 3 to Algorithm S3.14 and rec_rate_ph or introduce neutropenic parameter
            #  In S3.14: num_reps = 6 and int( (time-48)/ 8), both are 1/3 values here
            num_reps = int((time - 48) / 8) * 3

        if num_reps <= 0:
            return

        cyto_index = np.argwhere(np.logical_and(tissue == TissueType.BLOOD.value, cyto >= rec_r))
        if len(cyto_index) <= 0:
            # nowhere to place cells
            return

        for _ in range(num_reps):
            ii = rg.integers(cyto_index.shape[0])
            point = Point(
                x=grid.x[cyto_index[ii, 2]],
                y=grid.y[cyto_index[ii, 1]],
                z=grid.z[cyto_index[ii, 0]],
            )

            self.append(
                NeutrophilCellData.create_cell(
                    point=point,
                    status=NeutrophilCellData.Status.NONGRANULATING,
                    granule_count=granule_count,
                )
            )

    def absorb_cytokines(self, n_absorb, cyto, grid):
        for index in self.alive():
            vox = grid.get_voxel(self[index]['point'])
            x = vox.x
            y = vox.y
            z = vox.z
            cyto[z, y, x] = (1 - n_absorb) * cyto[z, y, x]

    def produce_cytokines(self, n_det, n_n, grid, fungus: FungusCellList, cyto):
        for i in self.alive():
            vox = grid.get_voxel(self[i]['point'])

            hyphae_count = 0

            # Moore neighborhood
            neighborhood = tuple(itertools.product(tuple(range(-1 * n_det, n_det + 1)), repeat=3))

            for dx, dy, dz in neighborhood:
                zi = vox.z + dz
                yj = vox.y + dy
                xk = vox.x + dx
                if grid.is_valid_voxel(Voxel(x=xk, y=yj, z=zi)):
                    index_arr = fungus.get_cells_in_voxel(Voxel(x=xk, y=yj, z=zi))
                    for index in index_arr:
                        if fungus[index]['form'] == FungusCellData.Form.HYPHAE:
                            hyphae_count += 1

            cyto[vox.z, vox.y, vox.x] = cyto[vox.z, vox.y, vox.x] + (n_n * hyphae_count)

    def move(self, rec_r, grid, cyto, tissue):
        for cell_index in self.alive(
            self.cell_data['status'] == NeutrophilCellData.Status.NONGRANULATING
        ):
            # TODO: Algorithm S3.17 says "if degranulating nearby hyphae, do not move" but do
            #  we have the "nearby hyphae" part of this condition?
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

    def damage_hyphae(self, n_det, n_kill, time, health, grid, fungus: FungusCellList, iron):
        for i in self.alive(self.cell_data['granule_count'] > 0):
            cell = self[i]
            vox = grid.get_voxel(cell['point'])

            # Moore neighborhood, but order partially randomized. Closest to furthest order, but
            # the order of any set of points of equal distance is random
            neighborhood = list(itertools.product(tuple(range(-1 * n_det, n_det + 1)), repeat=3))
            shuffle(neighborhood)
            neighborhood = sorted(neighborhood, key=lambda v: v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

            for dx, dy, dz in neighborhood:
                zi = vox.z + dz
                yj = vox.y + dy
                xk = vox.x + dx
                if grid.is_valid_voxel(Voxel(x=xk, y=yj, z=zi)):
                    index_arr = fungus.get_cells_in_voxel(Voxel(x=xk, y=yj, z=zi))
                    if len(index_arr) > 0:
                        iron[zi, yj, xk] = 0
                    for index in index_arr:
                        if (
                            fungus[index]['form'] == FungusCellData.Form.HYPHAE
                            and cell['granule_count'] > 0
                        ):
                            fungus[index]['health'] -= health * (time / n_kill)
                            cell['granule_count'] -= 1
                            cell['status'] = NeutrophilCellData.Status.GRANULATING
                        elif cell['granule_count'] == 0:
                            cell['status'] = NeutrophilCellData.Status.NONGRANULATING
                            break

    def update(self):
        for i in self.alive(self.cell_data['granule_count'] == 0):
            self[i]['status'] = NeutrophilCellData.Status.NONGRANULATING

    def age(self):
        self.cell_data['iteration'] += 1

    def kill_by_age(self, age_limit):
        for i in self.alive(self.cell_data['iteration'] > age_limit):
            self[i]['dead'] = True


def cell_list_factory(self: 'NeutrophilState'):
    return NeutrophilCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class NeutrophilState(ModuleState):
    cells: NeutrophilCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    neutropenic: bool
    rec_rate_ph: int
    rec_r: float
    n_absorb: float
    n_n: float
    n_det: int
    granule_count: int
    n_kill: float
    time_n: float
    age_limit: int


class Neutrophil(ModuleModel):
    name = 'neutrophil'

    StateClass = NeutrophilState

    def initialize(self, state: State):
        neutrophil: NeutrophilState = state.neutrophil
        grid: RectangularGrid = state.grid

        neutrophil.neutropenic = self.config.getboolean('neutropenic')
        neutrophil.rec_rate_ph = self.config.getint('rec_rate_ph')
        neutrophil.rec_r = self.config.getfloat('rec_r')
        neutrophil.n_absorb = self.config.getfloat('n_absorb')
        neutrophil.n_n = self.config.getfloat('Nn')
        neutrophil.n_det = self.config.getint('n_det')
        neutrophil.granule_count = self.config.getint('granule_count')
        neutrophil.n_kill = self.config.getfloat('n_kill')
        neutrophil.time_n = self.config.getfloat('time_n')
        neutrophil.age_limit = self.config.getint('age_limit')

        neutrophil.cells = NeutrophilCellList(grid=grid)

        return state

    def advance(self, state: State, previous_time: float):
        neutrophil: NeutrophilState = state.neutrophil
        n_cells = neutrophil.cells
        fungus = state.fungus.cells
        health = state.fungus.health

        tissue = state.geometry.lung_tissue
        grid = state.grid
        cyto = state.molecules.grid['n_cyto']
        iron = state.molecules.grid['iron']

        # recruit new
        n_cells.recruit_new(
            neutrophil.rec_rate_ph,
            neutrophil.rec_r,
            neutrophil.granule_count,
            neutrophil.neutropenic,
            previous_time,
            grid,
            tissue,
            cyto,
        )

        # absorb cytokines
        n_cells.absorb_cytokines(neutrophil.n_absorb, cyto, grid)

        # produce cytokines
        n_cells.produce_cytokines(neutrophil.n_det, neutrophil.n_n, grid, fungus, cyto)

        # move
        n_cells.move(neutrophil.rec_r, grid, cyto, tissue)

        n_cells.damage_hyphae(
            neutrophil.n_det, neutrophil.n_kill, neutrophil.time_n, health, grid, fungus, iron
        )

        # update granule == 0 status
        n_cells.update()

        n_cells.age()

        n_cells.kill_by_age(neutrophil.age_limit)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        neutrophil: NeutrophilState = state.neutrophil

        return {
            'count': len(neutrophil.cells.alive()),
            'granules': int(neutrophil.granule_count),
        }

    def visualization_data(self, state: State) -> Tuple[str, Any]:
        return 'cells', state.neutrophil.cells
