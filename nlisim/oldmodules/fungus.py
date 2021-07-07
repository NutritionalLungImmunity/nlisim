from enum import IntEnum
from typing import Any, Dict, Tuple

import attr
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.module import ModuleModel, ModuleState
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import TissueType


class FungusCellData(CellData):
    class Status(IntEnum):
        DRIFTING = 99
        RESTING = 0  # CONIDIA relevant
        SWOLLEN = 1  # CONIDIA relevant
        GERMINATED = 2  # CONIDIA relevant
        GROWABLE = 3  # HYPHAE relevant
        GROWN = 4  # HYPHAE relevant
        DEAD = 5

    class Form(IntEnum):
        CONIDIA = 0
        HYPHAE = 1

    FUNGUS_FIELDS = [
        ('form', 'u1'),
        ('status', 'u1'),
        ('iteration', 'i4'),
        ('mobile', 'b1'),
        ('internalized', 'b1'),
        ('iron', 'f8'),
        ('health', 'f8'),
    ]

    dtype = np.dtype(CellData.FIELDS + FUNGUS_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        *,
        iron: float = 0,
        status: Status = Status.RESTING,
        form: Form = Form.CONIDIA,
        iteration=0,
        mobile=False,
        internalized=False,
        health=100,
        **kwargs,
    ) -> Tuple:
        return CellData.create_cell_tuple(**kwargs) + (
            form,
            status,
            iteration,
            mobile,
            internalized,
            iron,
            health,
        )


@attr.s(kw_only=True, frozen=True, repr=False)
class FungusCellList(CellList):
    CellDataClass = FungusCellData

    def iron_uptake(self, iron: np.ndarray, iron_max: float, iron_min: float, iron_absorb: float):
        """Absorb iron from external environment."""
        cells = self.cell_data
        for vox_index in np.argwhere(iron > iron_min):
            vox = Voxel(x=vox_index[2], y=vox_index[1], z=vox_index[0])

            cells_here = self.get_cells_in_voxel(vox)

            indices = []
            for index in cells_here:
                if (
                    cells[index]['form'] == FungusCellData.Form.HYPHAE.value
                    and np.invert(cells[index]['internalized'])
                    and cells[index]['iron'] < iron_max
                ):
                    indices.append(index)

            if len(indices) > 0:
                iron_split = iron_absorb * (iron[vox.z, vox.y, vox.x] / len(indices))
                for cell_index in indices:
                    cells[cell_index]['iron'] += iron_split
                    if cells[cell_index]['iron'] > iron_max:
                        cells[cell_index]['iron'] = iron_max

                iron[vox.z, vox.y, vox.x] = (1 - iron_absorb) * iron[vox.z, vox.y, vox.x]

    def spawn_hypahael_cell(self, children):
        children['status'] = FungusCellData.Status.GROWABLE
        children['form'] = FungusCellData.Form.HYPHAE
        children['mobile'] = False
        self.extend(children)

    def spawn_spores(self, points):
        n, m = points.shape

        if m != 3:
            raise ValueError('Invalid shape for a point object')

        spores = FungusCellData(n, initialize=True)
        spores['point'] = points
        spores['status'] = FungusCellData.Status.RESTING

        self.extend(spores)

    def initialize_spores(self, tissue: np.ndarray, init_num: int):
        """Initialize spores on epithelium cells."""
        grid = self.grid
        if init_num > 0:
            points = np.zeros((init_num, 3))
            indices = np.argwhere(tissue == TissueType.EPITHELIUM.value)
            if len(indices) > 0:
                rg.shuffle(indices)
                for i in range(init_num):
                    # putting in some protection for the occasional time that we place the cell on
                    # the boundary of the voxel-space
                    if indices[i][2] == grid.xv.shape[0] - 1:
                        x = grid.xv[indices[i][2]]
                    else:
                        x = rg.uniform(grid.xv[indices[i][2]], grid.xv[indices[i][2] + 1])

                    if indices[i][1] == grid.yv.shape[0] - 1:
                        y = grid.yv[indices[i][1]]
                    else:
                        y = rg.uniform(grid.yv[indices[i][1]], grid.yv[indices[i][1] + 1])

                    if indices[i][0] == grid.zv.shape[0] - 1:
                        z = grid.zv[indices[i][0]]
                    else:
                        z = rg.uniform(grid.zv[indices[i][0]], grid.zv[indices[i][0] + 1])

                    point = Point(x=x, y=y, z=z)
                    points[i] = point

                self.spawn_spores(points)

    def grow_hyphae(self, iron_min_grow, grow_time, p_branch, spacing):
        """Grow fungal hyphae."""
        cells = self.cell_data

        conidia_indices = self.alive(
            (cells['form'] == FungusCellData.Form.CONIDIA)
            & (cells['status'] == FungusCellData.Status.GERMINATED)
            & (np.invert(cells['internalized']))
        )

        hyphae_indices = self.alive(
            (cells['form'] == FungusCellData.Form.HYPHAE)
            & (cells['status'] == FungusCellData.Status.GROWABLE)
            & (np.invert(cells['internalized']))
            & (cells['iron'] > iron_min_grow)
            & (cells['iteration'] > grow_time)
        )

        # grow conidia
        if len(conidia_indices) != 0:
            cells['status'][conidia_indices] = FungusCellData.Status.GROWN
            cells['form'][conidia_indices] = FungusCellData.Form.HYPHAE
            children = FungusCellData(len(conidia_indices), initialize=True)
            children['iron'] = cells['iron'][conidia_indices]
            growth = spacing * (rg.random((len(conidia_indices), 3)) * 2 - 1)
            children['point'] = cells['point'][conidia_indices] + growth
            self.spawn_hypahael_cell(children)

        # grow hyphae
        if len(hyphae_indices) != 0:
            cells['status'][hyphae_indices] = FungusCellData.Status.GROWN
            branch_mask = rg.random(len(hyphae_indices)) < p_branch
            not_branch_indices = (np.invert(branch_mask)).nonzero()[0]
            branch_indices = branch_mask.nonzero()[0]

            elongate_children = FungusCellData(len(hyphae_indices), initialize=True)
            branch_children = FungusCellData(len(branch_indices), initialize=True)

            elongate_children['iron'] = cells['iron'][hyphae_indices] / 2
            growth = spacing * (rg.random((len(hyphae_indices), 3)) * 2 - 1)
            elongate_children['point'] = cells['point'][hyphae_indices] + growth

            if len(branch_indices) != 0:
                elongate_children['iron'][branch_indices] = (
                    cells['iron'][hyphae_indices[branch_indices]] / 3
                )

                branch_children['iron'] = cells['iron'][hyphae_indices[branch_indices]] / 3
                growth = spacing * (rg.random((len(hyphae_indices[branch_indices]), 3)) * 2 - 1)
                branch_children['point'] = cells['point'][hyphae_indices[branch_indices]] + growth

            # update iron in orignal cells
            cells['iron'][hyphae_indices[not_branch_indices]] /= 2
            cells['iron'][hyphae_indices[branch_indices]] /= 3

            self.spawn_hypahael_cell(elongate_children)
            self.spawn_hypahael_cell(branch_children)

    def age(self):
        """Add one iteration to all alive cells."""
        np.add.at(self.cell_data['iteration'], self.alive(), 1)

    def kill(self):
        """If a cell have 0 health point or out of range, kill the cell."""
        cells = self.cell_data
        mask = cells.point_mask(cells['point'], self.grid)
        indices = self.alive(np.logical_or(cells['health'] <= 0, np.invert(mask)))

        cells['status'][indices] = FungusCellData.Status.DEAD
        cells['dead'][indices] = True

    def change_status(self, p_internal_swell: float, rest_time: int, swell_time: int):
        cells = self.cell_data

        indices = self.alive(
            cells['form'] != FungusCellData.Form.HYPHAE,
        )

        internalized_indices = (cells['internalized'][indices]).nonzero()[0]
        not_internalized_indices = (np.invert(cells['internalized'][indices])).nonzero()[0]

        internalized_rest_indices = np.logical_and(
            cells['status'][internalized_indices] == FungusCellData.Status.RESTING,
            cells['iteration'][internalized_indices] >= rest_time,
        ).nonzero()[0]

        internalized_swollen_indices = np.logical_and(
            cells['status'][internalized_indices] == FungusCellData.Status.SWOLLEN,
            cells['iteration'][internalized_indices] >= swell_time,
        ).nonzero()[0]

        # internal fungus with REST status
        swall_mask = rg.random(len(internalized_rest_indices)) < p_internal_swell
        internalized_rest_indices = swall_mask.nonzero()[0]

        cells['status'][internalized_rest_indices] = FungusCellData.Status.SWOLLEN
        cells['iteration'][internalized_rest_indices] = 0

        # internal fungus with SWOLLEN status
        cells['status'][internalized_swollen_indices] = FungusCellData.Status.GERMINATED
        cells['iteration'][internalized_swollen_indices] = 0

        rest_indices = np.logical_and(
            cells['status'][not_internalized_indices] == FungusCellData.Status.RESTING,
            cells['iteration'][not_internalized_indices] >= rest_time,
        ).nonzero()[0]
        swollen_indices = np.logical_and(
            cells['status'][not_internalized_indices] == FungusCellData.Status.SWOLLEN,
            cells['iteration'][not_internalized_indices] >= swell_time,
        ).nonzero()[0]

        # free fungus with REST status
        cells['status'][rest_indices] = FungusCellData.Status.SWOLLEN
        cells['iteration'][rest_indices] = 0

        # free fungus with SWOLLEN status
        cells['status'][swollen_indices] = FungusCellData.Status.GERMINATED
        cells['iteration'][swollen_indices] = 0


def cell_list_factory(self: 'FungusState'):
    return FungusCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class FungusState(ModuleState):
    cells: FungusCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    # init_num: int = 0
    # p_lodge: float = 0
    # p_internal_swell: float = 0.05
    # iron_min: int = 0
    # iron_max: float = 0.0
    # iron_absorb: float = 0.0
    # spacing: float = 0.0
    # iron_min_grow: float = 0.0
    # grow_time: float = 0.0
    # p_branch: float = 0.0
    # p_internalize: float = 0.0
    health: float = 100.0


class Fungus(ModuleModel):
    name = 'fungus'
    StateClass = FungusState

    def initialize(self, state: State):
        fungus: FungusState = state.fungus
        # grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        self.init_num = self.config.getint('init_num')
        self.p_lodge = self.config.getfloat('p_lodge')
        self.p_internal_swell = self.config.getfloat('p_internal_swell')
        self.iron_min = self.config.getint('iron_min')
        self.iron_max = self.config.getfloat('iron_max')
        self.iron_absorb = self.config.getfloat('iron_absorb')
        self.spacing = self.config.getfloat('spacing')
        self.iron_min_grow = self.config.getfloat('iron_min_grow')
        self.p_branch = self.config.getfloat('p_branch')
        self.p_internalize = self.config.getfloat('p_internalize')
        self.rest_time = self.config.getint('rest_time')
        self.swell_time = self.config.getint('swell_time')
        self.grow_time = self.config.getint('grow_time')

        fungus.health = self.config.getfloat('init_health')

        cells = fungus.cells
        cells.initialize_spores(tissue, self.init_num)

        return state

    def advance(self, state: State, previous_time: float):
        cells = state.fungus.cells

        cells.kill()  # clear dead cell
        cells.age()
        cells.change_status(self.p_internal_swell, self.rest_time, self.swell_time)
        if hasattr(state, 'molecules'):
            iron = state.molecules.grid['iron']
            cells.iron_uptake(iron, self.iron_max, self.iron_min, self.iron_absorb)
        cells.grow_hyphae(self.iron_min_grow, self.grow_time, self.p_branch, self.spacing)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        fungus: FungusState = state.fungus

        num_conidia: int = 0
        num_hyphae: int = 0
        total_iron: float = 0.0
        for cell_index in fungus.cells.alive():
            cell: FungusCellData = fungus.cells[cell_index]
            if cell['form'] == FungusCellData.Form.HYPHAE:
                num_hyphae += 1
            elif cell['form'] == FungusCellData.Form.CONIDIA:
                num_conidia += 1
            total_iron += cell['iron']

        return {
            'count': len(fungus.cells.alive()),
            'conidia': int(num_conidia),
            'hyphae': int(num_hyphae),
            'total_iron': float(total_iron),
        }

    def visualization_data(self, state: State) -> Tuple[str, Any]:
        return 'cells', state.fungus.cells
