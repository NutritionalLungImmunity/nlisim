from enum import IntEnum
import random

import attr
import numpy as np

from simulation.cell import CellData, CellList
from simulation.coordinates import Point, Voxel
from simulation.grid import RectangularGrid
from simulation.module import Module, ModuleState
from simulation.modules.geometry import TissueTypes
from simulation.state import State


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
        status: Status = Status.DRIFTING,
        form: Form = Form.CONIDIA,
        iteration=0,
        mobile=True,
        internalized=False,
        health=100,
        **kwargs,
    ) -> np.record:

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

    def spawn_hypahael_cell(self, point, iron, spacing):
        new_x = point[2] + spacing * random.uniform(-1, 1)
        new_y = point[1] + spacing * random.uniform(-1, 1)
        new_z = point[0] + spacing * random.uniform(-1, 1)

        new_point = Point(x=new_x, y=new_y, z=new_z,)

        self.append(
            FungusCellData.create_cell(
                point=new_point,
                status=FungusCellData.Status.GROWABLE,
                form=FungusCellData.Form.HYPHAE,
                iron=iron,
                mobile=False,
            )
        )


    def iron_uptake(self, iron: np.ndarray, iron_max: float, iron_min: float, iron_absorb: float):
        ''' Absorb iron from external environment '''
        
        # alive cells
        #cells = self.cells[self.alive()]
        #vox_index = cells.voxel_index

        # qualified voxel
        #vox = iron[iron > iron_min]
        cells = self.cell_data
        for vox_index in np.argwhere(iron > iron_min):
            vox = Voxel(x=vox_index[2], y=vox_index[1], z=vox_index[0])

            cells_here = self.get_cells_in_voxel(vox)

            indices = []
            for index in cells_here:
                if (
                    cells[index]['form'] == FungusCellData.Form.HYPHAE.value
                    and cells[index]['internalized'] == False
                    and cells[index]['iron'] < iron_max
                ):
                    indices.append(index)
                    print(index)

            if len(indices) > 0:
                iron_split = iron_absorb * (iron[vox.z, vox.y, vox.x] / len(indices))
                for cell_index in indices:
                    cells[cell_index]['iron'] += iron_split
                    if cells[cell_index]['iron'] > iron_max:
                        cells[cell_index]['iron'] = iron_max

                iron[vox.z, vox.y, vox.x] = (1 - iron_absorb) * iron[vox.z, vox.y, vox.x]


    def grow_hyphae(self, min_grow, grow_time, p_branch, spacing):

        # spawn from hyphae from conidia
        for index in self.alive():
            cell = self.cell_data[index]

            # spawn from hyphae from conidia
            if (
                cell['form'] == FungusCellData.Form.CONIDIA
                and cell['status'] == FungusCellData.Status.GERMINATED
                and cell['internalized'] is False
            ):

                cell['status'] = FungusCellData.Status.GROWN
                spawn_hypahael_cell(cells, cell['point'], cell['iron'], spacing)

            # grow hyphae
            elif (
                cell['form'] == FungusCellData.Form.HYPHAE
                and cell['status'] == FungusCellData.Status.GROWABLE
                and cell['internalized'] is False
                and cell['iron'] > iron_min_grow
                and cell['iteration'] > grow_time
            ):

                if p_branch > random.random():
                    iron_f = cell['iron'] / 3
                    spawn_hypahael_cell(cell['point'], iron_f, spacing)
                    spawn_hypahael_cell(cell['point'], iron_f, spacing)
                else:
                    iron_f = cell['iron'] / 2
                    spawn_hypahael_cell(cell['point'], iron_f, spacing)

    def age(self):
        '''Add one iteration to all alive cells.'''
        np.add.at(self.cell_data['iteration'], self.alive(), 1)

    def kill(self):
        '''If a cell have 0 health point or out of range, kill the cell.'''
        cells = self.cell_data
        mask = cells.point_mask(cells['point'], self.grid)
        indices = self.alive(np.logical_or(cells['health'] <= 0, mask != True))
        
        cells['status'][indices] = FungusCellData.Status.DEAD
        cells['dead'][indices] = True

    def change_status(self, iter_to_change_status):
        cells = self.cell_data

        indices = self.alive(np.logical_and(
            cells['iteration'] >= iter_to_change_status,
            cells['form'] != FungusCellData.Form.HYPHAE
        ))

        rest_indices = (cells['status'][indices] == FungusCellData.Status.RESTING).nonzero()[0]
        swollen_indices = (cells['status'][indices] == FungusCellData.Status.SWOLLEN).nonzero()[0]

        cells['status'][rest_indices] = FungusCellData.Status.SWOLLEN
        cells['iteration'][rest_indices] = 0
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
    # ITER_TO_CHANGE_STATUS: int = 1
    # iron_min: int = 0
    # iron_max: float = 0.0
    # iron_absorb: float = 0.0
    # spacing: float = 0.0
    # iron_min_grow: float = 0.0
    # grow_time: float = 0.0
    # p_branch: float = 0.0
    # p_internalize: float = 0.0


class Fungus(Module):
    name = 'fungus'
    StateClass = FungusState

    # defaults config opt
    defaults = {
        'init_num': '10',
        'p_lodge': '0.1',
        'p_internal_swell': '0.05',
        'ITER_TO_CHANGE_STATUS': '1',
        'iron_min': '10',
        'iron_absorb': '1',
        'spacing': '0.1',
        'iron_min_grow': '5',
        'grow_time': '2',
        'p_branch': '0.2',
        'p_internalize': '0.1'
        # ...
    }

    def initialize(self, state: State):
        fungus: FungusState = state.fungus
        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        
        self.init_num = self.config.getint('init_num')
        self.p_lodge = self.config.getfloat('p_lodge')
        self.p_internal_swell = self.config.getfloat('p_internal_swell')
        self.ITER_TO_CHANGE_STATUS = self.config.getint('ITER_TO_CHANGE_STATUS')
        self.iron_min = self.config.getint('iron_min')
        self.iron_max = self.config.getfloat('iron_max')
        self.iron_absorb = self.config.getfloat('iron_absorb')
        self.spacing = self.config.getfloat('spacing')
        self.iron_min_grow = self.config.getfloat('iron_min_grow')
        self.grow_time = self.config.getfloat('grow_time')
        self.p_branch = self.config.getfloat('p_branch')
        self.p_internalize = self.config.getfloat('p_internalize')
        

        fungus.cells = FungusCellList(grid=grid)
        if self.init_num > 0:
            # initialize the surfactant layer with some fungus in random locations
            indices = np.argwhere(tissue == TissueTypes.EPITHELIUM.value)
            if len(indices) > 0:
                np.random.shuffle(indices)

                for i in range(0, fungus.init_num):
                    x = random.uniform(grid.xv[indices[i][2]], grid.xv[indices[i][2] + 1])
                    y = random.uniform(grid.yv[indices[i][1]], grid.yv[indices[i][1] + 1])
                    z = random.uniform(grid.zv[indices[i][0]], grid.zv[indices[i][0] + 1])

                    point = Point(x=x, y=y, z=z)
                    status = FungusCellData.Status.DRIFTING

                    fungus.cells.append(FungusCellData.create_cell(point=point, status=status))

        return state

    def advance(self, state: State, previous_time: float):

        update(state)
        iron_uptake(state)
        grow_hyphae(state)

        return state


def update(state):
    fungus: FungusState = state.fungus
    grid: RectangularGrid = state.grid
    tissue = state.geometry.lung_tissue
    cells = fungus.cells

    for index in cells.alive(cells.cell_data['form'] != FungusCellData.Form.HYPHAE):
        cell = cells[index]
        vox = grid.get_voxel(cell['point'])

        if cell['health'] <= 0:
            cell['status'] = FungusCellData.Status.DEAD
            cell['dead'] = True

        if cell['mobile']:
            # check if at the edge of space
            if vox.x > Voxel(x=grid.xv[-1], y=grid.yv[0], z=grid.zv[0]).x:
                cell['status'] = FungusCellData.Status.DEAD
                cell['dead'] = True
            else:
                # move
                print('cells move')

            # check contact and lodging with epithelium
            if tissue[vox.z, vox.y, vox.x] == TissueTypes.EPITHELIUM.value:
                if fungus.p_lodge > random.random():
                    cell['mobile'] = False
                    cell['status'] = FungusCellData.Status.RESTING
                    cell['iteration'] += 1
                    if fungus.p_internalize < random.random():
                        cell['internalized'] = True
        else:
            cell['iteration'] += 1

        # change status
        if cell['iteration'] >= fungus.ITER_TO_CHANGE_STATUS:
            if cell['internalized']:
                if cell['status'] == FungusCellData.Status.RESTING:
                    if fungus.p_internal_swell > random.random():
                        cell['status'] = FungusCellData.Status.SWOLLEN
                        cell['iteration'] = 0
                elif cell['status'] == FungusCellData.Status.SWOLLEN:
                    cell['status'] = FungusCellData.Status.GERMINATED
                    cell['iteration'] = 0
            else:
                if cell['status'] == FungusCellData.Status.RESTING:
                    cell['status'] = FungusCellData.Status.SWOLLEN
                    cell['iteration'] = 0
                elif cell['status'] == FungusCellData.Status.SWOLLEN:
                    cell['status'] = FungusCellData.Status.GERMINATED
                    cell['iteration'] = 0
                    # cell['form'] = HYphae?

            # TODO - check ODD protocol
            if cell['status'] == FungusCellData.Status.SWOLLEN:
                cell['iteration'] = 0

    for index in cells.alive(cells.cell_data['form'] == FungusCellData.Form.HYPHAE):
        cells[index]['iteration'] += 1


def iron_uptake(state):
    fungus: FungusState = state.fungus
    cells = fungus.cells
    iron = state.molecules.grid['iron']

    for vox_index in np.argwhere(iron > fungus.iron_min):
        vox = Voxel(x=vox_index[2], y=vox_index[1], z=vox_index[0])

        cells_here = cells.get_cells_in_voxel(vox)
        indices = []
        for index in cells_here:
            if (
                cells.cell_data[index]['form'] == FungusCellData.Form.HYPHAE
                and cells.cell_data[index]['internalized'] is False
                and cells.cell_data[index]['iron'] < fungus.iron_max
            ):
                indices.append(index)

        if len(indices) > 0:
            iron_split = fungus.iron_absorb * (iron[vox.z, vox.y, vox.x] / len(indices))
            for cell_index in indices:
                cells[cell_index]['iron'] += iron_split
                if cells[cell_index]['iron'] > fungus.iron_max:
                    cells[cell_index]['iron'] = fungus.iron_max

            iron[vox.z, vox.y, vox.x] = (1 - fungus.iron_absorb) * iron[vox.z, vox.y, vox.x]


def grow_hyphae(state):
    fungus: FungusState = state.fungus
    grid: RectangularGrid = state.grid
    cells = fungus.cells

    # spawn from hyphae from conidia
    for index in cells.alive():
        cell = cells.cell_data[index]

        if (
            cell['form'] == FungusCellData.Form.CONIDIA
            and cell['status'] == FungusCellData.Status.GERMINATED
            and cell['internalized'] is False
        ):

            cell['status'] = FungusCellData.Status.GROWN
            spawn_hypahael_cell(cells, cell['point'], cell['iron'], fungus.spacing, grid)

    # grow hyphae
    for index in cells.alive():
        cell = cells.cell_data[index]

        if (
            cell['form'] == FungusCellData.Form.HYPHAE
            and cell['status'] == FungusCellData.Status.GROWABLE
            and cell['internalized'] is False
            and cell['iron'] > fungus.iron_min_grow
            and cell['iteration'] > fungus.grow_time
        ):

            if fungus.p_branch > random.random():
                iron_f = cell['iron'] / 3
                spawn_hypahael_cell(cells, cell['point'], iron_f, fungus.spacing, grid)
                spawn_hypahael_cell(cells, cell['point'], iron_f, fungus.spacing, grid)
            else:
                iron_f = cell['iron'] / 2
                spawn_hypahael_cell(cells, cell['point'], iron_f, fungus.spacing, grid)


def spawn_hypahael_cell(cells, point, iron, spacing, grid):
    new_x = 0
    new_y = 0
    new_z = 0

    if point[2] > grid.x[int(len(grid.x) / 2)]:
        new_x = point[2] + spacing * random.uniform(0, 1)
    else:
        new_x = point[2] + spacing * random.uniform(-1, 0)

    if point[1] > grid.y[int(len(grid.y) / 2)]:
        new_y = point[1] + spacing * random.uniform(0, 1)
    else:
        new_y = point[1] + spacing * random.uniform(-1, 0)

    if point[0] > grid.z[int(len(grid.z) / 2)]:
        new_z = point[0] + spacing * random.uniform(0, 1)
    else:
        new_z = point[0] + spacing * random.uniform(-1, 0)

    new_point = Point(x=new_x, y=new_y, z=new_z,)

    cells.append(
        FungusCellData.create_cell(
            point=new_point,
            status=FungusCellData.Status.GROWABLE,
            form=FungusCellData.Form.HYPHAE,
            iron=iron,
            mobile=False,
        )
    )
