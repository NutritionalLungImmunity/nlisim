import logging
import math
from typing import Any, Dict, Optional, Tuple

# noinspection PyPackageRequirements
import attr

# noinspection PyPackageRequirements
from attr import attrs

# noinspection PyPackageRequirements
import numpy as np

from nlisim.cell import CellData, CellFields, CellList
from nlisim.coordinates import Point
from nlisim.grid import TetrahedralMesh, TissueType
from nlisim.modules.phagocyte import (
    PhagocyteCellData,
    PhagocyteModel,
    PhagocyteModuleState,
    PhagocyteState,
    PhagocyteStatus,
)
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import GridTissueType, sample_point_from_simplex, tetrahedral_gradient


class MacrophageCellData(PhagocyteCellData):
    MACROPHAGE_FIELDS: CellFields = [
        ('status', np.uint8),
        ('state', np.uint8),
        ('fpn', bool),
        ('fpn_iteration', np.int64),
        ('tf', bool),  # TODO: descriptive name, transferrin?
        ('tnfa', bool),
        ('iron_pool', np.float64),  # units: atto-mols
        ('status_iteration', np.uint64),
        ('velocity', np.float64, 3),
    ]

    dtype = np.dtype(
        CellData.FIELDS + PhagocyteCellData.PHAGOCYTE_FIELDS + MACROPHAGE_FIELDS, align=True
    )  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        initializer = {
            'status': kwargs.get('status', PhagocyteStatus.RESTING),
            'state': kwargs.get('state', PhagocyteState.FREE),
            'fpn': kwargs.get('fpn', True),
            'fpn_iteration': kwargs.get('fpn_iteration', 0),
            'tf': kwargs.get('tf', False),
            'tnfa': kwargs.get('tnfa', False),
            'iron_pool': kwargs.get('iron_pool', 0.0),
            'status_iteration': kwargs.get('status_iteration', 0),
            'velocity': kwargs.get('root', np.zeros(3, dtype=np.float64)),
        }

        # ensure that these come in the correct order
        return PhagocyteCellData.create_cell_tuple(**kwargs) + tuple(
            [initializer[key] for key, *_ in MacrophageCellData.MACROPHAGE_FIELDS]
        )


@attrs(kw_only=True, frozen=True, repr=False)
class MacrophageCellList(CellList):
    CellDataClass = MacrophageCellData


def cell_list_factory(self: 'MacrophageState') -> MacrophageCellList:
    return MacrophageCellList(mesh=self.global_state.mesh)


@attr.s(kw_only=True)
class MacrophageState(PhagocyteModuleState):
    cells: MacrophageCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    time_to_rest: float  # units: min
    iter_to_rest: int  # units: steps
    time_to_change_state: float  # units: hours
    iter_to_change_state: int  # units: steps
    ma_internal_iron: float  # units: atto-mols
    prob_death_per_timestep: float  # units: probability * step^-1
    max_ma: int  # units: count
    min_ma: int  # units: count
    init_num_macrophages: int  # units: count
    recruitment_rate: float
    drift_bias: float
    ma_move_rate_act: float  # µm/min
    ma_move_rate_rest: float  # µm/min
    half_life: float  # units: hours
    # UNUSED:
    # kd_ma_iron: float
    # ma_vol: float  # units: pL


class Macrophage(PhagocyteModel):
    name = 'macrophage'
    StateClass = MacrophageState

    def initialize(self, state: State):
        logging.getLogger('nlisim').debug("Initializing " + self.name)
        macrophage: MacrophageState = state.macrophage
        time_step_size: float = self.time_step
        mesh: TetrahedralMesh = state.mesh

        macrophage.max_conidia = self.config.getint(
            'max_conidia'
        )  # (from phagocyte model) units: count
        macrophage.time_to_rest = self.config.getint('time_to_rest')  # units: min
        macrophage.time_to_change_state = self.config.getint('time_to_change_state')  # units: hours
        macrophage.ma_internal_iron = self.config.getfloat('ma_internal_iron')  # units: atto-mols

        macrophage.max_ma = self.config.getint('max_ma')  # units: count
        macrophage.min_ma = self.config.getint('min_ma')  # units: count
        macrophage.init_num_macrophages = self.config.getint('init_num_macrophages')  # units: count

        macrophage.recruitment_rate = self.config.getfloat('recruitment_rate')
        macrophage.drift_bias = self.config.getfloat('drift_bias')

        macrophage.ma_move_rate_act = self.config.getfloat('ma_move_rate_act')  # µm/min
        macrophage.ma_move_rate_rest = self.config.getfloat('ma_move_rate_rest')  # µm/min

        macrophage.half_life = self.config.getfloat('ma_half_life')  # units: hours

        # UNUSED:
        # macrophage.kd_ma_iron = self.config.getfloat('kd_ma_iron')
        # macrophage.ma_vol = self.config.getfloat('ma_vol')

        # computed values
        macrophage.iter_to_rest = int(
            macrophage.time_to_rest / self.time_step
        )  # units: min / (min/step) = steps
        macrophage.iter_to_change_state = int(
            macrophage.time_to_change_state * (60 / time_step_size)
        )  # units: hours * (min/hour) / (min/step) = step

        macrophage.prob_death_per_timestep = -math.log(0.5) / (
            macrophage.half_life * (60 / time_step_size)
        )  # units: 1/(  hours * (min/hour) / (min/step)  ) = 1/step

        # initialize macrophages cells. Cells will be distributed into non-air layers, in a
        # uniformly random manner.
        init_num_macrophages = self.config.getint('init_num_macrophages')
        locations = np.where(mesh.element_tissue_type != GridTissueType.AIR)[0]

        # define the cumulative distribution function so that elements are selected proportionally
        # to their volumes
        volumes = mesh.element_volumes[locations]
        cdf = np.cumsum(volumes)
        cdf /= cdf[-1]
        macrophage_elements = locations[
            np.argmax(np.random.random((init_num_macrophages, 1)) < cdf, axis=1)
        ]

        # the cell's points are then superpositions of the tetrahedral vertices weighted by randomly
        # generated simplex coordinates
        simplex_coords = sample_point_from_simplex(num_points=init_num_macrophages)
        points = np.einsum(
            'ijk,ji->ik',
            mesh.points[mesh.element_point_indices[macrophage_elements]],
            simplex_coords,
        )

        for element_index, point in zip(macrophage_elements, points):
            self.create_macrophage(
                state=state,
                x=point[2],
                y=point[1],
                z=point[0],
                iron_pool=macrophage.ma_internal_iron,
                element_index=element_index,
            )

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        macrophage: MacrophageState = state.macrophage

        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell = macrophage.cells[macrophage_cell_index]

            num_cells_in_phagosome = np.sum(macrophage_cell['phagosome'] >= 0)

            self.update_status(state, macrophage_cell, num_cells_in_phagosome)

            if (
                num_cells_in_phagosome == 0
                and rg.uniform() < macrophage.prob_death_per_timestep
                and len(macrophage.cells.alive()) > macrophage.min_ma
            ):
                macrophage_cell['status'] = PhagocyteStatus.DEAD
                macrophage_cell['dead'] = True

            if not macrophage_cell['fpn']:
                if macrophage_cell['fpn_iteration'] >= macrophage.iter_to_change_state:
                    macrophage_cell['fpn_iteration'] = 0
                    macrophage_cell['fpn'] = True
                else:
                    macrophage_cell['fpn_iteration'] += 1

            # Movement
            if macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
                max_move_step = (
                    macrophage.ma_move_rate_act * self.time_step
                )  # (µm/min) * (min/step) = µm * step
            else:
                max_move_step = (
                    macrophage.ma_move_rate_rest * self.time_step
                )  # (µm/min) * (min/step) = µm * step
            move_step: int = rg.poisson(max_move_step)
            # move the cell 1 µm, move_step number of times
            for _ in range(move_step):
                self.single_step_move(
                    state, macrophage_cell, macrophage_cell_index, macrophage.cells
                )

        # Recruitment
        self.recruit_macrophages(state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        macrophage: MacrophageState = state.macrophage
        live_macrophages = macrophage.cells.alive()

        max_index = max(map(int, PhagocyteStatus))
        status_counts = np.bincount(
            np.fromiter(
                (
                    macrophage.cells[macrophage_cell_index]['status']
                    for macrophage_cell_index in live_macrophages
                ),
                dtype=np.uint8,
            ),
            minlength=max_index + 1,
        )

        tnfa_active = int(
            np.sum(
                np.fromiter(
                    (
                        macrophage.cells[macrophage_cell_index]['tnfa']
                        for macrophage_cell_index in live_macrophages
                    ),
                    dtype=bool,
                )
            )
        )

        return {
            'count': len(live_macrophages),
            'inactive': int(status_counts[PhagocyteStatus.INACTIVE]),
            'inactivating': int(status_counts[PhagocyteStatus.INACTIVATING]),
            'resting': int(status_counts[PhagocyteStatus.RESTING]),
            'activating': int(status_counts[PhagocyteStatus.ACTIVATING]),
            'active': int(status_counts[PhagocyteStatus.ACTIVE]),
            'apoptotic': int(status_counts[PhagocyteStatus.APOPTOTIC]),
            'necrotic': int(status_counts[PhagocyteStatus.NECROTIC]),
            'anergic': int(status_counts[PhagocyteStatus.ANERGIC]),
            'interacting': int(status_counts[PhagocyteStatus.INTERACTING]),
            'TNFa active': tnfa_active,
        }

    def visualization_data(self, state: State):
        return 'cells', state.macrophage.cells

    def recruit_macrophages(self, state: State) -> None:
        """
        Recruit macrophages based on MIP1b activation

        Parameters
        ----------
        state : State
            global simulation state

        Returns
        -------
        nothing
        """
        from nlisim.modules.mip1b import MIP1BState
        from nlisim.util import activation_function

        macrophage: MacrophageState = state.macrophage
        mip1b: MIP1BState = state.mip1b
        mesh: TetrahedralMesh = state.mesh

        # 1. compute number of macrophages to recruit
        num_live_macrophages = len(macrophage.cells.alive())
        avg = (
            macrophage.recruitment_rate
            * (mesh.integrate_point_function(mip1b.field) / mesh.total_volume)
            * (1 - num_live_macrophages / macrophage.max_ma)
            / mip1b.k_d
        )
        number_to_recruit = max(
            rg.poisson(avg) if avg > 0 else 0, macrophage.min_ma - num_live_macrophages
        )
        # 2. get voxels for new macrophages, based on activation
        if number_to_recruit > 0:
            activated_points = np.where(
                activation_function(
                    x=mip1b.field,
                    k_d=mip1b.k_d,
                    h=self.time_step / 60,
                    volume=1.0,  # mip1b.field already in concentration (M) units
                    b=1.0,
                )
                < rg.uniform(size=mip1b.field.shape)
            )
            activated_elements = mesh.elements_incident_to(points=activated_points)
            for element_index in rg.choice(
                activated_elements, size=number_to_recruit, replace=True
            ):
                simplex_coords = sample_point_from_simplex()
                point = mesh.points[mesh.element_point_indices[element_index]] @ simplex_coords
                self.create_macrophage(
                    state=state,
                    x=point[2],
                    y=point[1],
                    z=point[0],
                    iron_pool=macrophage.ma_internal_iron,
                )

    def single_step_probabilistic_drift(
        self, state: State, cell: PhagocyteCellData, element_index: int
    ) -> Tuple[Point, int]:
        """
        Calculate a 1µm movement of a macrophage

        Parameters
        ----------
        state : State
            global simulation state
        cell : MacrophageCellData
            a macrophage cell
        element_index : int
            index of the current tetrahedral element occupied by the macrophage

        Returns
        -------
        Point
            the new position of the macrophage
        """
        # macrophages are attracted by MIP1b
        from nlisim.modules.mip1b import MIP1BState
        from nlisim.util import activation_function

        mip1b: MIP1BState = state.mip1b
        macrophage: MacrophageState = state.macrophage
        mesh: TetrahedralMesh = state.mesh

        # compute chemokine influence on velocity, with some randomness.
        # macrophage has a non-zero probability of moving into non-air voxels.
        # if not any of these, stay in place. This could happen if e.g. you are
        # somehow stranded in air.
        chemokine_levels = mip1b.field[mesh.element_point_indices[element_index]]
        weights = activation_function(
            x=chemokine_levels,
            k_d=mip1b.k_d,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            volume=1.0,  # already in concentration (M)
            b=1.0,
        )

        # movement tends toward the gradient direction, with some randomization
        dp_dt = tetrahedral_gradient(
            field=weights, points=mesh.points[mesh.element_point_indices[element_index]]
        ) + rg.normal(scale=macrophage.drift_bias, size=3)

        # average and re-normalize with existing velocity
        dp_dt += cell['velocity']
        norm = np.linalg.norm(dp_dt)
        if norm > 0.0:
            dp_dt /= norm

        # we need to determine if this movement will put us into an air element.  If that happens,
        # we reduce the rate of movement exponentially (up to 4 times) until we stay within a
        # non-air element. If exponential shortening is unsuccessful after 4 tries, we stay in
        # place. Velocity is updated to dp/dt in either case.
        new_position = cell['point'] + dp_dt
        new_element_index: int = mesh.get_element_index(new_position)
        assert cell['element_index'] > 0
        for _ in range(4):
            # state.log.debug(f"{iteration=}")
            # state.log.debug(f"{new_element_index=}")
            # state.log.debug(f"{cell['element_index']=}")
            # state.log.debug(f"{mesh.element_tissue_type[new_element_index]=}")
            assert cell['element_index'] > 0
            if (
                new_element_index >= 0
                and mesh.element_tissue_type[new_element_index] != TissueType.AIR
            ):
                cell['velocity'][:] = dp_dt
                return new_position, new_element_index

            dp_dt /= 2.0
            new_position = cell['point'] + dp_dt
            new_element_index = mesh.get_element_index(new_position)

        state.log.info(f"{macrophage.cells.cell_data['element_index']=}")

        cell['velocity'].fill(0.0)
        return cell['point'], cell['element_index']

    @staticmethod
    def create_macrophage(
        *, state: State, x: float, y: float, z: float, element_index: Optional[int] = None, **kwargs
    ) -> None:
        """
        Create a new macrophage cell

        Parameters
        ----------
        state : State
            global simulation state
        x : float
        y : float
        z : float
            coordinates of created macrophage
        element_index: int
            id of element that the macrophage lives in, optional.
        kwargs
            parameters for macrophage, passed to MacrophageCellData.create_cell

        Returns
        -------
        nothing
        """
        macrophage: MacrophageState = state.macrophage

        # use default value of iron pool if not present
        iron_pool = kwargs.get('iron_pool', macrophage.ma_internal_iron)
        kwargs.pop('iron_pool', None)

        macrophage.cells.append(
            MacrophageCellData.create_cell(
                point=Point(x=x, y=y, z=z),
                element_index=-1 if element_index is None else element_index,
                iron_pool=iron_pool,
                **kwargs,
            ),
        )

    def update_status(
        self, state: State, macrophage_cell: MacrophageCellData, num_cells_in_phagosome
    ) -> None:
        """
        Update the status of the cell, progressing between states after a certain number of ticks.

        Parameters
        ----------
        state : State
            global simulation state
        macrophage_cell : MacrophageCellData
        num_cells_in_phagosome

        Returns
        -------
        nothing
        """
        macrophage: MacrophageState = state.macrophage

        if macrophage_cell['status'] == PhagocyteStatus.NECROTIC:
            # TODO: what about APOPTOTIC?
            self.release_phagosome(state, macrophage_cell)

        elif num_cells_in_phagosome > macrophage.max_conidia:
            # TODO: how do we get here?
            macrophage_cell['status'] = PhagocyteStatus.NECROTIC

        elif macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
            if macrophage_cell['status_iteration'] >= macrophage.iter_to_rest:
                macrophage_cell['status_iteration'] = 0
                macrophage_cell['tnfa'] = False
                macrophage_cell['status'] = PhagocyteStatus.RESTING
            else:
                macrophage_cell['status_iteration'] += 1

        elif macrophage_cell['status'] == PhagocyteStatus.INACTIVE:
            if macrophage_cell['status_iteration'] >= macrophage.iter_to_change_state:
                macrophage_cell['status_iteration'] = 0
                macrophage_cell['status'] = PhagocyteStatus.RESTING
            else:
                macrophage_cell['status_iteration'] += 1

        elif macrophage_cell['status'] == PhagocyteStatus.ACTIVATING:
            if macrophage_cell['status_iteration'] >= macrophage.iter_to_change_state:
                macrophage_cell['status_iteration'] = 0
                macrophage_cell['status'] = PhagocyteStatus.ACTIVE
            else:
                macrophage_cell['status_iteration'] += 1

        elif macrophage_cell['status'] == PhagocyteStatus.INACTIVATING:
            if macrophage_cell['status_iteration'] >= macrophage.iter_to_change_state:
                macrophage_cell['status_iteration'] = 0
                macrophage_cell['status'] = PhagocyteStatus.INACTIVE
            else:
                macrophage_cell['status_iteration'] += 1

        elif macrophage_cell['status'] == PhagocyteStatus.ANERGIC:
            if macrophage_cell['status_iteration'] >= macrophage.iter_to_change_state:
                macrophage_cell['status_iteration'] = 0
                macrophage_cell['status'] = PhagocyteStatus.RESTING
            else:
                macrophage_cell['status_iteration'] += 1
