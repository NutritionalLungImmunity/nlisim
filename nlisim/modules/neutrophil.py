import logging
import math
from typing import Any, Dict, Tuple

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellFields, CellList
from nlisim.coordinates import Point
from nlisim.grid import TetrahedralMesh
from nlisim.modules.mip2 import MIP2State
from nlisim.modules.phagocyte import (
    PhagocyteCellData,
    PhagocyteModel,
    PhagocyteModuleState,
    PhagocyteState,
    PhagocyteStatus,
    interact_with_aspergillus,
)
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import (
    GridTissueType,
    activation_function,
    sample_point_from_simplex,
    secrete_in_element,
    tetrahedral_gradient,
)

MAX_CONIDIA = (
    50  # note: this the max that we can set the max to. i.e. not an actual model parameter
)


class NeutrophilCellData(PhagocyteCellData):
    NEUTROPHIL_FIELDS: CellFields = [
        ('status', np.uint8),
        ('state', np.uint8),
        ('iron_pool', np.float64),  # units: atto-mol
        ('tnfa', bool),
        ('status_iteration', np.uint),
    ]

    dtype = np.dtype(
        CellData.FIELDS + PhagocyteCellData.PHAGOCYTE_FIELDS + NEUTROPHIL_FIELDS, align=True
    )  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        initializer = {
            'status': kwargs.get('status', PhagocyteStatus.RESTING),
            'state': kwargs.get('state', PhagocyteState.FREE),
            'iron_pool': kwargs.get('iron_pool', 0.0),
            'tnfa': kwargs.get('tnfa', False),
            'status_iteration': kwargs.get('status_iteration', 0),
        }

        # ensure that these come in the correct order
        return PhagocyteCellData.create_cell_tuple(**kwargs) + tuple(
            [initializer[key] for key, *_ in NeutrophilCellData.NEUTROPHIL_FIELDS]
        )


@attrs(kw_only=True, frozen=True, repr=False)
class NeutrophilCellList(CellList):
    CellDataClass = NeutrophilCellData


def cell_list_factory(self: 'NeutrophilState') -> NeutrophilCellList:
    return NeutrophilCellList(mesh=self.global_state.mesh)


@attrs(kw_only=True)
class NeutrophilState(PhagocyteModuleState):
    cells: NeutrophilCellList = attrib(default=attr.Factory(cell_list_factory, takes_self=True))
    half_life: float  # units: hours
    apoptosis_probability: float  # units: probability
    time_to_change_state: float  # units: hours
    iter_to_change_state: int  # units: steps
    pr_n_hyphae_param: float
    pr_n_phagocyte_param: float
    recruitment_rate: float
    rec_bias: float
    max_neutrophils: float  # TODO: 0.5?
    n_frac: float
    drift_bias: float
    n_move_rate_act: float  # units: µm
    n_move_rate_rest: float  # units: µm
    init_num_neutrophils: int  # units: count


class Neutrophil(PhagocyteModel):
    name = 'neutrophil'
    StateClass = NeutrophilState

    def initialize(self, state: State):
        logging.getLogger('nlisim').debug("Initializing " + self.name)
        neutrophil: NeutrophilState = state.neutrophil
        mesh: TetrahedralMesh = state.mesh

        neutrophil.init_num_neutrophils = self.config.getint('init_num_neutrophils')  # units: count

        neutrophil.time_to_change_state = self.config.getfloat(
            'time_to_change_state'
        )  # units: hours
        neutrophil.max_conidia = self.config.getint(
            'max_conidia'
        )  # (from phagocyte model) units: count

        neutrophil.recruitment_rate = self.config.getfloat('recruitment_rate')
        neutrophil.rec_bias = self.config.getfloat('rec_bias')
        neutrophil.max_neutrophils = self.config.getfloat('max_neutrophils')  # units: count
        neutrophil.n_frac = self.config.getfloat('n_frac')

        neutrophil.drift_bias = self.config.getfloat('drift_bias')
        neutrophil.n_move_rate_act = self.config.getfloat('n_move_rate_act')
        neutrophil.n_move_rate_rest = self.config.getfloat('n_move_rate_rest')

        neutrophil.pr_n_hyphae_param = self.config.getfloat('pr_n_hyphae_param')  # units: h/L
        neutrophil.pr_n_phagocyte_param = self.config.getfloat('pr_n_phagocyte_param')  # units: h/L

        neutrophil.half_life = self.config.getfloat('half_life')  # units: hours

        # computed values
        neutrophil.apoptosis_probability = -math.log(0.5) / (
            neutrophil.half_life
            * (60 / self.time_step)
            # units: hours*(min/hour)/(min/step)=steps
        )  # units: probability
        neutrophil.iter_to_change_state = int(
            neutrophil.time_to_change_state * 60 / self.time_step
        )  # units: hours * (min/hour) / (min/step) = steps

        # initialize neutrophil cells. Cells will be distributed into non-air layers, in a
        # uniformly random manner.
        locations = np.where(mesh.element_tissue_type != GridTissueType.AIR)[0]
        volumes = mesh.element_volumes[locations]
        probabilities = volumes / np.sum(volumes)
        for _ in range(neutrophil.init_num_neutrophils):
            element_index = locations[np.argmax(np.random.random() < probabilities)]
            simplex_coords = sample_point_from_simplex()
            point = mesh.points[mesh.element_point_indices[element_index]] @ simplex_coords
            self.create_neutrophil(
                state=state,
                x=point[2],
                y=point[1],
                z=point[0],
                element_index=element_index,
            )

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        from nlisim.modules.afumigatus import (
            Afumigatus,
            AfumigatusCellData,
            AfumigatusCellStatus,
            AfumigatusState,
        )
        from nlisim.modules.iron import IronState
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState

        neutrophil: NeutrophilState = state.neutrophil
        macrophage: MacrophageState = state.macrophage
        afumigatus: AfumigatusState = state.afumigatus
        iron: IronState = state.iron
        mesh: TetrahedralMesh = state.mesh

        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell: NeutrophilCellData = neutrophil.cells[neutrophil_cell_index]
            neutrophil_cell_element: int = neutrophil_cell['element_index']

            self.update_status(state, neutrophil_cell)

            # ---------- interactions

            # dead and dying cells release iron
            if neutrophil_cell['status'] in {
                PhagocyteStatus.NECROTIC,
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.DEAD,
            }:
                secrete_in_element(
                    mesh=mesh,
                    point_field=iron.field,
                    element_index=neutrophil_cell_element,
                    point=neutrophil_cell['point'],
                    amount=neutrophil_cell['iron_pool'],
                )
                neutrophil_cell['iron_pool'] = 0
                neutrophil_cell['dead'] = True

            # interact with fungus
            if neutrophil_cell['state'] == PhagocyteState.FREE and neutrophil_cell[
                'status'
            ] not in {
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.NECROTIC,
                PhagocyteStatus.DEAD,
            }:
                # get fungal cells in this element
                local_aspergillus = afumigatus.cells.get_cells_in_element(neutrophil_cell_element)
                for aspergillus_cell_index in local_aspergillus:
                    aspergillus_cell: AfumigatusCellData = afumigatus.cells[aspergillus_cell_index]
                    if aspergillus_cell['dead']:
                        continue

                    if aspergillus_cell['status'] in {
                        AfumigatusCellStatus.HYPHAE,
                        AfumigatusCellStatus.GERM_TUBE,
                    }:
                        # possibly kill the fungal cell, extracellularly
                        if rg.uniform() < -math.expm1(
                            -self.time_step
                            / (
                                60
                                * mesh.element_volumes[neutrophil_cell_element]
                                * neutrophil.pr_n_hyphae_param
                            )
                        ):  # units: probability

                            interact_with_aspergillus(
                                phagocyte_cell=neutrophil_cell,
                                phagocyte_cell_index=neutrophil_cell_index,
                                phagocyte_cells=neutrophil.cells,
                                aspergillus_cell=aspergillus_cell,
                                aspergillus_cell_index=aspergillus_cell_index,
                                phagocyte=neutrophil,
                            )
                            Afumigatus.kill_fungal_cell(
                                afumigatus=afumigatus,
                                afumigatus_cell=aspergillus_cell,
                                afumigatus_cell_index=aspergillus_cell_index,
                                iron=iron,
                                mesh=mesh,
                            )
                        else:
                            neutrophil_cell['state'] = PhagocyteState.INTERACTING

                    elif aspergillus_cell['status'] == AfumigatusCellStatus.SWELLING_CONIDIA:
                        if rg.uniform() < -math.expm1(
                            -self.time_step
                            / (
                                60
                                * mesh.element_volumes[neutrophil_cell_element]
                                * neutrophil.pr_n_phagocyte_param
                            )
                        ):
                            interact_with_aspergillus(
                                phagocyte_cell=neutrophil_cell,
                                phagocyte_cell_index=neutrophil_cell_index,
                                phagocyte_cells=neutrophil.cells,
                                aspergillus_cell=aspergillus_cell,
                                aspergillus_cell_index=aspergillus_cell_index,
                                phagocyte=neutrophil,
                            )

            # interact with macrophages:
            # if we are apoptotic, give our iron and phagosome to a nearby
            # present macrophage (if empty)
            if neutrophil_cell['status'] == PhagocyteStatus.APOPTOTIC:
                local_macrophages = macrophage.cells.get_cells_in_element(neutrophil_cell_element)
                for macrophage_index in local_macrophages:
                    macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_index]
                    macrophage_num_cells_in_phagosome = np.sum(macrophage_cell['phagosome'] >= 0)
                    # TODO: Henrique, why only if empty?
                    if macrophage_num_cells_in_phagosome == 0:
                        macrophage_cell['phagosome'] = neutrophil_cell['phagosome']
                        macrophage_cell['iron_pool'] += neutrophil_cell['iron_pool']
                        neutrophil_cell['iron_pool'] = 0.0
                        neutrophil_cell['status'] = PhagocyteStatus.DEAD
                        macrophage_cell['status'] = PhagocyteStatus.INACTIVE

            # Movement
            if neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
                max_move_step = neutrophil.n_move_rate_act * self.time_step
            else:
                max_move_step = neutrophil.n_move_rate_rest * self.time_step
            move_step: int = rg.poisson(max_move_step)
            # move the cell 1 µm, move_step number of times
            for _ in range(move_step):
                self.single_step_move(
                    state, neutrophil_cell, neutrophil_cell_index, neutrophil.cells
                )
            # TODO: understand the meaning of the parameter here: moving randomly n steps is
            #  different than moving n steps in a random direction. Which is it?

        # Recruitment
        self.recruit_neutrophils(state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        neutrophil: NeutrophilState = state.neutrophil
        live_neutrophils = neutrophil.cells.alive()

        max_index = max(map(int, PhagocyteStatus))
        status_counts = np.bincount(
            np.fromiter(
                (
                    neutrophil.cells[neutrophil_cell_index]['status']
                    for neutrophil_cell_index in live_neutrophils
                ),
                dtype=np.uint8,
            ),
            minlength=max_index + 1,
        )

        tnfa_active = int(
            np.sum(
                np.fromiter(
                    (
                        neutrophil.cells[neutrophil_cell_index]['tnfa']
                        for neutrophil_cell_index in live_neutrophils
                    ),
                    dtype=bool,
                )
            )
        )

        return {
            'count': len(neutrophil.cells.alive()),
            'inactive': int(status_counts[PhagocyteStatus.INACTIVE]),
            'inactivating': int(status_counts[PhagocyteStatus.INACTIVATING]),
            'resting': int(status_counts[PhagocyteStatus.RESTING]),
            'activating': int(status_counts[PhagocyteStatus.ACTIVATING]),
            'active': int(status_counts[PhagocyteStatus.ACTIVE]),
            'apoptotic': int(status_counts[PhagocyteStatus.APOPTOTIC]),
            'necrotic': int(status_counts[PhagocyteStatus.NECROTIC]),
            'interacting': int(status_counts[PhagocyteStatus.INTERACTING]),
            'TNFa active': tnfa_active,
        }

    def visualization_data(self, state: State):
        return 'cells', state.neutrophil.cells

    def single_step_probabilistic_drift(
        self, state: State, cell: PhagocyteCellData, element_index: int
    ) -> Tuple[Point, int]:
        """
        Calculate a 1µm movement of a neutrophil

        Parameters
        ----------
        state : State
            global simulation state
        cell : NeutrophilCellData
            a neutrophil cell
        element_index : int
            index of the current tetrahedral element occupied by the neutrophil

        Returns
        -------
        Point
            the new position of the neutrophil
        """
        # neutrophils are attracted by MIP2

        mip2: MIP2State = state.mip2
        mesh: TetrahedralMesh = state.mesh

        # compute chemokine influence on velocity, with some randomness.
        # neutrophil has a non-zero probability of moving into non-air voxels.
        # if not any of these, stay in place. This could happen if e.g. you are
        # somehow stranded in air.
        chemokine_levels = mip2.field[mesh.element_point_indices[element_index]]
        weights = activation_function(
            x=chemokine_levels,
            k_d=mip2.k_d,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            volume=1.0,  # already in concentration (M)
            b=1.0,
        )

        # movement tends toward the gradient direction
        dp_dt = tetrahedral_gradient(
            field=weights, points=mesh.points[mesh.element_point_indices[element_index]]
        ) + rg.normal(
            scale=0.1, size=3
        )  # TODO: expose scale to config file

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
        for _ in range(4):
            if (
                new_element_index >= 0
                and mesh.element_tissue_type[new_element_index] != GridTissueType.AIR
            ):
                cell['velocity'][:] = dp_dt
                return new_position, new_element_index
            dp_dt /= 2.0
            new_position = cell['point'] + dp_dt
            new_element_index = mesh.get_element_index(new_position)

        cell['velocity'].fill(0.0)
        return cell['point'], cell['element_index']

    def update_status(self, state: State, neutrophil_cell: NeutrophilCellData) -> None:
        """
        Update the status of the cell, progressing between states after a certain number of ticks.

        Parameters
        ----------
        state : State
            global simulation state
        neutrophil_cell : NeutrophilCellData

        Returns
        -------
        nothing
        """
        neutrophil: NeutrophilState = state.neutrophil

        if neutrophil_cell['status'] in {PhagocyteStatus.NECROTIC, PhagocyteStatus.APOPTOTIC}:
            self.release_phagosome(state, neutrophil_cell)
            # releases iron & dies later

        elif rg.uniform() < neutrophil.apoptosis_probability:
            neutrophil_cell['status'] = PhagocyteStatus.APOPTOTIC

        elif neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
            if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                neutrophil_cell['status_iteration'] = 0
                neutrophil_cell['tnfa'] = False
                neutrophil_cell['status'] = PhagocyteStatus.RESTING
                neutrophil_cell['state'] = PhagocyteState.FREE
            else:
                neutrophil_cell['status_iteration'] += 1

        elif neutrophil_cell['status'] == PhagocyteStatus.ACTIVATING:
            if neutrophil_cell['status_iteration'] >= neutrophil.iter_to_change_state:
                neutrophil_cell['status_iteration'] = 0
                neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
            else:
                neutrophil_cell['status_iteration'] += 1

    def recruit_neutrophils(self, state: State) -> None:
        """
        Recruit neutrophils based on MIP2 activation

        Parameters
        ----------
        state : State
            global simulation state

        Returns
        -------
        nothing
        """
        from nlisim.modules.mip2 import MIP2State
        from nlisim.util import activation_function

        neutrophil: NeutrophilState = state.neutrophil
        mip2: MIP2State = state.mip2
        mesh: TetrahedralMesh = state.mesh

        # 1. compute number of neutrophils to recruit
        num_live_neutrophils = len(neutrophil.cells.alive())
        avg = (
            neutrophil.recruitment_rate
            * neutrophil.n_frac
            * (mesh.integrate_point_function(mip2.field) / mesh.total_volume)
            * (1 - num_live_neutrophils / neutrophil.max_neutrophils)
            / mip2.k_d
        )
        number_to_recruit = np.random.poisson(avg) if avg > 0 else 0
        if number_to_recruit <= 0:
            return
        # 2. get voxels for new neutrophils, based on activation
        if number_to_recruit > 0:
            activated_points = np.where(
                activation_function(
                    x=mip2.field,
                    k_d=mip2.k_d,
                    h=self.time_step / 60,
                    volume=1.0,  # mip2.field already in concentration (M) units
                    b=1.0,
                )
                < rg.uniform(size=mip2.field.shape)
            )
            activated_elements = mesh.elements_incident_to(points=activated_points)
            for element_index in rg.choice(
                activated_elements, size=number_to_recruit, replace=True
            ):
                simplex_coords = sample_point_from_simplex()
                point = mesh.points[mesh.element_point_indices[element_index]] @ simplex_coords
                self.create_neutrophil(
                    state=state,
                    x=point[2],
                    y=point[1],
                    z=point[0],
                )

    @staticmethod
    def create_neutrophil(state: State, x: float, y: float, z: float, **kwargs) -> None:
        """
        Create a new neutrophil cell

        Parameters
        ----------
        state : State
            global simulation state
        x : float
        y : float
        z : float
            coordinates of created neutrophil
        kwargs
            parameters for neutrophil, passed to NeutrophilCellData.create_cell

        Returns
        -------
        nothing
        """
        neutrophil: NeutrophilState = state.neutrophil

        # use default value of iron pool if not present
        iron_pool = kwargs.get('iron_pool', 0.0)
        kwargs.pop('iron_pool', None)

        neutrophil.cells.append(
            NeutrophilCellData.create_cell(
                point=Point(x=x, y=y, z=z), iron_pool=iron_pool, **kwargs
            )
        )
