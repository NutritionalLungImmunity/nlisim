from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.coordinates import Point, Voxel
from nlisim.diffusion import apply_grid_diffusion, assemble_mesh_laplacian_crank_nicholson
from nlisim.grid import TetrahedralMesh
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import turnover_rate


def molecule_point_field_factory(self: 'HemoglobinState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attrs(kw_only=True, repr=False)
class HemoglobinState(ModuleState):
    grid: np.ndarray = attrib(default=attr.Factory(molecule_point_field_factory, takes_self=True))
    uptake_rate: float
    ma_heme_import_rate: float
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class Hemoglobin(ModuleModel):
    """Hemoglobin"""

    name = 'hemoglobin'
    StateClass = HemoglobinState

    def initialize(self, state: State) -> State:
        hemoglobin: HemoglobinState = state.hemoglobin

        # config file values
        hemoglobin.uptake_rate = self.config.getfloat('uptake_rate')
        hemoglobin.ma_heme_import_rate = self.config.getfloat('ma_heme_import_rate')
        hemoglobin.diffusion_constant = self.config.getfloat(
            'diffusion_constant'
        )  # units: µm^2/min

        # computed values (none)

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=hemoglobin.diffusion_constant, dt=self.time_step
        )
        hemoglobin.cn_a = cn_a
        hemoglobin.cn_b = cn_b
        hemoglobin.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.afumigatus import (
            AfumigatusCellData,
            AfumigatusCellStatus,
            AfumigatusState,
        )

        hemoglobin: HemoglobinState = state.hemoglobin
        molecules: MoleculesState = state.molecules
        afumigatus: AfumigatusState = state.afumigatus
        mesh: TetrahedralMesh = state.mesh

        def afumigatus_iron_uptake(*, element_index: int, point: Point, amount: float) -> None:
            proportions = np.asarray(mesh.tetrahedral_proportions(element_index, point))
            points = mesh.element_point_indices[element_index]
            # new pt concentration = (old pt amount + new amount) / pt dual volume
            #    = (old conc * pt dual volume + new amount) / pt dual volume
            #    = old conc + (new amount / pt dual volume)
            il6.field[points] += (
                proportions * amount / mesh.point_dual_volumes[points]
            )  # units: prop * atto-mol / L = atto-M

        # afumigatus uptakes iron from hemoglobin
        live_afumigatus = afumigatus.cells.alive()
        iron_uptaking_afumigatus = np.logical_or(
            afumigatus.cells.cell_data[live_afumigatus]['status'] == AfumigatusCellStatus.HYPHAE,
            afumigatus.cells.cell_data[live_afumigatus]['status'] == AfumigatusCellStatus.GERM_TUBE,
        )
        afumigatus_elements = np.array(afumigatus.cells.element_index)[
            iron_uptaking_afumigatus
        ]  # TODO: convert _reverse_element_index in CellList to ndarray
        afumigatus_cells = afumigatus.cells.cell_data[iron_uptaking_afumigatus]

        for afumigatus_cell_index in afumigatus.cells.alive():
            afumigatus_cell: AfumigatusCellData = afumigatus.cells[afumigatus_cell_index]
            if afumigatus_cell['status'] in {
                AfumigatusCellStatus.HYPHAE,
                AfumigatusCellStatus.GERM_TUBE,
            }:
                afumigatus_iron_uptake(
                    element_index=afumigatus.cells.element_index[afumigatus_cell_index],
                    point=afumigatus_cell['point'],
                    amount=il6.macrophage_secretion_rate_unit_t,
                )
                afumigatus_cell_voxel: Voxel = grid.get_voxel(afumigatus_cell['point'])
                fungal_absorbed_hemoglobin = (
                    hemoglobin.uptake_rate * hemoglobin.grid[tuple(afumigatus_cell_voxel)]
                )
                hemoglobin.grid[tuple(afumigatus_cell_voxel)] -= fungal_absorbed_hemoglobin
                afumigatus_cell['iron_pool'] += 4 * fungal_absorbed_hemoglobin

        # Degrade Hemoglobin
        hemoglobin.grid *= turnover_rate(
            x=hemoglobin.grid,
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of Hemoglobin
        hemoglobin.grid[:] = apply_grid_diffusion(
            variable=hemoglobin.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        hemoglobin: HemoglobinState = state.hemoglobin
        voxel_volume = state.voxel_volume

        return {
            'concentration (nM)': float(np.mean(hemoglobin.grid) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        hemoglobin: HemoglobinState = state.hemoglobin
        return 'molecule', hemoglobin.grid
