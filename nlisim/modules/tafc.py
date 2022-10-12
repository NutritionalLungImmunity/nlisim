import logging
from typing import Any, Dict

# noinspection PyPackageRequirements
import attr

# noinspection PyPackageRequirements
import numpy as np

# noinspection PyPackageRequirements
from scipy.sparse import csr_matrix

from nlisim.diffusion import (
    apply_mesh_diffusion_crank_nicholson,
    assemble_mesh_laplacian_crank_nicholson,
)
from nlisim.grid import TetrahedralMesh
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import (
    michaelian_kinetics_molarity,
    secrete_in_element,
    turnover_rate,
    uptake_proportionally,
)


def molecule_point_field_factory(self: 'TAFCState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(
        dtype=[('TAFC', np.float64), ('TAFCBI', np.float64)]
    )


@attr.s(kw_only=True, repr=False)
class TAFCState(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    k_m_tf_tafc: float  # units: aM
    tafcbi_uptake_rate: float  # units: L * cell^-1 * h^-1
    tafcbi_uptake_rate_unit_t: float  # units: L * cell^-1 * step^-1
    afumigatus_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    afumigatus_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    turnover_rate: float
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class TAFC(ModuleModel):
    # noinspection SpellCheckingInspection
    """TAFC: (T)ri(A)cetyl(F)usarinine C"""

    name = 'tafc'
    StateClass = TAFCState

    def initialize(self, state: State) -> State:
        logging.getLogger('nlisim').debug("Initializing " + self.name)
        tafc: TAFCState = state.tafc
        molecules: MoleculesState = state.molecules

        # config file values
        tafc.k_m_tf_tafc = self.config.getfloat('k_m_tf_tafc')  # units: aM
        tafc.afumigatus_secretion_rate = self.config.getfloat(
            'afumigatus_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        tafc.tafcbi_uptake_rate = self.config.getfloat(
            'tafcbi_uptake_rate'
        )  # units: L * cell^-1 * h^-1
        tafc.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        tafc.turnover_rate = turnover_rate(
            x=1.0,
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        tafc.afumigatus_secretion_rate_unit_t = tafc.afumigatus_secretion_rate * (
            self.time_step / 60
        )  # units: (atto-mol * cell^-1 * h^-1) * (min/step) / (min/hour)
        tafc.tafcbi_uptake_rate_unit_t = (
            tafc.tafcbi_uptake_rate
            * (self.time_step / 60)
            # units: (L * cell^-1 * h^-1) * (min/step) / (min/hour)
            # = L * cell^-1 * step^-1
        )

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=tafc.diffusion_constant, dt=self.time_step
        )
        tafc.cn_a = cn_a
        tafc.cn_b = cn_b
        tafc.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.afumigatus import (
            AfumigatusCellData,
            AfumigatusCellState,
            AfumigatusCellStatus,
            AfumigatusState,
            NetworkSpecies,
        )
        from nlisim.modules.iron import IronState
        from nlisim.modules.transferrin import TransferrinState

        tafc: TAFCState = state.tafc
        transferrin: TransferrinState = state.transferrin
        iron: IronState = state.iron
        molecules: MoleculesState = state.molecules
        afumigatus: AfumigatusState = state.afumigatus
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(tafc.field['TAFC'] >= 0.0)
        assert np.alltrue(tafc.field['TAFCBI'] >= 0.0)

        # interaction with transferrin
        # - calculate iron transfer from transferrin+[1,2]Fe to TAFC
        dfe2_dt = michaelian_kinetics_molarity(
            substrate=transferrin.field["TfFe2"],  # units: atto-M
            enzyme=tafc.field["TAFC"],  # units: atto-M
            k_m=tafc.k_m_tf_tafc,  # units: atto-M
            h=self.time_step / 60,  # units: (min/step) / (min/hour) = hours/step
            k_cat=1.0,  # default
        )  # units: atto-M/step
        for idx in range(transferrin.field["TfFe2"].shape[0]):
            state.log.debug(
                f"{transferrin.field['TfFe2'][idx]=}"
                f"\t{tafc.field['TAFC'][idx]=}"
                f"\t{dfe2_dt[idx]=}"
            )
        dfe_dt = michaelian_kinetics_molarity(
            substrate=transferrin.field["TfFe"],  # units: atto-M
            enzyme=tafc.field["TAFC"],  # units: atto-M
            k_m=tafc.k_m_tf_tafc,  # units: atto-M
            h=self.time_step / 60,  # units: (min/step) / (min/hour) = hours/step
            k_cat=1.0,  # default
        )  # units: atto-M/step
        state.log.debug(f"{np.min(dfe_dt)=} {np.max(dfe_dt)=}")

        # - enforce bounds from TAFC quantity
        total_change = dfe2_dt + dfe_dt
        rel = np.divide(  # safe division, defaults to zero when dividing by zero
            tafc.field['TAFC'],
            total_change,
            out=np.zeros_like(tafc.field['TAFC']),  # source of defaults
            where=total_change != 0.0,
        )
        state.log.debug(f"tafc {np.min(rel)=} {np.max(rel)=}")
        np.clip(rel, 0.0, 1.0, out=rel)  # fix any remaining problem divides
        state.log.debug(f"tafc {np.min(rel)=} {np.max(rel)=}")
        np.multiply(dfe2_dt, rel, out=dfe2_dt)
        np.multiply(dfe_dt, rel, out=dfe_dt)

        # transferrin+2Fe loses an iron, becomes transferrin+Fe
        transferrin.field['TfFe2'] -= dfe2_dt
        transferrin.field['TfFe'] += dfe2_dt

        # transferrin+Fe loses an iron, becomes transferrin
        transferrin.field['TfFe'] -= dfe_dt
        transferrin.field['Tf'] += dfe_dt

        # iron from transferrin becomes bound to TAFC (TAFC->TAFCBI)
        tafc.field['TAFC'] -= dfe2_dt + dfe_dt
        tafc.field['TAFCBI'] += dfe2_dt + dfe_dt

        assert np.alltrue(
            tafc.field['TAFCBI'] >= 0.0
        ), f"{np.min(dfe2_dt)=} {np.min(dfe_dt)=} {dfe2_dt+dfe_dt=}"

        # interaction with iron, all available iron is bound to TAFC
        potential_reactive_quantity = np.minimum(iron.field, tafc.field['TAFC'])
        tafc.field['TAFC'] -= potential_reactive_quantity
        tafc.field['TAFCBI'] += potential_reactive_quantity
        iron.field -= potential_reactive_quantity

        assert np.alltrue(tafc.field['TAFCBI'] >= 0.0)

        # interaction with fungus
        for afumigatus_cell_index in afumigatus.cells.alive():
            afumigatus_cell: AfumigatusCellData = afumigatus.cells[afumigatus_cell_index]

            if afumigatus_cell['state'] != AfumigatusCellState.FREE:
                continue

            afumigatus_cell_element: int = afumigatus_cell['element_index']
            afumigatus_bool_net: np.ndarray = afumigatus_cell['boolean_network']

            # uptake iron from TAFCBI
            if afumigatus_bool_net[NetworkSpecies.MirB] & afumigatus_bool_net[NetworkSpecies.EstB]:
                quantity = uptake_proportionally(
                    mesh=mesh,
                    point_field=tafc.field['TAFCBI'],
                    element_index=afumigatus_cell_element,
                    point=afumigatus_cell['point'],
                    k=tafc.tafcbi_uptake_rate_unit_t,
                )
                # quantity = (
                #     mesh.integrate_point_function_single_element(
                #         point_function=tafc.field['TAFCBI'],
                #         element_index=afumigatus_cell_element
                #     )
                #     * tafc.tafcbi_uptake_rate_unit_t
                #     / mesh.element_volumes[afumigatus_cell_element]
                # )  # units: atto-mols * (L * cell^-1 * step^-1) / L
                #    # = atto-mols * cell^-1 * step^-1
                state.log.debug(' ')
                state.log.debug(
                    f"{tafc.field['TAFCBI'][mesh.element_point_indices[afumigatus_cell_element]]=}"
                )
                state.log.debug(
                    "mesh.integrate_point_function_single_element(\n"
                    "    point_function=tafc.field['TAFCBI'],\n"
                    "    element_index=afumigatus_cell_element,\n"
                    ")="
                    + str(
                        mesh.integrate_point_function_single_element(
                            point_function=tafc.field['TAFCBI'],
                            element_index=afumigatus_cell_element,
                        )
                    )
                )
                state.log.debug(' ')
                state.log.debug(f"{quantity=}")
                state.log.debug(f"{tafc.tafcbi_uptake_rate_unit_t=}")
                # uptake_in_element(
                #     mesh=mesh,
                #     point_field=tafc.field['TAFCBI'],
                #     element_index=afumigatus_cell_element,
                #     point=afumigatus_cell['point'],
                #     amount=quantity,
                # )
                afumigatus_cell['iron_pool'] += quantity

            assert np.alltrue(tafc.field['TAFCBI'] >= 0.0)

            # secrete TAFC
            if afumigatus_bool_net[NetworkSpecies.TAFC] and afumigatus_cell['status'] in {
                AfumigatusCellStatus.SWELLING_CONIDIA,
                AfumigatusCellStatus.HYPHAE,
                AfumigatusCellStatus.GERM_TUBE,
            }:
                secrete_in_element(
                    mesh=mesh,
                    point_field=tafc.field['TAFC'],
                    element_index=afumigatus_cell_element,
                    point=afumigatus_cell['point'],
                    amount=tafc.afumigatus_secretion_rate_unit_t,
                )

        # Degrade TAFC
        trnvr_rt = turnover_rate(
            x=np.array(1.0, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        tafc.field['TAFC'] *= trnvr_rt
        tafc.field['TAFCBI'] *= trnvr_rt

        # Diffusion of TAFC
        for component in {'TAFC', 'TAFCBI'}:
            tafc.field[component][:] = apply_mesh_diffusion_crank_nicholson(
                variable=tafc.field[component],
                cn_a=tafc.cn_a,
                cn_b=tafc.cn_b,
                dofs=tafc.dofs,
            )

        assert np.alltrue(tafc.field['TAFC'] >= 0.0)
        assert np.alltrue(tafc.field['TAFCBI'] >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        tafc: TAFCState = state.tafc
        mesh: TetrahedralMesh = state.mesh

        concentration_no_fe = (
            mesh.integrate_point_function(tafc.field['TAFC']) / 1e9 / mesh.total_volume
        )
        concentration_fe = (
            mesh.integrate_point_function(tafc.field['TAFCBI']) / 1e9 / mesh.total_volume
        )

        concentration = concentration_no_fe + concentration_fe

        return {
            'concentration (nM)': float(concentration / 1e9),
            'concentration TAFC (nM)': float(concentration_no_fe / 1e9),
            'concentration TAFCBI (nM)': float(concentration_fe / 1e9),
        }

    def visualization_data(self, state: State):
        tafc: TAFCState = state.tafc
        return 'molecule', tafc.field
