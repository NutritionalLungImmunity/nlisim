from typing import Any, Dict

import attr
import numpy as np
from scipy.sparse import csr_matrix

from nlisim.diffusion import (
    apply_mesh_diffusion_crank_nicholson,
    assemble_mesh_laplacian_crank_nicholson,
)
from nlisim.grid import TetrahedralMesh
from nlisim.module import ModuleModel, ModuleState
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import (
    iron_tf_reaction,
    logger,
    michaelian_kinetics_molarity,
    secrete_in_element,
    turnover,
    uptake_in_element,
)


def molecule_point_field_factory(self: 'LactoferrinState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(
        dtype=[
            ('Lactoferrin', np.float64),
            ('LactoferrinFe', np.float64),
            ('LactoferrinFe2', np.float64),
        ]
    )


@attr.s(kw_only=True, repr=False)
class LactoferrinState(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    k_m_tf_lac: float  # units: atto-M
    p1: float  # units: none
    p2: float  # units: none
    p3: float  # units: none
    ma_iron_import_rate_vol: float  # units: L * cell^-1 * h^-1
    ma_iron_import_rate_vol_unit_t: float  # units: L * cell^-1 * step^-1
    neutrophil_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    neutrophil_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class Lactoferrin(ModuleModel):
    """Lactoferrin"""

    name = 'lactoferrin'
    StateClass = LactoferrinState

    def initialize(self, state: State) -> State:
        logger.info("Initializing " + self.name)
        lactoferrin: LactoferrinState = state.lactoferrin

        # config file values
        lactoferrin.k_m_tf_lac = self.config.getfloat('k_m_tf_lac')  # units: aM
        lactoferrin.p1 = self.config.getfloat('p1')
        lactoferrin.p2 = self.config.getfloat('p2')
        lactoferrin.p3 = self.config.getfloat('p3')
        lactoferrin.ma_iron_import_rate_vol = self.config.getfloat(
            'ma_iron_import_rate_vol'
        )  # units: L * cell^-1 * h^-1
        lactoferrin.neutrophil_secretion_rate = self.config.getfloat(
            'neutrophil_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        lactoferrin.diffusion_constant = self.config.getfloat(
            'diffusion_constant'
        )  # units: µm^2/min

        # computed values
        lactoferrin.neutrophil_secretion_rate_unit_t = lactoferrin.neutrophil_secretion_rate * (
            self.time_step / 60
        )  # units: atto-mol * cell^-1 * step^-1
        logger.info(f"Computed {lactoferrin.neutrophil_secretion_rate_unit_t=}")
        lactoferrin.ma_iron_import_rate_vol_unit_t = lactoferrin.ma_iron_import_rate_vol * (
            self.time_step / 60
        )  # units: L * cell^-1 * step^-1
        logger.info(f"Computed {lactoferrin.ma_iron_import_rate_vol_unit_t=}")

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=lactoferrin.diffusion_constant, dt=self.time_step
        )
        lactoferrin.cn_a = cn_a
        lactoferrin.cn_b = cn_b
        lactoferrin.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        logger.info("Advancing " + self.name + f" from t={previous_time}")

        from nlisim.modules.iron import IronState
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.molecules import MoleculesState
        from nlisim.modules.neutrophil import NeutrophilCellData, NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteState, PhagocyteStatus
        from nlisim.modules.transferrin import TransferrinState

        lactoferrin: LactoferrinState = state.lactoferrin
        transferrin: TransferrinState = state.transferrin
        iron: IronState = state.iron
        molecules: MoleculesState = state.molecules
        macrophage: MacrophageState = state.macrophage
        neutrophil: NeutrophilState = state.neutrophil
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(lactoferrin.field['Lactoferrin'] >= 0.0)
        assert np.alltrue(lactoferrin.field['LactoferrinFe'] >= 0.0)
        assert np.alltrue(lactoferrin.field['LactoferrinFe2'] >= 0.0)

        # macrophages uptake iron from lactoferrin
        live_macrophages = macrophage.cells.alive()
        rg.shuffle(live_macrophages)
        for macrophage_cell_index in live_macrophages:
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_element_index: int = macrophage_cell['element_index']

            uptake_proportion = np.minimum(
                lactoferrin.ma_iron_import_rate_vol_unit_t
                / mesh.element_volumes[macrophage_element_index],
                1.0,
            )  # units: L * cell^-1 * step^-1 / L = proportion * cell^-1 * step^-1

            qtty_fe2 = (
                mesh.evaluate_point_function(
                    point_function=lactoferrin.field['LactoferrinFe2'],
                    point=macrophage_cell['point'],
                    element_index=macrophage_element_index,
                )
                * uptake_proportion
            )  # units: atto-M * cell^-1 * step^-1
            qtty_fe = (
                mesh.evaluate_point_function(
                    point_function=lactoferrin.field['LactoferrinFe'],
                    point=macrophage_cell['point'],
                    element_index=macrophage_element_index,
                )
                * uptake_proportion
            )  # units: atto-M * cell^-1 * step^-1

            assert mesh.in_tetrahedral_element(
                element_index=macrophage_element_index, point=macrophage_cell['point']
            ), (
                f"{macrophage_element_index=},\n"
                f"{macrophage_cell['point']=},\n"
                f"{mesh.element_point_indices[macrophage_element_index]=}\n"
                f"{mesh.points[mesh.element_point_indices[macrophage_element_index]]=}"
            )
            uptake_in_element(
                mesh=mesh,
                point_field=lactoferrin.field['LactoferrinFe2'],
                element_index=macrophage_element_index,
                point=macrophage_cell['point'],
                amount=qtty_fe2,
            )
            uptake_in_element(
                mesh=mesh,
                point_field=lactoferrin.field['LactoferrinFe'],
                element_index=macrophage_element_index,
                point=macrophage_cell['point'],
                amount=qtty_fe,
            )
            macrophage_cell['iron_pool'] += (2 * qtty_fe2 + qtty_fe) * mesh.element_volumes[
                macrophage_element_index
            ]  # units: atto-M * cell^-1 * step^-1 * L = atto-mol * cell^-1 * step^-1

        # active and interacting neutrophils secrete lactoferrin
        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell: NeutrophilCellData = neutrophil.cells[neutrophil_cell_index]

            if (
                neutrophil_cell['status'] != PhagocyteStatus.ACTIVE
                or neutrophil_cell['state'] != PhagocyteState.INTERACTING
            ):
                continue

            neutrophil_element_index: int = neutrophil_cell['element_index']
            secrete_in_element(
                mesh=mesh,
                point_field=lactoferrin.field['Lactoferrin'],
                element_index=neutrophil_element_index,
                point=neutrophil_cell['point'],
                amount=lactoferrin.neutrophil_secretion_rate_unit_t,
            )

        # interaction with transferrin
        # - calculate iron transfer from transferrin+[1,2]Fe to lactoferrin
        dfe2_dt = michaelian_kinetics_molarity(
            substrate=transferrin.field['TfFe2'],  # units: atto-M
            enzyme=lactoferrin.field["Lactoferrin"],  # units: atto-M
            k_m=lactoferrin.k_m_tf_lac,  # units: atto-M
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=1.0,
        )  # units: atto-M/step
        dfe_dt = michaelian_kinetics_molarity(
            substrate=transferrin.field['TfFe'],  # units: atto-M
            enzyme=lactoferrin.field['Lactoferrin'],  # units: atto-M
            k_m=lactoferrin.k_m_tf_lac,  # units: atto-M
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=1.0,
        )  # units: atto-M/step

        # - scale when lactoferrin quantity is exceeded
        dfex_dt = dfe2_dt + dfe_dt  # units: atto-M/step
        mask = dfex_dt > lactoferrin.field['Lactoferrin']
        rel = np.divide(  # safe division, defaults to zero when dividing by zero
            lactoferrin.field['Lactoferrin'],
            dfex_dt,
            out=np.zeros_like(lactoferrin.field['Lactoferrin']),  # source of defaults
            where=dfex_dt != 0.0,
        )
        np.clip(rel, 0.0, 1.0, out=rel)  # fix any remaining problem divides
        np.multiply(dfe2_dt, rel, out=dfe2_dt, where=mask)
        np.multiply(dfe_dt, rel, out=dfe_dt, where=mask)

        # - calculate iron transfer from transferrin+[1,2]Fe to lactoferrin+Fe
        dfe2_dt_fe = michaelian_kinetics_molarity(
            substrate=transferrin.field['TfFe2'],  # units: atto-M
            enzyme=lactoferrin.field['LactoferrinFe'],  # units: atto-M
            k_m=lactoferrin.k_m_tf_lac,  # units: atto-M
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=1.0,
        )  # units: atto-M/step
        dfe_dt_fe = michaelian_kinetics_molarity(
            substrate=transferrin.field['TfFe'],  # units: atto-M
            enzyme=lactoferrin.field['LactoferrinFe'],  # units: atto-M
            k_m=lactoferrin.k_m_tf_lac,  # units: atto-M
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=1.0,
        )  # units: atto-M/step

        # - scale when lactoferrin+fe quantity is exceeded
        dfex_dt_fe = dfe2_dt_fe + dfe_dt_fe  # units: atto-M/step
        mask = dfex_dt_fe > lactoferrin.field['LactoferrinFe']
        rel = np.divide(  # safe division, defaults to zero when dividing by zero
            lactoferrin.field['LactoferrinFe'],
            dfex_dt_fe,
            out=np.zeros_like(lactoferrin.field['Lactoferrin']),  # source of defaults
            where=dfex_dt_fe != 0.0,
        )
        np.clip(rel, 0.0, 1.0, out=rel)  # fix any remaining problem divides
        np.multiply(dfe2_dt_fe, rel, out=dfe2_dt_fe, where=mask)
        np.multiply(dfe_dt_fe, rel, out=dfe_dt_fe, where=mask)

        # transferrin+2Fe loses an iron, becomes transferrin+Fe
        transferrin.field['TfFe2'] -= dfe2_dt + dfe2_dt_fe
        transferrin.field['TfFe'] += dfe2_dt + dfe2_dt_fe

        # transferrin+Fe loses an iron, becomes transferrin
        transferrin.field['TfFe'] -= dfe_dt + dfe_dt_fe
        transferrin.field['Tf'] += dfe_dt + dfe_dt_fe

        # lactoferrin gains an iron, becomes lactoferrin+Fe
        lactoferrin.field['Lactoferrin'] -= dfe2_dt + dfe_dt
        lactoferrin.field['LactoferrinFe'] += dfe2_dt + dfe_dt

        # lactoferrin+Fe gains an iron, becomes lactoferrin+2Fe
        lactoferrin.field['LactoferrinFe'] -= dfe2_dt_fe + dfe_dt_fe
        lactoferrin.field['LactoferrinFe2'] += dfe2_dt_fe + dfe_dt_fe

        # interaction with iron
        lactoferrin_fe_capacity = (
            2 * lactoferrin.field["Lactoferrin"] + lactoferrin.field["LactoferrinFe"]
        )  # units: atto-M
        potential_reactive_quantity = np.minimum(iron.field, lactoferrin_fe_capacity)
        rel_tf_fe = iron_tf_reaction(
            iron=potential_reactive_quantity,
            tf=lactoferrin.field["Lactoferrin"],
            tf_fe=lactoferrin.field["LactoferrinFe"],
            p1=lactoferrin.p1,
            p2=lactoferrin.p2,
            p3=lactoferrin.p3,
        )
        tffe_qtty = rel_tf_fe * potential_reactive_quantity
        tffe2_qtty = (potential_reactive_quantity - tffe_qtty) / 2
        lactoferrin.field['Lactoferrin'] -= tffe_qtty + tffe2_qtty
        lactoferrin.field['LactoferrinFe'] += tffe_qtty
        lactoferrin.field['LactoferrinFe2'] += tffe2_qtty
        iron.field -= potential_reactive_quantity

        # Degrade Lactoferrin
        turnover(
            field=lactoferrin.field['Lactoferrin'],
            system_concentration=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        turnover(
            field=lactoferrin.field['LactoferrinFe'],
            system_concentration=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        turnover(
            field=lactoferrin.field['LactoferrinFe2'],
            system_concentration=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of lactoferrin
        for component in {'Lactoferrin', 'LactoferrinFe', 'LactoferrinFe2'}:
            lactoferrin.field[component][:] = apply_mesh_diffusion_crank_nicholson(
                variable=lactoferrin.field[component],
                cn_a=lactoferrin.cn_a,
                cn_b=lactoferrin.cn_b,
                dofs=lactoferrin.dofs,
            )

        assert np.alltrue(lactoferrin.field['Lactoferrin'] >= 0.0)
        assert np.alltrue(lactoferrin.field['LactoferrinFe'] >= 0.0)
        assert np.alltrue(lactoferrin.field['LactoferrinFe2'] >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        lactoferrin: LactoferrinState = state.lactoferrin
        mesh: TetrahedralMesh = state.mesh

        concentration_0fe = (
            mesh.integrate_point_function(lactoferrin.field['Lactoferrin'])
            / 1e9
            / mesh.total_volume
        )
        concentration_1fe = (
            mesh.integrate_point_function(lactoferrin.field['LactoferrinFe'])
            / 1e9
            / mesh.total_volume
        )
        concentration_2fe = (
            mesh.integrate_point_function(lactoferrin.field['LactoferrinFe2'])
            / 1e9
            / mesh.total_volume
        )

        concentration = concentration_0fe + concentration_1fe + concentration_2fe

        return {
            'concentration (nM)': float(concentration),
            '+0Fe concentration (nM)': float(concentration_0fe),
            '+1Fe concentration (nM)': float(concentration_1fe),
            '+2Fe concentration (nM)': float(concentration_2fe),
        }

    def visualization_data(self, state: State):
        lactoferrin: LactoferrinState = state.lactoferrin

        return 'molecule', lactoferrin.field
