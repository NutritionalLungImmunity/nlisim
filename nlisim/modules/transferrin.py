from typing import Any, Dict, cast

import attr
from attr import attrib, attrs
import numpy as np
from scipy.sparse import csr_matrix

from nlisim.diffusion import (
    apply_mesh_diffusion_crank_nicholson,
    assemble_mesh_laplacian_crank_nicholson,
)
from nlisim.grid import TetrahedralMesh, secrete_in_element, uptake_in_element
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State
from nlisim.util import iron_tf_reaction, logger


def molecule_point_field_factory(self: 'TransferrinState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(
        dtype=[('Tf', np.float64), ('TfFe', np.float64), ('TfFe2', np.float64)]
    )


@attrs(kw_only=True, repr=False)
class TransferrinState(ModuleState):
    field: np.ndarray = attrib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    p1: float
    p2: float
    p3: float
    threshold_log_hep: float  # units: log(atto-mols)
    default_apotf_rel_concentration: float  # units: proportion
    default_tffe_rel_concentration: float  # units: proportion
    default_tffe2_rel_concentration: float  # units: proportion
    default_tf_concentration: float  # units: atto-M
    default_apotf_concentration: float  # units: atto-M
    default_tffe_concentration: float  # units: atto-M
    default_tffe2_concentration: float  # units: atto-M
    tf_intercept: float  # units: atto-M
    tf_slope: float  # units: atto-M
    ma_iron_import_rate: float  # units: L * cell^-1 * h^-1
    ma_iron_import_rate_unit_t: float  # units: L * cell^-1 * step^-1
    ma_iron_export_rate: float  # units: L * mol^-1 * cell^-1 * h^-1
    ma_iron_export_rate_unit_t: float  # units: L * mol^-1 * cell^-1 * step^-1
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson


class Transferrin(ModuleModel):
    """Transferrin"""

    name = 'transferrin'
    StateClass = TransferrinState

    def initialize(self, state: State) -> State:
        logger.info("Initializing " + self.name)
        transferrin: TransferrinState = state.transferrin
        mesh: TetrahedralMesh = state.mesh

        # config file values
        transferrin.p1 = self.config.getfloat('p1')
        transferrin.p2 = self.config.getfloat('p2')
        transferrin.p3 = self.config.getfloat('p3')
        transferrin.threshold_log_hep = self.config.getfloat('threshold_log_hep')

        transferrin.tf_intercept = self.config.getfloat('tf_intercept')
        transferrin.tf_slope = self.config.getfloat('tf_slope')

        transferrin.default_apotf_rel_concentration = self.config.getfloat(
            'default_apotf_rel_concentration'
        )  # units: proportion
        transferrin.default_tffe_rel_concentration = self.config.getfloat(
            'default_tffe_rel_concentration'
        )  # units: proportion
        transferrin.default_tffe2_rel_concentration = self.config.getfloat(
            'default_tffe2_rel_concentration'
        )  # units: proportion

        transferrin.ma_iron_import_rate = self.config.getfloat(
            'ma_iron_import_rate_vol'
        )  # units: L * cell^-1 * h^-1
        transferrin.ma_iron_export_rate = self.config.getfloat(
            'ma_iron_export_rate_vol'
        )  # units: L * mol^-1 * cell^-1 * h^-1

        transferrin.diffusion_constant = self.config.getfloat(
            'diffusion_constant'
        )  # units: µm^2/min

        # computed values
        transferrin.default_tf_concentration = (
            transferrin.tf_intercept + transferrin.tf_slope * transferrin.threshold_log_hep
        )  # based on y--log(x) plot. units: aM * L = aM
        logger.info(f"Computed {transferrin.default_tf_concentration=}")
        transferrin.default_apotf_concentration = (
            transferrin.default_apotf_rel_concentration * transferrin.default_tf_concentration
        )  # units: atto-M
        logger.info(f"Computed {transferrin.default_apotf_concentration=}")
        transferrin.default_tffe_concentration = (
            transferrin.default_tffe_rel_concentration * transferrin.default_tf_concentration
        )  # units: atto-M
        logger.info(f"Computed {transferrin.default_tffe_concentration=}")
        transferrin.default_tffe2_concentration = (
            transferrin.default_tffe2_rel_concentration * transferrin.default_tf_concentration
        )  # units: atto-M
        logger.info(f"Computed {transferrin.default_tffe2_concentration=}")

        transferrin.ma_iron_import_rate_unit_t = transferrin.ma_iron_import_rate / (
            self.time_step / 60
        )  # units: L * cell^-1 * step^-1
        logger.info(f"Computed {transferrin.ma_iron_import_rate_unit_t=}")
        transferrin.ma_iron_export_rate_unit_t = transferrin.ma_iron_export_rate / (
            self.time_step / 60
        )  # units: L * mol^-1 * cell^-1 * step^-1
        logger.info(f"Computed {transferrin.ma_iron_export_rate_unit_t=}")

        # matrices for diffusion
        cn_a, cn_b = assemble_mesh_laplacian_crank_nicholson(
            laplacian=mesh.laplacian, diffusivity=transferrin.diffusion_constant, dt=self.time_step
        )
        transferrin.cn_a = cn_a
        transferrin.cn_b = cn_b

        # initialize the molecular field
        transferrin.field['Tf'].fill(transferrin.default_apotf_concentration)
        transferrin.field['TfFe'].fill(transferrin.default_tffe_concentration)
        transferrin.field['TfFe2'].fill(transferrin.default_tffe2_concentration)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        logger.info("Advancing " + self.name + f" from t={previous_time}")

        from nlisim.modules.iron import IronState
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.phagocyte import PhagocyteStatus

        transferrin: TransferrinState = state.transferrin
        iron: IronState = state.iron
        macrophage: MacrophageState = state.macrophage
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(transferrin.field['Tf'] >= 0.0)
        assert np.alltrue(transferrin.field['TfFe'] >= 0.0)
        assert np.alltrue(transferrin.field['TfFe2'] >= 0.0)

        # logger.debug(f"{np.min(transferrin.field['Tf'])=} "
        #              f"{np.max(transferrin.field['Tf'])=}")
        # logger.debug(f"{np.min(transferrin.field['TfFe'])=} "
        #              f"{np.max(transferrin.field['TfFe'])=}")
        # logger.debug(f"{np.min(transferrin.field['TfFe2'])=} "
        #              f"{np.max(transferrin.field['TfFe2'])=}")

        # interact with macrophages
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_element_index: int = cast(int, macrophage_cell['element_index'])

            uptake_proportion = np.minimum(
                transferrin.ma_iron_import_rate_unit_t
                / mesh.element_volumes[macrophage_element_index],
                1.0,
            )  # units: L * cell^-1 * step^-1 / L = proportion * cell^-1 * step^-1
            qtty_fe2 = (
                mesh.evaluate_point_function(
                    point_function=transferrin.field['TfFe2'],
                    element_index=macrophage_element_index,
                    point=macrophage_cell['point'],
                )
                * uptake_proportion
                * mesh.element_volumes[macrophage_element_index]
            )  # units: atto-mols * cell^-1 * step^-1
            qtty_fe = (
                mesh.evaluate_point_function(
                    point_function=transferrin.field['TfFe'],
                    element_index=macrophage_element_index,
                    point=macrophage_cell['point'],
                )
                * uptake_proportion
                * mesh.element_volumes[macrophage_element_index]
            )  # units: atto-mols * cell^-1 * step^-1

            assert mesh.in_element(
                element_index=macrophage_element_index, point=macrophage_cell['point']
            ), (
                f"{macrophage_element_index=},\n"
                f"{macrophage_cell['point']=},\n"
                f"{mesh.element_point_indices[macrophage_element_index]=}\n"
                f"{mesh.points[mesh.element_point_indices[macrophage_element_index]]=}"
            )

            # macrophage uptakes iron, leaves transferrin+0Fe behind
            uptake_in_element(
                mesh=mesh,
                point_field=transferrin.field['TfFe2'],
                element_index=macrophage_element_index,
                point=macrophage_cell['point'],
                amount=qtty_fe2,  # units: atto-mols * cell^-1 * step^-1
            )
            uptake_in_element(
                mesh=mesh,
                point_field=transferrin.field['TfFe'],
                element_index=macrophage_element_index,
                point=macrophage_cell['point'],
                amount=qtty_fe,  # units: atto-mols * cell^-1 * step^-1
            )
            secrete_in_element(
                mesh=mesh,
                point_field=transferrin.field['Tf'],
                element_index=macrophage_element_index,
                point=macrophage_cell['point'],
                amount=qtty_fe + qtty_fe2,  # units: atto-mols * cell^-1 * step^-1
            )
            macrophage_cell['iron_pool'] += (
                2 * qtty_fe2 + qtty_fe
            )  # units: atto-mol * cell^-1 * step^-1

            if macrophage_cell['fpn'] and macrophage_cell['status'] not in {
                PhagocyteStatus.ACTIVE,
                PhagocyteStatus.ACTIVATING,
            }:
                # amount of iron to export is bounded by the amount of iron in the cell as well
                # as the amount which can be accepted by transferrin
                transferrin_in_element: float = mesh.integrate_point_function_in_element(
                    element_index=macrophage_element_index,
                    point_function=transferrin.field['Tf'],
                )  # units: atto-mols
                transferrin_fe_in_element: float = mesh.integrate_point_function_in_element(
                    element_index=macrophage_element_index,
                    point_function=transferrin.field['TfFe'],
                )  # units: atto-mols

                # logger.debug(f"{macrophage_cell['iron_pool']=}")
                # logger.debug(f"{transferrin_in_element=}")
                # logger.debug(f"{mesh.element_volumes[macrophage_element_index]=}")
                # logger.debug(f"{transferrin.ma_iron_export_rate_unit_t=}")
                qtty: float = min(
                    cast(float, macrophage_cell['iron_pool']),  # units: atto-mols
                    cast(float, 2 * transferrin_in_element),  # units: atto-mols
                    cast(
                        float,
                        macrophage_cell['iron_pool']
                        * transferrin_in_element
                        / mesh.element_volumes[macrophage_element_index]  # TODO: verify units
                        * transferrin.ma_iron_export_rate_unit_t,
                    ),
                )  # units: 3) atto-mols * atto-mols * L^-1 * (L * mol^-1 * cell^-1 * step^-1)

                rel_tf_fe = iron_tf_reaction(
                    iron=qtty,
                    tf=transferrin_in_element,
                    tf_fe=transferrin_fe_in_element,
                    p1=transferrin.p1,
                    p2=transferrin.p2,
                    p3=transferrin.p3,
                )
                # logger.debug(f"{rel_tf_fe=}")
                # logger.debug(f"{qtty=}")
                tffe_qtty = rel_tf_fe * qtty  # units: atto-mols
                tffe2_qtty = (qtty - tffe_qtty) / 2  # units: atto-mols

                uptake_in_element(
                    mesh=mesh,
                    point_field=transferrin.field['Tf'],
                    element_index=macrophage_element_index,
                    point=macrophage_cell['point'],
                    amount=tffe_qtty + tffe2_qtty,  # units: atto-mols
                )
                secrete_in_element(
                    mesh=mesh,
                    point_field=transferrin.field['TfFe'],
                    element_index=macrophage_element_index,
                    point=macrophage_cell['point'],
                    amount=tffe_qtty,  # units: atto-mols
                )
                secrete_in_element(
                    mesh=mesh,
                    point_field=transferrin.field['TfFe2'],
                    element_index=macrophage_element_index,
                    point=macrophage_cell['point'],
                    amount=tffe2_qtty,  # units: atto-mols
                )
                macrophage_cell['iron_pool'] -= qtty  # units: atto-M * L = atto-mols

        assert np.alltrue(transferrin.field['Tf'] >= 0.0)
        assert np.alltrue(transferrin.field['TfFe'] >= 0.0)
        assert np.alltrue(transferrin.field['TfFe2'] >= 0.0)

        # logger.debug(f"{np.min(transferrin.field['Tf'])=} "
        #              f"{np.max(transferrin.field['Tf'])=}")
        # logger.debug(f"{np.min(transferrin.field['TfFe'])=} "
        #              f"{np.max(transferrin.field['TfFe'])=}")
        # logger.debug(f"{np.min(transferrin.field['TfFe2'])=} "
        #              f"{np.max(transferrin.field['TfFe2'])=}")

        # interaction with iron: transferrin -> transferrin+[1,2]Fe
        transferrin_fe_capacity = 2 * transferrin.field['Tf'] + transferrin.field['TfFe']
        potential_reactive_quantity = np.minimum(iron.field, transferrin_fe_capacity)
        rel_tf_fe = iron_tf_reaction(
            iron=potential_reactive_quantity,
            tf=transferrin.field["Tf"],
            tf_fe=transferrin.field["TfFe"],
            p1=transferrin.p1,
            p2=transferrin.p2,
            p3=transferrin.p3,
        )
        tffe_qtty = rel_tf_fe * potential_reactive_quantity
        tffe2_qtty = (potential_reactive_quantity - tffe_qtty) / 2
        transferrin.field['Tf'] -= tffe_qtty + tffe2_qtty
        transferrin.field['TfFe'] += tffe_qtty
        transferrin.field['TfFe2'] += tffe2_qtty
        iron.field -= potential_reactive_quantity
        # Note: asked Henrique why there is no transferrin+Fe -> transferrin+2Fe reaction
        # answer was that this should already be accounted for

        # Degrade transferrin: done in liver

        assert np.alltrue(transferrin.field['Tf'] >= 0.0)
        assert np.alltrue(transferrin.field['TfFe'] >= 0.0)
        assert np.alltrue(transferrin.field['TfFe2'] >= 0.0)

        # logger.debug(f"{np.min(transferrin.field['Tf'])=} "
        #              f"{np.max(transferrin.field['Tf'])=}")
        # logger.debug(f"{np.min(transferrin.field['TfFe'])=} "
        #              f"{np.max(transferrin.field['TfFe'])=}")
        # logger.debug(f"{np.min(transferrin.field['TfFe2'])=} "
        #              f"{np.max(transferrin.field['TfFe2'])=}")

        # Diffusion of transferrin
        for component in {'Tf', 'TfFe', 'TfFe2'}:
            logger.info(f"diffusing {self.name}:{component}")
            apply_mesh_diffusion_crank_nicholson(
                variable=transferrin.field[component],
                cn_a=transferrin.cn_a,
                cn_b=transferrin.cn_b,
            )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        transferrin: TransferrinState = state.transferrin
        mesh: TetrahedralMesh = state.mesh

        concentration_0fe = (
            mesh.integrate_point_function(transferrin.field['Tf']) / 1e9 / mesh.total_volume
        )
        concentration_1fe = (
            mesh.integrate_point_function(transferrin.field['TfFe']) / 1e9 / mesh.total_volume
        )
        concentration_2fe = (
            mesh.integrate_point_function(transferrin.field['TfFe2']) / 1e9 / mesh.total_volume
        )

        concentration = concentration_0fe + concentration_1fe + concentration_2fe

        return {
            'concentration (nM)': float(concentration),
            '+0Fe concentration (nM)': float(concentration_0fe),
            '+1Fe concentration (nM)': float(concentration_1fe),
            '+2Fe concentration (nM)': float(concentration_2fe),
        }

    def visualization_data(self, state: State):
        transferrin: TransferrinState = state.transferrin
        return 'molecule', transferrin.field
