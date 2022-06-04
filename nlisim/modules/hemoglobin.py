from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.molecule_maker import MoleculeFactory, MoleculeModel
from nlisim.state import State


# noinspection PyUnusedLocal
def afumigatus_uptakes_iron_from_hemoglobin(state: State, hemoglobin_model: MoleculeModel):
    from nlisim.modules.afumigatus import AfumigatusCellData, AfumigatusCellStatus, AfumigatusState

    hemoglobin: HemoglobinState = state.hemoglobin
    afumigatus: AfumigatusState = state.afumigatus
    grid: RectangularGrid = state.grid

    # afumigatus uptakes iron from hemoglobin
    for afumigatus_cell_index in afumigatus.cells.alive():
        afumigatus_cell: AfumigatusCellData = afumigatus.cells[afumigatus_cell_index]
        if afumigatus_cell['status'] in {
            AfumigatusCellStatus.HYPHAE,
            AfumigatusCellStatus.GERM_TUBE,
        }:
            afumigatus_cell_voxel: Voxel = grid.get_voxel(afumigatus_cell['point'])
            fungal_absorbed_hemoglobin = (
                hemoglobin.uptake_rate * hemoglobin.grid[tuple(afumigatus_cell_voxel)]
            )
            hemoglobin.grid[tuple(afumigatus_cell_voxel)] -= fungal_absorbed_hemoglobin
            afumigatus_cell['iron_pool'] += 4 * fungal_absorbed_hemoglobin

    return state


HemoglobinState, Hemoglobin = (
    MoleculeFactory('hemoglobin')
    .add_config_field('uptake_rate', float)
    .add_config_field('ma_heme_import_rate', float)
    .add_advance_action(action=afumigatus_uptakes_iron_from_hemoglobin, order=0)
    .add_degradation(order=1)
    .add_diffusion(order=2)
    .build()
)
