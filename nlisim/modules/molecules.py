import json

import attr
import numpy as np
from scipy.ndimage import convolve

# from nlisim.coordinates import Voxel
# from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.geometry import GeometryState, TissueTypes
from nlisim.molecule import MoleculeGrid, MoleculeTypes
from nlisim.state import State


def molecule_grid_factory(self: 'MoleculesState'):
    return MoleculeGrid(grid=self.global_state.grid)


@attr.s(kw_only=True, repr=False)
class MoleculesState(ModuleState):
    grid: MoleculeGrid = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    diffusion_rate: float
    cyto_evap_m: float
    cyto_evap_n: float
    iron_max: float


class Molecules(ModuleModel):
    name = 'molecules'
    StateClass = MoleculesState

    def initialize(self, state: State):
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry

        # check if the geometry array is empty
        if not np.any(geometry.lung_tissue):
            raise RuntimeError('geometry molecule has to be initialized first')

        molecules.diffusion_rate = self.config.getfloat('diffusion_rate')
        molecules.cyto_evap_m = self.config.getfloat('cyto_evap_m')
        molecules.cyto_evap_n = self.config.getfloat('cyto_evap_n')
        molecules.iron_max = self.config.getfloat('iron_max')

        molecules_config = self.config.get('molecules')
        json_config = json.loads(molecules_config)

        for molecule in json_config:
            name = molecule['name']
            init_val = molecule['init_val']
            init_loc = molecule['init_loc']

            if name not in [e.name for e in MoleculeTypes]:
                raise TypeError(f'Molecule {name} is not implemented yet')

            for loc in init_loc:
                if loc not in [e.name for e in TissueTypes]:
                    raise TypeError(f'Cannot find lung tissue type {loc}')

            molecules.grid.append_molecule_type(name)

            for loc in init_loc:
                molecules.grid.concentrations[name][
                    np.where(geometry.lung_tissue == TissueTypes[loc].value)
                ] = init_val

            if 'source' in molecule:
                source = molecule['source']
                incr = molecule['incr']
                if source not in [e.name for e in TissueTypes]:
                    raise TypeError(f'Cannot find lung tissue type {source}')

                molecules.grid.sources[name][
                    np.where(geometry.lung_tissue == TissueTypes[init_loc[0]].value)
                ] = incr

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        molecules: MoleculesState = state.molecules

        # iron = molecules.grid['iron']
        # with open('testfile.txt', 'w') as outfile:
        #     for data_slice in iron:
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')

        # for molecule in molecules.grid.types:
        #    self.degrade(molecules.grid[molecule])
        #    self.diffuse(molecules.grid[molecule], state.grid, state.geometry.lung_tissue)

        # self.diffuse_iron(
        #     molecules.grid['iron'], state.grid, state.geometry.lung_tissue, molecules.iron_max
        # )

        # self.degrade(molecules.grid['m_cyto'], molecules.cyto_evap_m)
        # self.diffuse(molecules.grid['m_cyto'], state.grid, state.geometry.lung_tissue)

        # self.degrade(molecules.grid['n_cyto'], molecules.cyto_evap_n)
        # self.diffuse(molecules.grid['n_cyto'], state.grid, state.geometry.lung_tissue)

        molecules.grid.incr()
        self.convolution_diffusion(
            molecules.grid['iron'], state.geometry.lung_tissue, molecules.iron_max
        )

        self.degrade(molecules.grid['m_cyto'], molecules.cyto_evap_m)
        self.convolution_diffusion(molecules.grid['m_cyto'], state.geometry.lung_tissue)

        self.degrade(molecules.grid['n_cyto'], molecules.cyto_evap_n)
        self.convolution_diffusion(molecules.grid['n_cyto'], state.geometry.lung_tissue)

        return state

    @classmethod
    def convolution_diffusion(cls, molecule: np.ndarray, tissue: np.ndarray, threshold=None):
        if len(molecule.shape) != 3:
            raise ValueError(f'Expecting a 3d array. Got dim = {len(molecule.shape)}')
        weights = np.full((3, 3, 3), 1 / 27)
        molecule[:] = convolve(molecule, weights, mode='constant')

        molecule[(tissue == TissueTypes.AIR.value)] = 0

        if threshold:
            molecule[molecule > threshold] = threshold

    @classmethod
    def degrade(cls, molecule: np.ndarray, evap: float):
        molecule *= 1 - evap

    # @classmethod
    # def degrade(cls, molecule: np.ndarray, evap: float):
    #     # TODO These 2 functions should be implemented for all moleculess
    #     # the rest of the behavior (uptake, secretion, etc.) should be
    #     # handled in the cell specific module.

    #     for index in np.argwhere(molecule > 0):
    #         z = index[0]
    #         y = index[1]
    #         x = index[2]

    #         molecule[z, y, x] = molecule[z, y, x] * (1 - evap)

    #     return

    # @classmethod
    # def diffuse_iron(cls, iron: np.ndarray, grid: RectangularGrid, tissue, iron_max):
    #     # TODO These 2 functions should be implemented for all moleculess
    #     # the rest of the behavior (uptake, secretion, etc.) should be
    #     # handled in the cell specific module.
    #     for index in np.argwhere(tissue == TissueTypes.BLOOD.value):
    #         iron[index[0], index[1], index[2]]
    #      = min([iron[index[0], index[1], index[2]], iron_max])

    #     temp = np.zeros(iron.shape)

    #     x_r = [-1, 0, 1]
    #     y_r = [-1, 0, 1]
    #     z_r = [-1, 0, 1]

    #     for index in np.argwhere(temp == 0):
    #         for x in x_r:
    #             for y in y_r:
    #                 for z in z_r:
    #                     zk = index[0] + z
    #                     yj = index[1] + y
    #                     xi = index[2] + x

    #                     if grid.is_valid_voxel(Voxel(x=xi, y=yj, z=zk)):
    #                         temp[index[0], index[1], index[2]] += iron[zk, yj, xi] / 26

    #         if tissue[index[0], index[1], index[2]] == TissueTypes.AIR.value:
    #             temp[index[0], index[1], index[2]] = 0

    #     iron[:] = temp[:]

    #     return

    # @classmethod
    # def diffuse(cls, molecule: np.ndarray, grid: RectangularGrid, tissue):
    #     # TODO These 2 functions should be implemented for all moleculess
    #     # the rest of the behavior (uptake, secretion, etc.) should be
    #     # handled in the cell specific module.
    #     temp = np.zeros(molecule.shape)

    #     x_r = [-1, 0, 1]
    #     y_r = [-1, 0, 1]
    #     z_r = [-1, 0, 1]

    #     for index in np.argwhere(temp == 0):
    #         for x in x_r:
    #             for y in y_r:
    #                 for z in z_r:
    #                     zk = index[0] + z
    #                     yj = index[1] + y
    #                     xi = index[2] + x

    #                     if grid.is_valid_voxel(Voxel(x=xi, y=yj, z=zk)):
    #                         temp[index[0], index[1], index[2]] += molecule[zk, yj, xi] / 26

    #         if tissue[index[0], index[1], index[2]] == TissueTypes.AIR.value:
    #             temp[index[0], index[1], index[2]] = 0

    #     molecule[:] = temp[:]

    #     return
