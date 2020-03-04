import json

import attr
import numpy as np

from simulation.diffusion import apply_diffusion
from simulation.module import Module, ModuleState
from simulation.modules.geometry import GeometryState, TissueTypes
from simulation.molecule import MoleculeGrid, MoleculeTypes
from simulation.state import State


def molecule_grid_factory(self: 'MoleculesState'):
    return MoleculeGrid(grid=self.global_state.grid)


@attr.s(kw_only=True, repr=False)
class MoleculesState(ModuleState):
    grid: MoleculeGrid = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))


class Molecules(Module):
    name = 'molecules'
    StateClass = MoleculesState

    def initialize(self, state: State):
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry

        # check if the geometry array is empty
        if not np.any(geometry.lung_tissue):
            raise RuntimeError('geometry molecule has to be initialized first')

        molecules_config = self.config.get('molecules')
        json_config = json.loads(molecules_config)

        for molecule in json_config:
            name = molecule['name']
            init_val = molecule['init_val']
            init_loc = molecule['init_loc']
            diffusivity = molecule['diffusivity']

            if name not in [e.name for e in MoleculeTypes]:
                raise TypeError(f'Molecule {name} is not implemented yet')

            for loc in init_loc:
                if loc not in [e.name for e in TissueTypes]:
                    raise TypeError(f'Cannot find lung tissue type {loc}')

            molecules.grid.append_molecule_type(name)

            molecules.grid.set_diffusivity(name, diffusivity)

            for loc in init_loc:
                molecules.grid.concentrations[name][
                    np.where(geometry.lung_tissue == TissueTypes[loc].value)
                ] = init_val

                # test
                molecules.grid.concentrations[name][44, 76, 25] = 1000

            if 'source' in molecule:
                source = molecule['source']
                incr = molecule['incr']
                if source not in [e.name for e in TissueTypes]:
                    raise TypeError(f'Cannot find lung tissue type {source}')

                molecules.grid.sources[name][
                    np.where(geometry.lung_tissue == TissueTypes[source].value)
                ] = incr

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry

        surf_lapl = geometry.laplacian_matrix['surf_lapl']
        dt = state.time - previous_time

        # iron = molecules.grid['iron']
        # with open('testfile.txt', 'w') as outfile:
        #     for data_slice in iron:
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')

        for molecule in molecules.grid.types:
            diffusivity = molecules.grid.diffusivity[molecule]
            molecule_grid = molecules.grid[molecule]
            molecule_grid[:] = apply_diffusion(molecules.grid[molecule], surf_lapl, diffusivity, dt)
            molecule_grid[molecule_grid < 1e-10] = 0

            self.degrade(molecules.grid[molecule])
            # molecules.grid[molecule][:] = np.maximum(0, molecules.grid[molecule])
            # assert(molecules.grid[molecule].all() >= 0)
            # print(np.sum(molecules.grid[molecule]))
        molecules.grid.incr()

        return state

    @classmethod
    def degrade(cls, molecule: np.ndarray):
        # TODO These 2 functions should be implemented for all moleculess
        # the rest of the behavior (uptake, secretion, etc.) should be
        # handled in the cell specific module.
        return
