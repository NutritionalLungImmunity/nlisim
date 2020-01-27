import json

import attr
import numpy as np

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
            if name not in [e.name for e in MoleculeTypes]:
                raise TypeError(f'Molecule {name} is not implemented yet')

            elif init_loc not in [e.name for e in TissueTypes]:
                raise TypeError(f'Cannot find lung tissue type {init_loc}')

            molecules.grid.concentrations[name][
                np.where(geometry.lung_tissue == TissueTypes[init_loc].value)
            ] = init_val

            if 'source' in molecule:
                source = molecule['source']
                incr = molecule['incr']
                if source not in [e.name for e in TissueTypes]:
                    raise TypeError(f'Cannot find lung tissue type {source}')

                molecules.grid.sources[name][
                    np.where(geometry.lung_tissue == TissueTypes[init_loc].value)
                ] = incr

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        molecules: MoleculesState = state.molecules

        # iron = molecules.grid['iron']
        # with open('testfile.txt', 'w') as outfile:
        #     for data_slice in iron:
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')

        for molecule in MoleculeTypes:
            self.diffuse(molecules.grid[molecule.name])
            self.degrade(molecules.grid[molecule.name])
        molecules.grid.incr()

        return state

    @classmethod
    def diffuse(cls, molecule: np.ndarray):
        # TODO These 2 functions should be implemented for all moleculess
        # the rest of the behavior (uptake, secretion, etc.) should be
        # handled in the cell specific module.
        return

    @classmethod
    def degrade(cls, molecule: np.ndarray):
        # TODO These 2 functions should be implemented for all moleculess
        # the rest of the behavior (uptake, secretion, etc.) should be
        # handled in the cell specific module.
        return
