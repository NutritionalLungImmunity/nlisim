from functools import reduce
import math
from operator import mul

from attr import attrs
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

from nlisim.diffusion import discrete_laplacian
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State


@attrs(kw_only=True, repr=False)
class MoleculesState(ModuleState):
    turnover_rate: float
    cyt_bind_t: float
    rel_cyt_bind_unit_t: float
    diffusion_constant_timestep: float
    implicit_euler_matrix: np.ndarray


class Molecules(ModuleModel):
    name = 'molecules'
    StateClass = MoleculesState

    # noinspection SpellCheckingInspection
    def initialize(self, state: State):
        from nlisim.util import TissueType

        molecules: MoleculesState = state.molecules

        molecules.cyt_bind_t = self.config.getfloat('cyt_bind_t')
        molecules.turnover_rate = self.config.getfloat('turnover_rate')

        # TODO: there is an implicit assumption, below, that all molecules are on the
        #  same time step. This is true on Aug 24, 2021.

        # Computed values
        molecules.rel_cyt_bind_unit_t = self.time_step / molecules.cyt_bind_t
        # TODO: original comments as below. Is the param 0.2?
        #  i.e. ...math.log(1+0.2)... Yes, 20% per hour
        # 0.2 # 10.1124/jpet.118.250134 (approx) 0.2/h CHANGE!!!!
        molecules.turnover_rate = 1 - math.log(1.2) / int(30 / 2.0)  # TODO: hard coded the 2.0 ...
        molecules.diffusion_constant_timestep = (
            self.config.getfloat('diffusion_constant') * self.time_step
        )  # units: (µm^2/min) * (min/step) = µm^2/step

        # construct the laplacian
        grid_cardinality = reduce(mul, state.grid.shape)
        tissue = state.lung_tissue
        # laplacian units: 1/(µm^2)
        laplacian: csr_matrix = discrete_laplacian(grid=state.grid, mask=tissue != TissueType.AIR)

        molecules.implicit_euler_matrix = (
            scipy.sparse.identity(grid_cardinality)
            - molecules.diffusion_constant_timestep * 1 * laplacian
        )  # units: (µm^2/step) * (step) * (1/(µm^2)) = 1

        # Note: Henrique's code did things a little differently here:
        #  1) three 1D laplacians instead of one 3D laplacian
        #  2) four implicit steps instead of one

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        return state


class MoleculeModel(ModuleModel):
    @staticmethod
    def diffuse(grid: np.ndarray, state: State, tolerance: float = 1e-64):
        molecules: MoleculesState = state.molecules

        var_next, info = cg(molecules.implicit_euler_matrix, grid.ravel(), tol=tolerance)
        if info != 0:
            raise Exception(f'GMRES failed ({info})')

        grid[:] = var_next.reshape(grid.shape)
