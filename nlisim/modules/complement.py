import itertools
from typing import Any, Dict

import attr
import numpy as np
from scipy.integrate import solve_ivp

from nlisim.diffusion import apply_diffusion
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State
from nlisim.util import turnover_rate


def minimal_model(t, var, k, ks, d, FD, FI):
    c3, c3b, c3bb_c, c3bb_o, c3bbb, fb, fh, c3bh, c3bbbh = var

    dC3 = ks[1] - d[1] * c3 - k[1] * c3 - (k[2] * c3bbb * c3) / (k[3] + c3)
    dC3b = (
        k[1] * c3
        + (k[2] * c3bbb * c3) / (k[3] + c3)
        - k[4] * c3b * fb
        + k[5] * c3bb_c
        + k[6] * c3bbb
        - k[15] * c3b * fh
        + k[16] * c3bh
        + k[21] * c3bbbh
    )
    dC3bB_c = k[4] * c3b * fb - k[5] * c3bb_c - k[9] * c3bb_c + k[10] * c3bb_o
    dC3bB_o = k[9] * c3bb_c - k[10] * c3bb_o - (k[7] * FD * c3bb_o) / (k[8] + c3bb_o)
    dC3bBb = (
        (k[7] * FD * c3bb_o) / (k[8] + c3bb_o) - k[6] * c3bbb - k[25] * c3bbb * fh + k[16] * c3bbbh
    )
    dFB = ks[2] - d[2] * fb - k[4] * c3b * fb + k[5] * c3bb_c
    dFH = (
        ks[3]
        - d[3] * fh
        - k[15] * c3b * fh
        + k[16] * c3bh
        - k[25] * c3bbb * fh
        + k[16] * c3bbbh
        + (k[19] * c3bh * FI) / (k[20] + c3bh)
        + k[21] * c3bbbh
    )
    dC3bH = k[15] * c3b * fh - k[16] * c3bh - (k[19] * c3bh * FI) / (k[20] + c3bh)
    dC3bBbH = k[25] * c3bbb * fh - k[16] * c3bbbh - k[21] * c3bbbh

    return np.array([dC3, dC3b, dC3bB_c, dC3bB_o, dC3bBb, dFB, dFH, dC3bH, dC3bBbH])

def jacobian(t, var, k, ks, d, FD, FI):
    C3, C3b, C3bB_c, C3bB_o, C3bBb, FB, FH, C3bH, C3bBbH = var

    return np.array(
        [
            [
                -d[1] - k[1] - C3bBb * k[2] / (C3 + k[3]) + C3 * C3bBb * k[2] / (C3 + k[3]) ** 2,
                k[1] + C3bBb * k[2] / (C3 + k[3]) - C3 * C3bBb * k[2] / (C3 + k[3]) ** 2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                -FH * k[15] - FB * k[4],
                FB * k[4],
                0,
                0,
                -FB * k[4],
                -FH * k[15],
                FH * k[15],
                0,
            ],
            [0, k[5], -k[5] - k[9], k[9], 0, k[5], 0, 0, 0],
            [
                0,
                0,
                k[10],
                -k[10] - FD * k[7] / (C3bB_o + k[8]) + C3bB_o * FD * k[7] / (C3bB_o + k[8]) ** 2,
                FD * k[7] / (C3bB_o + k[8]) - C3bB_o * FD * k[7] / (C3bB_o + k[8]) ** 2,
                0,
                0,
                0,
                0,
            ],
            [
                -C3 * k[2] / (C3 + k[3]),
                C3 * k[2] / (C3 + k[3]) + k[6],
                0,
                0,
                -FH * k[25] - k[6],
                0,
                -FH * k[25],
                0,
                FH * k[25],
            ],
            [0, -C3b * k[4], C3b * k[4], 0, 0, -C3b * k[4] - d[2], 0, 0, 0],
            [
                0,
                -C3b * k[15],
                0,
                0,
                -C3bBb * k[25],
                0,
                -C3b * k[15] - C3bBb * k[25] - d[3],
                C3b * k[15],
                C3bBb * k[25],
            ],
            [
                0,
                k[16],
                0,
                0,
                0,
                0,
                k[16] + FI * k[19] / (C3bH + k[20]) - C3bH * FI * k[19] / (C3bH + k[20]) ** 2,
                -k[16] - FI * k[19] / (C3bH + k[20]) + C3bH * FI * k[19] / (C3bH + k[20]) ** 2,
                0,
            ],
            [0, k[21], 0, 0, k[16], 0, k[16] + k[21], 0, -k[16] - k[21]],
        ], dtype=np.float64
    ).T


def molecule_grid_factory(self: 'ComplementState') -> np.ndarray:
    return np.zeros(
        shape=self.global_state.grid.shape,
        dtype=[
            ("C3", np.float64),
            ("C3b", np.float64),
            ("C3bB_c", np.float64),
            ("C3bB_o", np.float64),
            ("C3bBb", np.float64),
            ("FB", np.float64),
            ("FH", np.float64),
            ("C3bH", np.float64),
            ("C3bBbH", np.float64),
        ],
    )


@attr.s(kw_only=True, repr=False)
class ComplementState(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mols
    FD: float  # units: TODO
    FI: float  # units: TODO
    k1: float  # units: TODO
    k2: float  # units: TODO
    k3: float  # units: TODO
    k4: float  # units: TODO
    k5: float  # units: TODO
    k6: float  # units: TODO
    k7: float  # units: TODO
    k8: float  # units: TODO
    k9: float  # units: TODO
    k10: float  # units: TODO
    k15: float  # units: TODO
    k16: float  # units: TODO
    k19: float  # units: TODO
    k20: float  # units: TODO
    k21: float  # units: TODO
    k25: float  # units: TODO
    ks1: float  # units: TODO
    ks2: float  # units: TODO
    ks3: float  # units: TODO
    d1: float  # units: TODO
    d2: float  # units: TODO
    d3: float  # units: TODO
    fd: float  # units: TODO
    fi: float  # units: TODO
    init_C3: float
    init_C3b: float
    init_C3bB_c: float
    init_C3bB_o: float
    init_C3bBb: float
    init_FB: float
    init_FH: float
    init_C3bH: float
    init_C3bBbH: float


class Complement(ModuleModel):
    """Complement"""

    name = 'complement'
    StateClass = ComplementState

    def initialize(self, state: State) -> State:
        complement: ComplementState = state.complement
        # voxel_volume: float = state.voxel_volume  # units: L

        # config file values
        complement.FD = self.config.getfloat('FD')
        complement.FI = self.config.getfloat('FI')
        complement.k1 = self.config.getfloat('k1')
        complement.k2 = self.config.getfloat('k2')
        complement.k3 = self.config.getfloat('k3')
        complement.k4 = self.config.getfloat('k4')
        complement.k5 = self.config.getfloat('k5')
        complement.k6 = self.config.getfloat('k6')
        complement.k7 = self.config.getfloat('k7')
        complement.k8 = self.config.getfloat('k8')
        complement.k9 = self.config.getfloat('k9')
        complement.k10 = self.config.getfloat('k10')
        complement.k15 = self.config.getfloat('k15')
        complement.k16 = self.config.getfloat('k16')
        complement.k19 = self.config.getfloat('k19')
        complement.k20 = self.config.getfloat('k20')
        complement.k21 = self.config.getfloat('k21')
        complement.k25 = self.config.getfloat('k25')

        complement.ks1 = self.config.getfloat('ks1')
        complement.ks2 = self.config.getfloat('ks2')
        complement.ks3 = self.config.getfloat('ks3')

        complement.d1 = self.config.getfloat('d1')
        complement.d2 = self.config.getfloat('d2')
        complement.d3 = self.config.getfloat('d3')

        complement.fd = self.config.getfloat('FD')
        complement.fi = self.config.getfloat('FI')

        complement.init_C3 = self.config.getfloat('init_C3')
        complement.grid['C3'][:] = complement.init_C3
        complement.init_C3b = self.config.getfloat('init_C3b')
        complement.grid['C3b'][:] = complement.init_C3b
        complement.init_C3bB_c = self.config.getfloat('init_C3bB_c')
        complement.grid['C3bB_c'][:] = complement.init_C3bB_c
        complement.init_C3bB_o = self.config.getfloat('init_C3bB_o')
        complement.grid['C3bB_o'][:] = complement.init_C3bB_o
        complement.init_C3bBb = self.config.getfloat('init_C3bBb')
        complement.grid['C3bBb'][:] = complement.init_C3bBb
        complement.init_FB = self.config.getfloat('init_FB')
        complement.grid['FB'][:] = complement.init_FB
        complement.init_FH = self.config.getfloat('init_FH')
        complement.grid['FH'][:] = complement.init_FH
        complement.init_C3bH = self.config.getfloat('init_C3bH')
        complement.grid['C3bH'][:] = complement.init_C3bH
        complement.init_C3bBbH = self.config.getfloat('init_C3bBbH')
        complement.grid['C3bBbH'][:] = complement.init_C3bBbH

        # computed values

        return state

    # noinspection SpellCheckingInspection
    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.molecules import MoleculesState

        complement: ComplementState = state.complement
        molecules: MoleculesState = state.molecules
        # grid: RectangularGrid = state.grid
        # voxel_volume = state.voxel_volume

        species = ["C3", "C3b", "C3bB_c", "C3bB_o", "C3bBb", "FB", "FH", "C3bH", "C3bBbH"]

        k = np.zeros(26, dtype=np.float64)
        k[1] = complement.k1
        k[2] = complement.k2
        k[3] = complement.k3
        k[4] = complement.k4
        k[5] = complement.k5
        k[6] = complement.k6
        k[7] = complement.k7
        k[8] = complement.k8
        k[9] = complement.k9
        k[10] = complement.k10
        k[15] = complement.k15
        k[16] = complement.k16
        k[19] = complement.k19
        k[20] = complement.k20
        k[21] = complement.k21
        k[25] = complement.k25

        ks = np.zeros(4, dtype=np.float64)
        ks[1] = complement.ks1
        ks[2] = complement.ks2
        ks[3] = complement.ks3

        d = np.zeros(4, dtype=np.float64)
        d[1] = complement.d1
        d[2] = complement.d2
        d[3] = complement.d3

        # TODO: find or write a vectorized version. solve_ivp doesn't do that.
        for i,j,k in itertools.product(*map(range,complement.grid.shape)):
            result = solve_ivp(
                minimal_model,
                t_span=(previous_time, previous_time + self.time_step),
                y0=np.array([complement.grid[spec][i,j,k] for spec in species], dtype=np.float64),
                args=(k, ks, d, complement.fd, complement.fi),
                method='BDF',
                jac=jacobian,
            )
            for idx, spec in enumerate(species):
                complement.grid[spec][i,j,k] = result.y[idx]

        # TODO: do we want this?
        # # Degrade Complement proteins
        # # Note: ideally, this would be a constant computed in initialize, but we would have to
        # # know that "molecules" is initialized first
        # trnvr_rt = turnover_rate(
        #     x=np.array(1.0, dtype=np.float64),
        #     x_system=0.0,
        #     base_turnover_rate=molecules.turnover_rate,
        #     rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        # )
        # for spec in species:
        #     complement.grid[spec] *= trnvr_rt

        # TODO: do we want this?
        # # Diffusion of complement proteins
        # for spec in species:
        #     complement.grid[spec][:] = apply_diffusion(
        #         variable=complement.grid[spec],
        #         laplacian=molecules.laplacian,
        #         diffusivity=molecules.diffusion_constant,
        #         dt=self.time_step,
        #     )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        complement: ComplementState = state.complement
        voxel_volume = state.voxel_volume

        species = ["C3", "C3b", "C3bB_c", "C3bB_o", "C3bBb", "FB", "FH", "C3bH", "C3bBbH"]

        concentrations = [np.mean(complement.grid[spec]) / voxel_volume / 1e9 for spec in species]

        return {
            spec + ' concentration (nM)': float(concentration)
            for spec, concentration in zip(species, concentrations)
        }

    def visualization_data(self, state: State):
        complement: ComplementState = state.complement

        return 'molecule', complement.grid
