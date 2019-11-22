from math import floor
from time import time

import attr
import numpy as np
import pylab

from simulation.config import SimulationConfig
from simulation.module import Module, ModuleState
from simulation.state import State


class Plot2dSlice(Module):
    name = 'plot2d_slice'
    defaults = {
        'draw_interval': '0.1',
        'block': 'False',
        'mask_threshold': '0',
        'z_plane': '0.5',
        'variables': '',
        'cmap': 'hot',
    }

    @attr.s(kw_only=True)
    class StateClass(ModuleState):
        last_draw: float = attr.ib(default=0)

    @classmethod
    def display(
        cls,
        state: State,
        z_plane: float,
        variable: str,
        block: bool = False,
        mask_threshold: float = 0,
        cmap: str = 'hot',
    ) -> None:
        module_name, var_name = variable.split('.')
        var = getattr(getattr(state, module_name), var_name)
        iz = floor(z_plane * (var.shape[0] - 1))
        slice = var[iz, ...]

        masked = np.ma.masked_where(np.abs(slice) < mask_threshold, slice)
        x = state.grid.xv
        y = state.grid.yv
        pylab.clf()
        pylab.pcolormesh(x, y, masked, cmap=cmap)
        pylab.colorbar()
        pylab.axis('scaled')
        pylab.title('%.2f' % state.time)
        pylab.draw()
        pylab.pause(0.001)
        if block:
            print('\nPress "q" in the plot window to continue')
            pylab.show()

    def advance(self, state: State, previous_time: float) -> State:
        draw_interval = self.config.getfloat('draw_interval')
        block = self.config.getboolean('block')
        mask_threshold = self.config.getfloat('mask_threshold')
        cmap = self.config.get('cmap')
        z_plane = self.config.getfloat('z_plane')
        variables = SimulationConfig.parselist(self.config.get('variables'))

        now = time()
        if now - state.plot2d_slice.last_draw >= draw_interval:
            for variable in variables:
                self.display(
                    state, z_plane, variable, block=block, cmap=cmap, mask_threshold=mask_threshold
                )
                state.plot2d_slice.last_draw = time()

        return state
