from time import time

import numpy as np
import pylab

from simulation.state import State

_last_draw = 0.0


def display(state: State, block: bool = False):
    c = state.concentration
    # C = np.ma.masked_where(np.abs(state.concentration) < 1e-6, state.concentration)
    x = np.arange(c.shape[1] + 1) * state.dx
    y = np.arange(c.shape[0] + 1) * state.dy
    pylab.clf()
    pylab.pcolormesh(x, y, c, cmap='hot')
    pylab.colorbar()
    pylab.axis('scaled')
    pylab.title('%.2f' % state.time)
    pylab.draw()
    pylab.pause(0.001)
    if block:
        pylab.show()


def draw_simulation(state: State):
    global _last_draw

    draw_interval = state.config.getfloat('simulation.plot', 'draw_interval', fallback=0.5)
    block = state.config.getboolean('simulation.plot', 'block', fallback=False)
    now = time()
    if now - _last_draw >= draw_interval:
        display(state, block=block)
        _last_draw = time()

    return state
