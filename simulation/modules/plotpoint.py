import attr
import matplotlib.pyplot as plt
import numpy as np

from simulation.module import Module, ModuleState
from simulation.state import State

@attr.s(kw_only=True)
class PlotPointState(ModuleState):
    last_plotpoint: float = attr.ib(default=0)
    num_alive: int = attr.ib(default=0)

class PlotPoint(Module):
    name = 'plotpoint'
    defaults = {
        'plotpoint_interval': '1',
    }
    StateClass = PlotPointState

    def advance(self, state: State, previous_time: float):
        now = state.time
        alive_count = len(state.fungus.cells.alive())

        if now - state.plotpoint.last_plotpoint > 1 - 1e-8:
            state.plotpoint.num_alive = alive_count
            state.plotpoint.last_plotpoint = now

        return state

def create_plot(states):
    times = []
    fungal_burden = []
    for state in states:
        times.append(state.time)
        fungal_burden.append(state.plotpoint.num_alive)

    fig, ax = plt.subplots()
    plt.plot(times,fungal_burden)
    ax.set(xlabel='Time', ylabel='Fungal Burden', title='Fungal Burden vs Time')
    ax.grid()

    fig.savefig("fungal_burden_vs_time.png")
    plt.show()
