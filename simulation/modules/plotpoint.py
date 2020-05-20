import attr
import matplotlib.pyplot as plt
import numpy as np

from simulation.module import Module, ModuleState
from simulation.state import State


@attr.s(kw_only=True)
class PlotPointState(ModuleState):
    step_num: int = attr.ib(default=0)
    fungal_burdens: np.ndarray = attr.ib(factory=lambda: np.zeros(1000, dtype=int))
    macrophage_counts: np.ndarray = attr.ib(factory=lambda: np.zeros(1000, dtype=int))
    neutrophil_counts: np.ndarray = attr.ib(factory=lambda: np.zeros(1000, dtype=int))
    time_steps: np.ndarray = attr.ib(factory=lambda: np.zeros(1000, dtype=float))


class PlotPoint(Module):
    name = 'plotpoint'
    defaults = {
        'interval': '1',
    }
    StateClass = PlotPointState

    def advance(self, state: State, previous_time: float):
        step_num = state.plotpoint.step_num

        state.plotpoint.time_steps[step_num] = state.time
        state.plotpoint.fungal_burdens[step_num] = len(state.fungus.cells.alive())
        state.plotpoint.macrophage_counts[step_num] = len(state.macrophage.cells.alive())
        state.plotpoint.neutrophil_counts[step_num] = len(state.neutrophil.cells.alive())
        state.plotpoint.step_num += 1

        return state

    def finalize(self, state: State):
        num_steps = state.plotpoint.step_num
        time_steps = state.plotpoint.time_steps[:num_steps]
        fungal_burdens = state.plotpoint.fungal_burdens[:num_steps]
        macrophage_counts = state.plotpoint.macrophage_counts[:num_steps]
        neutrophil_counts = state.plotpoint.neutrophil_counts[:num_steps]

        plt.figure()
        plt.xlabel('Time')
        plt.grid(True)

        plt.clf()
        plt.plot(time_steps, fungal_burdens)
        plt.ylabel('Fungal Burden')
        plt.title('Fungal Burden vs Time')
        plt.savefig('fungal_burden_vs_time.png')

        plt.clf()
        plt.plot(time_steps, macrophage_counts)
        plt.ylabel('Macrophage Count')
        plt.title('Macrophage Count vs Time')
        plt.savefig('macrophage_count_vs_time.png')

        plt.clf()
        plt.plot(time_steps, neutrophil_counts)
        plt.ylabel('Neutrophil Count')
        plt.title('Neutrophil Count vs Time')
        plt.savefig('neutrophil_count_vs_time.png')

        return state
