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
        # define additional values to be plotted here

        state.plotpoint.step_num += 1

        return state

    def finalize(self, state: State):
        num_steps = state.plotpoint.step_num
        time_steps = state.plotpoint.time_steps[:num_steps]
        fungal_burdens = state.plotpoint.fungal_burdens[:num_steps]
        macrophage_counts = state.plotpoint.macrophage_counts[:num_steps]
        neutrophil_counts = state.plotpoint.neutrophil_counts[:num_steps]
        # define additional y_val here

        plt.figure()
        plt.xlabel('Time')
        plt.grid(True)

        for y_val, y_label, title, filename in [
            (fungal_burdens, 'Fungal Burden', 'Fungal Burden vs Time', 'fungal_burden_vs_time'),
            (
                macrophage_counts,
                'Macrophage Count',
                'Macrophage Count vs Time',
                'macrophage_count_vs_time',
            ),
            (
                neutrophil_counts,
                'Neutrophil Count',
                'Neutrophil Count vs Time',
                'neutrophil_count_vs_time',
            ),
            # define additional plots here at 4-tuples
        ]:
            plt.clf()
            plt.plot(time_steps, y_val)
            plt.ylabel(y_label)
            plt.title(title)
            plt.savefig(f'{filename }.png')

        return state
