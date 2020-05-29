import os

import attr
import matplotlib.pyplot as plt
import numpy as np

from simulation.module import Module, ModuleState
from simulation.state import State

MAX_ARRAY_LENGTH = 1000


@attr.s(kw_only=True)
class PlotState(ModuleState):
    step_num: int = attr.ib(default=0)
    time_steps: np.ndarray = attr.ib(factory=lambda: np.zeros(MAX_ARRAY_LENGTH, dtype=float))

    # arrays to plot against time_steps
    fungal_burdens: np.ndarray = attr.ib(factory=lambda: np.zeros(MAX_ARRAY_LENGTH, dtype=int))
    macrophage_counts: np.ndarray = attr.ib(factory=lambda: np.zeros(MAX_ARRAY_LENGTH, dtype=int))
    neutrophil_counts: np.ndarray = attr.ib(factory=lambda: np.zeros(MAX_ARRAY_LENGTH, dtype=int))


class Plot(Module):
    name = 'plot'
    defaults = {'plot_output_folder': 'output/plots/'}
    StateClass = PlotState

    def advance(self, state: State, previous_time: float):
        plot: PlotState = state.plot
        step_num = plot.step_num

        plot.time_steps[step_num] = state.time
        plot.fungal_burdens[step_num] = len(state.fungus.cells.alive())
        plot.macrophage_counts[step_num] = len(state.macrophage.cells.alive())
        plot.neutrophil_counts[step_num] = len(state.neutrophil.cells.alive())

        plot.step_num += 1

        return state

    def setup_plot(self, ylabel: str, title: str, filename: str):
        plt.xlabel('Time Unit (Typically Hours)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.savefig(f'{filename}.png')
        plt.clf()

    def finalize(self, state: State):
        plot_output_folder = self.config.get('plot_output_folder')
        if not os.path.exists(plot_output_folder):
            os.mkdir(plot_output_folder)
        plot: PlotState = state.plot
        num_steps = plot.step_num
        time_steps = plot.time_steps[:num_steps]
        plt.figure()

        fungal_burdens = plot.fungal_burdens[:num_steps]
        plt.plot(time_steps, fungal_burdens)
        self.setup_plot(
            'Fungal Burden', 'Fungal Burden vs Time', plot_output_folder + 'fungal_burden_vs_time'
        )

        macrophage_counts = plot.macrophage_counts[:num_steps]
        plt.plot(time_steps, macrophage_counts)
        self.setup_plot(
            'Macrophage Count',
            'Macrophage Count vs Time',
            plot_output_folder + 'vmacrophage_count_vs_time',
        )

        neutrophil_counts = plot.neutrophil_counts[:num_steps]
        plt.plot(time_steps, neutrophil_counts)
        self.setup_plot(
            'Neutrophil Count',
            'Neutrophil Count vs Time',
            plot_output_folder + 'neutrophil_count_vs_time',
        )

        return state
