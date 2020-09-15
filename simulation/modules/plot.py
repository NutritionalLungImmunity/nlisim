from pathlib import Path
import shutil
from typing import List

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
    defaults = {'output_dir': 'output/plots'}
    StateClass = PlotState

    @property
    def _output_dir(self) -> Path:
        return Path(self.config.get('output_dir'))

    def _update_arrays(self, state: State) -> State:
        plot: PlotState = state.plot
        step_num = plot.step_num

        plot.time_steps[step_num] = state.time
        plot.fungal_burdens[step_num] = len(state.fungus.cells.alive())
        plot.macrophage_counts[step_num] = len(state.macrophage.cells.alive())
        plot.neutrophil_counts[step_num] = len(state.neutrophil.cells.alive())

        plot.step_num += 1

        return state

    def initialize(self, state: State) -> State:
        output_dir = self._output_dir
        if output_dir.exists():
            print(f'File output directory {output_dir.resolve()} exists. Clearing it.')
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        return self._update_arrays(state)

    def advance(self, state: State, previous_time: float) -> State:
        return self._update_arrays(state)

    def _create_plot(
        self,
        fig_num: int,
        time_steps: np.ndarray,
        counts: List[np.ndarray],
        markers: List[str],
        cells: List[str],
        ylabel: str,
    ) -> None:
        plt.figure(fig_num)
        for (count, marker, cell) in zip(counts, markers, cells):
            plt.plot(time_steps, count, marker, label=cell)
        plt.xlabel('Time Unit (Typically Hours)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs. Time')
        plt.grid(True)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)

    def finalize(self, state: State) -> State:
        plot: PlotState = state.plot
        num_steps = plot.step_num
        time_steps = plot.time_steps[:num_steps]

        fungal_burdens = plot.fungal_burdens[:num_steps]
        self._create_plot(
            1, time_steps, [fungal_burdens], ['b.-'], ['Fungal Burden'], 'Fungal Burden'
        )
        plt.savefig(f'{self._output_dir}/fungal_burden_vs_time.png')

        macrophage_counts = plot.macrophage_counts[:num_steps]
        self._create_plot(
            2, time_steps, [macrophage_counts], ['b.-'], ['Macrophage Count'], 'Macrophage Count'
        )
        plt.savefig(f'{self._output_dir}/macrophage_count_vs_time.png')

        neutrophil_counts = plot.neutrophil_counts[:num_steps]
        self._create_plot(
            3, time_steps, [neutrophil_counts], ['b.-'], ['Neutrophil Count'], 'Neutrophil Count'
        )
        plt.savefig(f'{self._output_dir}/neutrophil_count_vs_time.png')

        self._create_plot(
            4,
            time_steps,
            [fungal_burdens, macrophage_counts, neutrophil_counts],
            ['b.-', 'g*-', 'r+-'],
            ['Fungal Burden', 'Macrophage Count', 'Neutrophil Count'],
            'Count',
        )
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0, fontsize='small')
        plt.savefig(f'{self._output_dir}/all_vs_time.png')

        return state
