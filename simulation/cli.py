from math import ceil
from pathlib import Path
from time import time

import click
import numpy as np
import pylab

from simulation.config import SimulationConfig
from simulation.initialization import create_state
from simulation.solver import advance
from simulation.state import State


def display(state: State, block=True):
    C = state.concentration
    # C = np.ma.masked_where(np.abs(state.concentration) < 1e-6, state.concentration)
    x = np.arange(C.shape[1] + 1) * state.dx
    y = np.arange(C.shape[0] + 1) * state.dy
    pylab.clf()
    pylab.pcolormesh(x, y, C, cmap='hot')
    pylab.colorbar()
    pylab.axis('scaled')
    pylab.title('%.2f' % state.time)
    pylab.draw()
    pylab.pause(0.001)
    if block:
        pylab.show()


@click.group()
def main():
    pass


@main.command()
@click.argument('target_time', type=click.FLOAT, default=100)
@click.option('--config', type=click.Path(exists=True), default='config.ini',
              help='Path to a simulation config', show_default=True)
@click.option('--output-dir', type=click.Path(file_okay=False), default='output',
              help='Directory to store output files', show_default=True)
@click.option('--output-interval', type=click.FLOAT,
              help='Simulation time interval for output files', default=-1)
@click.option('--draw/--no-draw', help='Plot live simulation state')
@click.option('--draw-interval', type=click.INT,
              help='Plot refresh rate in milliseconds', default=100)
def run(target_time, config, output_dir, output_interval, draw, draw_interval):
    """Run a simulation"""
    config = SimulationConfig(config)
    total = ceil(target_time / config.getfloat('simulation', 'time_step'))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def get_time(x):
        if x is None:
            return '0'
        return '%.2f' % x.time

    def save_state(x):
        if x is None:
            return
        x.save(output_dir / ('simulation-%010.3f.pkl' % x.time))

    with click.progressbar(advance(create_state(config), target_time),
                           label='Running simulation',
                           length=total,
                           item_show_func=get_time) as bar:
        last_output = 0
        last_draw = 0
        for state in bar:
            if output_interval >= 0 and last_output + output_interval >= state.time:
                save_state(state)
            now = time() * 1000
            if state and draw and now - last_draw > draw_interval:
                display(state, block=False)
                last_draw = time() * 1000

    state.save(output_dir / f'simulation-final.pkl')
    if draw:
        display(state)


@main.command()
@click.argument('file', type=click.File('rb'))
def show(file):
    """Display a simulation output file."""
    display(State.load(file))


if __name__ == '__main__':
    run()
