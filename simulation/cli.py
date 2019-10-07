from math import ceil

import click

from simulation.config import SimulationConfig
from simulation.contrib.plot import display
from simulation.initialization import create_state
from simulation.solver import advance
from simulation.state import State


@click.group()
def main():
    pass


@main.command()
@click.argument('target_time', type=click.FLOAT, default=100)
@click.option('--config', type=click.Path(exists=True), default='config.ini',
              help='Path to a simulation config', show_default=True)
def run(target_time, config, output_dir, output_interval):
    """Run a simulation"""
    config = SimulationConfig(config)
    total = ceil(target_time / config.getfloat('simulation', 'time_step'))

    def get_time(x):
        if x is None:
            return '0'
        return '%.2f' % x.time

    with click.progressbar(advance(create_state(config), target_time),
                           label='Running simulation',
                           length=total,
                           item_show_func=get_time) as bar:
        for state in bar:
            pass

    state.save('simulation-final.pkl')


@main.command()
@click.argument('file', type=click.File('rb'))
def show(file):
    """Display a simulation output file."""
    display(State.load(file))


if __name__ == '__main__':
    run()
