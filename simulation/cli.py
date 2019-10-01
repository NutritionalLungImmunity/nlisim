from math import ceil

import click

from simulation.config import SimulationConfig
from simulation.initialization import create_state
from simulation.solver import advance


@click.command()
@click.argument('target_time', type=click.FLOAT)
@click.option('--config', type=click.Path(exists=True))
def run(target_time, config):
    config = SimulationConfig(config)
    total = ceil(target_time / config.getfloat('simulation', 'time_step'))

    def get_time(x):
        if x is None:
            return '0'
        return '%.2f s' % x.time

    with click.progressbar(advance(create_state(config), target_time),
                           label='Running simulation',
                           length=total,
                           item_show_func=get_time) as bar:
        for s in bar:
            state = s

    state.save('output.npy')


if __name__ == '__main__':
    run()
