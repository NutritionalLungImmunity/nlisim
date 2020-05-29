from math import ceil
from pathlib import Path

import attr
import click
import click_pathlib

from simulation.config import SimulationConfig
from simulation.postprocess import process_output
from simulation.solver import advance, finalize, initialize
from simulation.state import State


FilePath = click_pathlib.Path(exists=True, file_okay=True, dir_okay=False, readable=True)


@click.group()
@click.option(
    '--config',
    'config_files',
    type=FilePath,
    multiple=True,
    default=['config.ini'],
    help='Path to a simulation config. May be specificed multiple times to cascade configurations.',
    show_default=True,
)
@click.pass_context
def main(ctx, config_files):
    ctx.obj = {'config': SimulationConfig(*config_files)}


@main.command()
@click.argument('target_time', type=click.FLOAT, default=20)
@click.pass_obj
def run(obj, target_time):
    """Run a simulation."""
    config = obj['config']
    total = ceil(target_time / config.getfloat('simulation', 'time_step'))

    attr.set_run_validators(config.getboolean('simulation', 'validate'))

    def get_time(x):
        if x is None:
            return '0'
        return '%.2f' % x.time

    state = initialize(State.create(config))
    with click.progressbar(
        advance(state, target_time),
        label='Running simulation',
        length=total,
        item_show_func=get_time,
    ) as bar:
        for _state in bar:
            pass

    state = finalize(state)


@main.command('postprocess', help='Postprocess simulation output files')
@click.option(
    '--output',
    type=click_pathlib.Path(file_okay=False, dir_okay=True, writable=True),
    default='postprocessed',
    help='Path to dump postprocessed data files',
    show_default=True,
)
@click.pass_obj
def postprocess(obj, output):
    files = Path(obj['config']['state_output'].get('output_dir')).glob('simulation-*.hdf5')
    for index, file in enumerate(sorted(files)):
        output_dir = Path(output) / ('%03i' % (index + 1))
        process_output(file, output_dir)


if __name__ == '__main__':
    run()
