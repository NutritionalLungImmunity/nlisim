from pathlib import Path

import click
import click_pathlib
from tqdm import tqdm

from simulation.config import SimulationConfig
from simulation.postprocess import process_output
from simulation.solver import run_iterator


FilePath = click_pathlib.Path(exists=True, file_okay=True, dir_okay=False, readable=True)


@click.group()
@click.option(
    '--config',
    'config_files',
    type=FilePath,
    multiple=True,
    default=['config.ini.example'],
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

    with tqdm(desc='Running simulation', total=target_time,) as pbar:
        for state in run_iterator(config, target_time):
            pbar.update(state.time - pbar.n)


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
