from pathlib import Path
import shutil
from typing import Tuple

import click
import click_pathlib
from tqdm import tqdm

from nlisim.config import SimulationConfig

InputFilePath = click_pathlib.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
OutputDirPath = click_pathlib.Path(file_okay=False, dir_okay=True, writable=True)


@click.group()
@click.option(
    '--config',
    'config_files',
    type=InputFilePath,
    multiple=True,
    default=['config.ini'],
    help='Path to a simulation config. May be specified multiple times to cascade configurations.',
    show_default=True,
)
@click.pass_context
def main(ctx: click.Context, config_files: Tuple[Path]) -> None:
    ctx.obj = {'config': SimulationConfig(*config_files)}


@main.command()
@click.argument('target_time', type=click.FLOAT, default=20)
@click.pass_obj
def run(obj, target_time: float) -> None:
    """Run a simulation."""
    # Don't import the solver module unless it's needed for this command
    from nlisim.solver import run_iterator

    config = obj['config']

    with tqdm(
        desc='Running simulation',
        unit='hour',
        total=target_time,
    ) as pbar:
        for state, _ in run_iterator(config, target_time):
            pbar.update(state.time - pbar.n)


@main.command('postprocess', help='Postprocess simulation output files')
@click.option(
    '--output',
    'postprocess_dir',
    type=OutputDirPath,
    default='postprocessed',
    help='Path to dump postprocessed data files',
    show_default=True,
)
@click.pass_obj
def postprocess(obj, postprocess_dir: Path) -> None:
    # Don't import the postprocess module unless it's needed for this command
    from nlisim.postprocess import process_output

    if postprocess_dir.exists():
        click.echo(f'Postprocess output directory {postprocess_dir.resolve()} exists. Clearing it.')
        shutil.rmtree(postprocess_dir)
    postprocess_dir.mkdir(parents=True)

    state_files = Path(obj['config']['state_output'].get('output_dir')).glob('simulation-*.hdf5')

    process_output(state_files, postprocess_dir)


if __name__ == '__main__':
    main()
