import click

from simulation.driver import run


@click.command()
@click.argument('time', type=click.FLOAT)
def main(time):
    run(time)


if __name__ == '__main__':
    main()
