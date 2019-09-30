import click

from simulation.driver import run


@click.command()
def main():
    run(10)


if __name__ == '__main__':
    main()
