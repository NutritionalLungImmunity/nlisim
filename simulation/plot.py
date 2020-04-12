import matplotlib.pyplot as plt
import numpy as np


def plot_cells_num(timestep: np.ndarray, cells_num: np.ndarray, cell_name: str, path: str):
    plt.figure()
    plt.plot(timestep, cells_num)
    plt.xlabel('time')
    plt.ylabel(f'{cell_name} number')
    plt.savefig(path)


def plot_cost(module_name: str, cost: np.ndarray, path: str):
    plt.figure()
    timestep = np.arange(len(cost))
    plt.plot(timestep, cost)
    plt.xlabel('timestep')
    plt.ylabel('second')
    plt.title(f'{module_name} cost')
    plt.savefig(path)
