import matplotlib.pyplot as plt
import numpy as np


def plot_cost(module_name: str, cost: np.ndarray, path: str):
    plt.figure()
    timestep = np.arange(len(cost))
    plt.plot(timestep, cost)
    plt.xlabel('timestep')
    plt.ylabel('second')
    plt.title(f'{module_name} cost')
    plt.savefig(path)
