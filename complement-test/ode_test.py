import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from ode_solver import make_ode_solver


def model_maker():
    # noinspection PyUnusedLocal
    @njit
    def f(t, y):
        return -y

    # noinspection PyUnusedLocal
    @njit
    def jac_f(t, y):
        return np.array([-1.0], dtype=np.float64)

    return f, jac_f


solver = make_ode_solver(*model_maker())

t_span = (0.0, 14.0)

result = solver(np.array([1.0], dtype=np.float64), t_span, 0.01)

plt.plot(result[0], result[1])
plt.tight_layout()
plt.show()

result = solver(np.random.normal(loc=1.0, scale=0.1, size=(2, 3, 1)), t_span, 0.01)

for i in range(2):
    for j in range(3):
        plt.plot(result[0], result[1][:, i, j])
plt.tight_layout()
plt.show()
