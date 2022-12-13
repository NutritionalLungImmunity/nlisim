import constants
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def minimal_model(t, var, k, ks, d, FD, FI):
    C3, C3b, C3bB_c, C3bB_o, C3bBb, FB, FH, C3bH, C3bBbH = var

    dC3 = ks[1] - d[1] * C3 - k[1] * C3 - (k[2] * C3bBb * C3) / (k[3] + C3)
    dC3b = (
        k[1] * C3
        + (k[2] * C3bBb * C3) / (k[3] + C3)
        - k[4] * C3b * FB
        + k[5] * C3bB_c
        + k[6] * C3bBb
        - k[15] * C3b * FH
        + k[16] * C3bH
        + k[21] * C3bBbH
    )
    dC3bB_c = k[4] * C3b * FB - k[5] * C3bB_c - k[9] * C3bB_c + k[10] * C3bB_o
    dC3bB_o = k[9] * C3bB_c - k[10] * C3bB_o - (k[7] * FD * C3bB_o) / (k[8] + C3bB_o)
    dC3bBb = (
        (k[7] * FD * C3bB_o) / (k[8] + C3bB_o) - k[6] * C3bBb - k[25] * C3bBb * FH + k[16] * C3bBbH
    )
    dFB = ks[2] - d[2] * FB - k[4] * C3b * FB + k[5] * C3bB_c
    dFH = (
        ks[3]
        - d[3] * FH
        - k[15] * C3b * FH
        + k[16] * C3bH
        - k[25] * C3bBb * FH
        + k[16] * C3bBbH
        + (k[19] * C3bH * FI) / (k[20] + C3bH)
        + k[21] * C3bBbH
    )
    dC3bH = k[15] * C3b * FH - k[16] * C3bH - (k[19] * C3bH * FI) / (k[20] + C3bH)
    dC3bBbH = k[25] * C3bBb * FH - k[16] * C3bBbH - k[21] * C3bBbH

    return [dC3, dC3b, dC3bB_c, dC3bB_o, dC3bBb, dFB, dFH, dC3bH, dC3bBbH]


const = [constants.k, constants.ks, constants.d, constants.FD, constants.FI]

# Initial state of system
var0 = np.zeros(9)

# C3 initial value
var0[0] = 6

# FB initial value
var0[5] = 2

# FH initial value
var0[6] = 3

t_span = (0.0, 100.0)

result = solve_ivp(minimal_model, t_span, var0, args=const, method='BDF', max_step=1.0)

var_names = ["C3", "C3b", "C3bB_c", "C3bB_o", "C3bBb", "FB", "FH", "C3bH", "C3bBbH"]

fig, axs = plt.subplots(3, 3)
for idx in range(9):
    row = idx % 3
    col = idx // 3
    axs[row, col].plot(result.t, result.y[idx, :])
    axs[row, col].set_title(var_names[idx])

plt.tight_layout()
plt.show()
