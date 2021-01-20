import math


def activation_function(*, x, kd, h, volume, b=1):
    x = x / volume  # CONVERT MOL TO MOLAR
    return h * (1 - b * math.exp(-(x / kd)))
