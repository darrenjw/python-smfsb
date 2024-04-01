# data.py
# some example synthetic data sets for testing inference algorithms

import numpy as np


# time, prey, predator
LVperfect = np.array([
    [   0,  50, 100],
    [   2, 145,  93],
    [   4, 265, 248],
    [   6,  64, 341],
    [   8,  35, 166],
    [  10,  52,  79],
    [  12, 201,  54],
    [  14, 305, 331],
    [  16,  26, 364],
    [  18,  19, 129],
    [  20,  90,  50],
    [  22, 334, 137],
    [  24,  61, 508],
    [  26,  15, 194],
    [  28,  24,  65],
    [  30, 145,  40]
    ])


# time, prey, predator
LVnoise10 = np.array([
    [   0,  34.19903,  98.11945],
    [   2, 156.54757,  86.52563],
    [   4, 267.77267, 260.94433],
    [   6,  86.40285, 345.20318],
    [   8,  46.47921, 146.85739],
    [  10,  55.24121,  68.51684],
    [  12, 198.35381,  53.08404],
    [  14, 305.98165, 337.47268],
    [  16,  31.67898, 359.75207],
    [  18,  29.13059, 116.88260],
    [  20,  89.27934,  35.02892],
    [  22, 313.28117, 129.03995],
    [  24,  86.99446, 503.42103],
    [  26,  28.49763, 191.07711],
    [  28,  36.19940,  64.54570],
    [  30, 136.51468,  40.89381]
    ])


# time, prey
LVpreyNoise10 = LVnoise10[:,range(2)]


# eof

