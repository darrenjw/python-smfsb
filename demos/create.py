#!/usr/bin/env python3

import smfsb
import numpy as np
import matplotlib.pyplot as plt

sir = smfsb.Spn(
    ["S", "I", "R"],
    ["S->I", "I->R"],
    [[1, 1, 0], [0, 1, 0]],
    [[0, 2, 0], [0, 0, 1]],
    lambda x, t: np.array([0.3 * x[0] * x[1] / 200, 0.1 * x[1]]),
    [197, 3, 0],
)
step_sir = sir.step_poisson()
sample = smfsb.sim_sample(100, sir.m, 0, 20, step_sir)
fig, axis = plt.subplots()
axis.hist(sample[:, 1], 30)
axis.set_title("Infected at time 20")
plt.savefig("create.pdf")

# eof
