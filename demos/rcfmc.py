#!/usr/bin/env python3
# rcfmc.py

import smfsb
import numpy as np
import matplotlib.pyplot as plt

Q = np.array([[-0.5, 0.5], [1, -1]])
pi0 = np.array([0.5, 0.5])
times, states = smfsb.rcfmc(30, Q, pi0)

fig, axis = plt.subplots()
axis.step(times, states[1:], where="post")

fig.savefig("rcfmc.pdf")


# eof
