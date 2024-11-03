#!/usr/bin/env python3
# rfmc.py

import smfsb
import numpy as np
import matplotlib.pyplot as plt

P = np.array([[0.9, 0.1], [0.2, 0.8]])
pi0 = np.array([0.5, 0.5])
out = smfsb.rfmc(200, P, pi0)

fig, axis = plt.subplots()
axis.step(range(200), out)

fig.savefig("rfmc.pdf")


# eof
