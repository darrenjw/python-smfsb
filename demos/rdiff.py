#!/usr/bin/env python3
# rfmc.py

import smfsb
import numpy as np
import matplotlib.pyplot as plt

out = smfsb.rdiff(lambda x: 1 - 0.1 * x, lambda x: np.sqrt(1 + 0.1 * x))

fig, axis = plt.subplots()
axis.step(np.arange(0, 50, 0.01), out)

fig.savefig("rdiff.pdf")


# eof
