# metrop.py

import smfsb
import numpy as np
import matplotlib.pyplot as plt

vec = smfsb.metrop(np.random.default_rng(), 10000, 1)

fig, axis = plt.subplots()
axis.hist(vec, bins=30)
fig.savefig("metrop.pdf")


# eof
