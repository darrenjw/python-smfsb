# metrop.py

import smfsb
import matplotlib.pyplot as plt

vec = smfsb.metrop(10000, 1)

fig, axis = plt.subplots()
axis.hist(vec, bins=30)
fig.savefig("metrop.pdf")


# eof
