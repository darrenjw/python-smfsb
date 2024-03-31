# metrop.py

import smfsb
import numpy as np

vec = smfsb.metrop(10000, 1)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
axis.hist(vec, bins=30)
fig.savefig("metrop.pdf")




# eof

