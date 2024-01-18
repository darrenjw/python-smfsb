#!/usr/bin/env python3
# imdeath.py

import smfsb
import numpy as np
import matplotlib.pyplot as plt

times, states = smfsb.imdeath(150)

fig, axis = plt.subplots()
axis.step(times, states[1:], where="post")

fig.savefig("imdeath.pdf")


# eof
