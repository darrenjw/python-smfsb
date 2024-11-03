#!/usr/bin/env python3
# simple_euler.py

import smfsb
import numpy as np
import matplotlib.pyplot as plt


def lv(th=[1, 0.1, 0.1]):
    def rhs(x, t):
        return np.array(
            [th[0] * x[0] - th[1] * x[0] * x[1], th[1] * x[0] * x[1] - th[2] * x[1]]
        )

    return rhs


out = smfsb.simple_euler(lv(), np.array([4, 10]), 100)

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(np.arange(0, 100, 0.001), out[:, i])
axis.legend(["Prey", "Predator"])
fig.savefig("simple_euler.pdf")


# eof
