#!/usr/bin/env python3

# Birth-death model

import smfsb
import smfsb.models
import numpy as np
import matplotlib.pyplot as plt

bd = smfsb.models.bd()
print(bd)
step = bd.step_gillespie()
out = smfsb.sim_time_series(np.random.default_rng(),
                            bd.m, 0, 100, 0.1, step)

fig, axis = plt.subplots()
axis.plot(range(out.shape[0]), out[:, 0])

axis.legend(bd.n)
fig.savefig("s-m-bd.pdf")

# eof
