#!/usr/bin/env python3

# Imigration-death model

import smfsb
import smfsb.models
import numpy as np
import matplotlib.pyplot as plt

id = smfsb.models.id()
print(id)
step = id.step_gillespie()
out = smfsb.sim_time_series(np.random.default_rng(), id.m, 0, 100, 0.1, step)


fig, axis = plt.subplots()
axis.plot(range(out.shape[0]), out[:, 0])

axis.legend(id.n)
fig.savefig("s-m-id.pdf")

# eof
