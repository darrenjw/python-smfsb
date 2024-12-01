#!/usr/bin/env python3

import smfsb
import smfsb.models
import matplotlib.pyplot as plt

id = smfsb.models.id()
print(id)
step = id.step_gillespie()
out = smfsb.sim_time_series(id.m, 0, 100, 0.1, step)


fig, axis = plt.subplots()
axis.plot(range(out.shape[0]), out[:, 0])

axis.legend(id.n)
fig.savefig("s-m-id.pdf")

# eof
