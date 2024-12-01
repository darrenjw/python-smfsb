#!/usr/bin/env python3

import smfsb
import numpy as np
import matplotlib.pyplot as plt

seir_sh = """
@model:3.1.1=SEIR "SEIR Epidemic model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:S=100 s
 Pop:E=0 s
 Pop:I=5 s
 Pop:R=0 s
@reactions
@r=Infection
 S + I -> E + I
 beta*S*I : beta=0.1
@r=Transition
 E -> I
 sigma*E : sigma=0.2
@r=Removal
 I -> R
 gamma*I : gamma=0.5
"""

seir = smfsb.shorthand_to_spn(seir_sh)
step_seir = seir.step_gillespie()
out = smfsb.sim_time_series(seir.m, 0, 40, 0.05, step_seir)


fig, axis = plt.subplots()
for i in range(len(seir.m)):
    axis.plot(np.arange(0, 40, 0.05), out[:, i])

axis.legend(seir.n)
fig.savefig("sbmlsh.pdf")

# eof
