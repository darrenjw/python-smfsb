#!/usr/bin/env python3
# sbml-params.py

# This demo is to illustrate how to modify the parameters of a SPN
# model that has been parsed from SBML (or SBML-shorthand).
# This is useful for simulation studies, paramter scans, and
# inference algorithms.

# Global parameters are stored in a dictionary called `gp`
# Local parameters are stored in a list of dictionaries called `lp`
# Parameters can be updated by modifing the values in the dictionaries

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
out = smfsb.sim_time_series(seir.m, 0, 50, 0.05, step_seir)

fig, axis = plt.subplots()
for i in range(len(seir.m)):
    axis.plot(np.arange(0, 50, 0.05), out[:, i])

axis.legend(seir.n)
fig.savefig("sbml-params-0.pdf")

# We will update the removal rate, gamma.
# This is associated with the third reaction, which is therefore in
#  position 2 in the list of dictionaries.

seir.lp[2]["gamma"] = 0.1

# Now re-run the simulation with the updated parameters:

out = smfsb.sim_time_series(seir.m, 0, 50, 0.05, step_seir)

fig, axis = plt.subplots()
for i in range(len(seir.m)):
    axis.plot(np.arange(0, 50, 0.05), out[:, i])

axis.legend(seir.n)
fig.savefig("sbml-params-1.pdf")


# eof
