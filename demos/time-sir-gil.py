#!/usr/bin/env python3

import smfsb
import numpy as np
import scipy as sp
import time

sir_sh = """
@model:3.1.1=SEIR "SEIR Epidemic model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:S=197 s
 Pop:I=3 s
 Pop:R=0 s
@reactions
@r=Infection
 S + I -> 2I
 beta*S*I : beta=0.0015
@r=Removal
 I -> R
 gamma*I : gamma=0.1
"""

sir = smfsb.shorthand_to_spn(sir_sh)
step_sir = sir.step_gillespie()
out = smfsb.sim_time_series(sir.m, 0, 40, 0.05, step_sir)
print("Starting timed run now")
# start timer
start_time = time.time()
out = smfsb.sim_sample(10000, sir.m, 0, 20, step_sir)
# end timer
end_time = time.time()
elapsed = end_time - start_time
print(f"\n\nElapsed time: {elapsed} seconds\n\n")
print(sp.stats.describe(out))

# Compare with built-in version
sir = smfsb.models.sir()
step_sir = sir.step_gillespie()
out = smfsb.sim_time_series(sir.m, 0, 40, 0.05, step_sir)
print("Starting timed run now")
# start timer
start_time = time.time()
out = smfsb.sim_sample(10000, sir.m, 0, 20, step_sir)
# end timer
end_time = time.time()
elapsed = end_time - start_time
print(f"\n\nElapsed time: {elapsed} seconds\n\n")
print(sp.stats.describe(out))


# eof
