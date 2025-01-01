# dsmts.py
# Utilities for running the DSMTS against smfsb

import smfsb
import numpy as np
import scipy as sp
import pandas as pd


def test_model(N, file_stem):
    model_file = file_stem + ".mod"
    mean_file = file_stem + "-mean.csv"
    sd_file = file_stem + "-sd.csv"
    mean_df = pd.read_csv(mean_file)
    sd_df = pd.read_csv(sd_file)
    spn = smfsb.mod_to_spn(model_file)
    u = len(spn.n)
    sx = np.zeros((51, u))
    sxx = np.zeros((51, u))
    step = spn.step_gillespie() # testing the exact simulator
    for i in range(N):
        out = smfsb.sim_time_series(spn.m, 0, 50, 1, step)
        sx = sx + out
        sxx = sxx + (out * out)
    sample_mean = sx/N
    #z_scores =
    return sample_mean




# Run a demo test if run as a script

if __name__ == '__main__':
    print("A demo test run. Use pytest to run the full suite properly.")
    N = 10
    print(test_model(N, "stochastic/00001/dsmts-001-01"))
    print(test_model(N, "stochastic/00020/dsmts-002-01"))
    print(test_model(N, "stochastic/00030/dsmts-003-01"))
    print("Done.")


# eof

