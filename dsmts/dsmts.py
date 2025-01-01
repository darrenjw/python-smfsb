# dsmts.py
# Utilities for running the DSMTS against smfsb

import smfsb
import numpy as np
import pandas as pd


def test_model(n, file_stem):
    model_file = file_stem + ".mod"
    mean_file = file_stem + "-mean.csv"
    sd_file = file_stem + "-sd.csv"
    mean = pd.read_csv(mean_file).to_numpy()[:, 1:]
    sd = pd.read_csv(sd_file).to_numpy()[:, 1:]
    spn = smfsb.mod_to_spn(model_file)
    u = len(spn.n)
    sx = np.zeros((51, u))
    sxx = np.zeros((51, u))
    step = spn.step_gillespie()  # testing the exact simulator
    for i in range(n):
        out = smfsb.sim_time_series(spn.m, 0, 50, 1, step)
        sx = sx + out
        si = out - mean
        sxx = sxx + (si * si)
    sample_mean = sx / n
    z_scores = np.sqrt(n) * (sample_mean - mean) / sd
    sts = sxx / n
    y_scores = (sts / (sd * sd) - 1) * np.sqrt(n / 2)
    fails = np.array([np.sum(abs(z_scores) > 3), np.sum(abs(y_scores) > 5)])
    if np.sum(fails) > 0:
        print(str(fails) + " for " + file_stem)
    return fails


# Run a demo test if run as a script

if __name__ == "__main__":
    print("A demo test run. Use pytest to run the full suite properly.")
    N = 1000
    print(test_model(N, "stochastic/00001/dsmts-001-01"))
    print(test_model(N, "stochastic/00020/dsmts-002-01"))
    print(test_model(N, "stochastic/00030/dsmts-003-01"))
    print("Done.")


# eof
