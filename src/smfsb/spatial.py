# spatial code from chapter 9

import numpy as np

def simTs1D(x0, t0, tt, dt, stepFun, verb=False):
    N = int((tt - t0)//dt + 1)
    u, n = x0.shape
    arr = np.zeros((u, n, N))
    x = x0
    t = t0
    arr[:,:,0] = x
    for i in range(1, N):
        if (verb):
            print(N-i)
        t = t + dt
        x = stepFun(x, t, dt)
        arr[:,:,i] = x
    return(arr)
    

# eof


