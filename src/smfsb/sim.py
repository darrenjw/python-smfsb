#!/usr/bin/env python3
# sim.py

import numpy as np



    
# Some simulation functions


def simTs(x0, t0, tt, dt, stepFun):
    """Simulate a model on a regular grid of times, using a function (closure)
    for advancing the state of the model

    This function simulates single realisation of a model on a regular
    grid of times using a function (closure) for advancing the state
    of the model, such as created by ‘stepGillespie’ or
    ‘stepEuler’.

    Parameters
    ----------
    x0: array of numbers
        The intial state of the system at time t0
    t0: float
        This intial time to be associated with the intial state.
    tt: float
        The terminal time of the simulation.
    dt: float
        The time step of the output. Note that this time step relates only to
        the recorded output, and has no bearing on the accuracy of the simulation
        process.
    stepFun: function
        A function (closure) for advancing the state of the process,
        such as produced by ‘stepGillespie’ or ‘stepEuler’.

    Returns
    -------
    A matrix with rows representing the state of the system at successive times.

    Examples
    --------
    >>> import smfsb.models
    >>> lv = smfsb.models.lv()
    >>> stepLv = lv.stepGillespie()
    >>> smfsb.simTs([50, 100], 0, 100, 0.1, stepLv)
    """
    n = int((tt-t0) // dt) + 1
    u = len(x0)
    mat = np.zeros((n, u))
    x = x0
    t = t0
    mat[0,:] = x
    for i in range(1, n):
        t = t + dt
        x = stepFun(x, t, dt)
        mat[i,:] = x
    return mat




def simSample(n, x0, t0, deltat, stepFun):
    """Simulate a many realisations of a model at a given fixed time in the
    future given an initial time and state, using a function (closure) for
    advancing the state of the model

    This function simulates many realisations of a model at a given
    fixed time in the future given an initial time and state, using a
    function (closure) for advancing the state of the model , such as
    created by ‘stepGillespie’ or ‘stepEuler’.

    Parameters
    ----------
    n: int
        The number of samples required.
    x0: array of numbers
        The intial state of the system at time t0.
    t0: float
        The intial time to be associated with the initial state.
    deltat: float
        The amount of time in the future of t0 at which samples of the
        system state are required.
    stepFun: function
        A function (closure) for advancing the state of the process,
        such as produced by `stepGillespie' or `stepEuler'.

    Returns
    -------
    A matrix with rows representing simulated states at time t0+deltat.

    Examples
    --------
    >>> import smfsb.models
    >>> lv = smfsb.models.lv()
    >>> stepLv = lv.stepGillespie()
    >>> smfsb.simSample(10, [50, 100], 0, 30, stepLv)
    """
    u = len(x0)
    mat = np.zeros((n, u))
    for i in range(n):
        mat[i,:] = stepFun(x0, t0, deltat)
    return mat




# Illustrative functions from early in the book


def rfmc(n, P, pi0):
    """Simulate a finite state space Markov chain

    This function simulates a single realisation from a discrete time
    Markov chain having a finite state space based on a given
    transition matrix.

    Parameters
    ----------
    n: int
        The number of states to be sampled from the Markov chain,
        including the initial state, which will be sampled using
        ‘pi0’.
    P: matrix
        The transition matrix of the Markov chain. This is assumed to
        be a stochastic matrix, having non-negative elements and rows
        summing to one, though in fact, the rows will in any case be
        normalised by the sampling procedure.
    pi0: array
        A vector representing the probability distribution of the
        initial state of the Markov chain. If this vector is of
        length ‘r’, then the transition matrix ‘P’ is assumed to be
        ‘r x r’. The elements of this vector are assumed to be
        non-negative and sum to one, though in fact, they will be
        normalised by the sampling procedure.

    Returns
    -------
    A numpy array containing the sampled values from the Markov chain.

    Examples
    --------
    >>> import smfsb
    >>> import numpy as np
    >>> P = np.array([[0.9,0.1],[0.2,0.8]])
    >>> pi0 = np.array([0.5,0.5])
    >>> smfsb.rfmc(200, P, pi0)
    """
    v = np.zeros(n)
    r = len(pi0)
    v[0] = np.random.choice(r, p=pi0)
    for i in range(1,n):
        v[i] = np.random.choice(r, p=P[int(v[i-1]),:])
    return v




# Misc utility functions

import inspect

def showSource(fun):
    """Print to console the source code of a function or method.

    Called for the side-effect of printing the function source to standard
    output.

    Parameters
    ----------
    fun: function or method
        The function of interest

    Returns
    -------
    None
    
    Examples
    --------
    >>> import smfsb
    >>> smfsb.showSource(smfsb.Spn.stepGillespie)
    """
    print(inspect.getsource(fun))


    


# eof

