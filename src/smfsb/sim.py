#!/usr/bin/env python3
# sim.py

import numpy as np
import inspect


# Some simulation functions


def sim_time_series(x0, t0, tt, dt, step_fun):
    """Simulate a model on a regular grid of times, using a function (closure)
    for advancing the state of the model

    This function simulates single realisation of a model on a regular
    grid of times using a function (closure) for advancing the state
    of the model, such as created by ‘step_gillespie’ or
    ‘step_euler’.

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
    step_fun: function
        A function (closure) for advancing the state of the process,
        such as produced by ‘step_gillespie’ or ‘step_euler’.

    Returns
    -------
    A matrix with rows representing the state of the system at successive times.

    Examples
    --------
    >>> import smfsb.models
    >>> lv = smfsb.models.lv()
    >>> stepLv = lv.step_gillespie()
    >>> smfsb.sim_time_series([50, 100], 0, 100, 0.1, stepLv)
    """
    n = int((tt - t0) // dt) + 1
    u = len(x0)
    mat = np.zeros((n, u))
    x = x0
    t = t0
    mat[0, :] = x
    for i in range(1, n):
        t = t + dt
        x = step_fun(x, t, dt)
        mat[i, :] = x
    return mat


def sim_sample(n, x0, t0, deltat, step_fun):
    """Simulate a many realisations of a model at a given fixed time in the
    future given an initial time and state, using a function (closure) for
    advancing the state of the model

    This function simulates many realisations of a model at a given
    fixed time in the future given an initial time and state, using a
    function (closure) for advancing the state of the model , such as
    created by `step_gillespie` or `step_euler`.

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
    step_fun: function
        A function (closure) for advancing the state of the process,
        such as produced by `step_gillespie` or `step_euler`.

    Returns
    -------
    A matrix with rows representing simulated states at time t0+deltat.

    Examples
    --------
    >>> import smfsb.models
    >>> lv = smfsb.models.lv()
    >>> stepLv = lv.step_gillespie()
    >>> smfsb.sim_sample(10, [50, 100], 0, 30, stepLv)
    """
    u = len(x0)
    mat = np.zeros((n, u))
    for i in range(n):
        mat[i, :] = step_fun(x0, t0, deltat)
    return mat


def step_sde(drift, diffusion, dt=0.01):
    """Create a function for advancing the state of an SDE model by using a
    simple Euler-Maruyama integration method

    This function creates a function for advancing the state of an SDE
    model using a simple Euler-Maruyama integration method. The
    resulting function (closure) can be used in conjunction with other
    functions (such as ‘sim_time_series’) for simulating realisations of SDE
    models.

    Parameters
    ----------
    drift: function
        A function representing the drift vector of the SDE model
        (corresponding roughly to the RHS of ante ODE model). ‘drift’
        should have arguments `x` and `t`, with ‘x’ representing
        current system state and ‘t’ representing current system
        time.  The value of the function should be a vector of the
        same dimension as ‘x’, representing the infinitesimal mean of
        the Ito SDE.
    diffusion: function
        A function representing the diffusion matrix of the SDE
        model (the square root of the infinitesimal variance matrix).
        ‘diffusion’ should have arguments `x` and `t`, with
        ‘x’ representing current system state and ‘t’ representing
        current system time. The value of the function should be a
        square matrix with both dimensions the same as the length of
        ‘x’.
    dt: float
        Time step to be used by the simple Euler-Maruyama integration
        method. Defaults to 0.01.

    Returns
    -------
    A function which can be used to advance the state of the SDE
    model with given drift vector and diffusion matrix, by using an
    Euler-Maruyama method with step size ‘dt’. The function closure
    returns a vector representing the simulated state of the system
    at the new time.

    Examples
    --------
    >>> import smfsb
    >>> import numpy as np
    >>> lamb = 2; alpha = 1; mu = 0.1; sig = 0.2
    >>> def myDrift(x, t):
    >>>     return np.array([lamb - x[0]*x[1],
    >>>                      alpha*(mu - x[1])])
    >>>
    >>> def myDiff(x, t):
    >>>     return np.array([[np.sqrt(lamb + x[0]*x[1]), 0],
    >>>                      [0 ,sig*np.sqrt(x[1])]])
    >>>
    >>> stepProc = smfsb.step_sde(myDrift, myDiff, dt=0.001)
    >>> smfsb.sim_time_series(np.array([1, 0.1]), 0, 30, 0.01, stepProc)
    """
    sdt = np.sqrt(dt)

    def step(x0, t0, deltat):
        x = x0
        t = t0
        termt = t0 + deltat
        v = len(x)
        while True:
            dw = np.random.normal(scale=sdt, size=v)
            x = np.add(x, drift(x, t) * dt + diffusion(x, t).dot(dw))
            t = t + dt
            if t > termt:
                return x

    return step


# Illustrative functions from early in the book


def rfmc(n, p_mat, pi0):
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
    p_mat: matrix
        The transition matrix of the Markov chain. This is assumed to
        be a stochastic matrix, having non-negative elements and rows
        summing to one.
    pi0: array
        A vector representing the probability distribution of the
        initial state of the Markov chain. If this vector is of
        length ‘r’, then the transition matrix ‘p_mat’ is assumed to be
        ‘r x r’. The elements of this vector are assumed to be
        non-negative and sum to one.

    Returns
    -------
    A numpy array containing the sampled values from the Markov chain.

    Examples
    --------
    >>> import smfsb
    >>> import numpy as np
    >>> p_mat = np.array([[0.9,0.1],[0.2,0.8]])
    >>> pi0 = np.array([0.5,0.5])
    >>> smfsb.rfmc(200, p_mat, pi0)
    """
    v = np.zeros(n)
    r = len(pi0)
    v[0] = np.random.choice(r, p=pi0)
    for i in range(1, n):
        v[i] = np.random.choice(r, p=p_mat[int(v[i - 1]), :])
    return v


def rcfmc(n, q_mat, pi0):
    """Simulate a continuous time finite state space Markov chain

    This function simulates a single realisation from a continuous
    time Markov chain having a finite state space based on a given
    transition rate matrix.

    Parameters
    ----------
    n: int
        The number of states to be sampled from the Markov chain,
        including the initial state, which will be sampled using
        ‘pi0’.
    q_mat: matrix
        The transition rate matrix of the Markov chain, where each
        off-diagonal element ‘q_mat[i,j]’ represents the rate of
        transition from state ‘i’ to state ‘j’. This matrix is
        assumed to be square, having rows summing to zero.
    pi0: array
        A vector representing the probability distribution of the
        initial state of the Markov chain. If this vector is of
        length ‘r’, then the transition matrix ‘P’ is assumed to be
        ‘r x r’. The elements of this vector are assumed to be
        non-negative and sum to one.

    Returns
    -------
    A tuple, `(tvec, xvec)`, where `tvec` is a vector of event times of
    length `n` and `xvec` is a vector of states, of length `n+1`.

    Examples
    --------
    >>> import smfsb
    >>> import numpy as np
    >>> smfsb.rcfmc(200, np.array([[-0.5,0.5],[1,-1]]), np.array([1,0]))
    """
    xvec = np.zeros(n + 1)
    tvec = np.zeros(n)
    r = len(pi0)
    x = np.random.choice(r, p=pi0)
    t = 0
    xvec[0] = x
    for i in range(n):
        t = t + np.random.exponential(-q_mat[int(x), int(x)])
        weights = q_mat[int(x), :].copy()
        weights[x] = 0
        weights = weights / np.sum(weights)
        x = np.random.choice(r, p=weights)
        xvec[i + 1] = x
        tvec[i] = t
    return tvec, xvec


def imdeath(n=20, x0=0, lamb=1, mu=0.1):
    """Simulate a sample path from the homogeneous immigration-death process

    This function simulates a single realisation from a
    time-homogeneous immigration-death process.

    Parameters
    ----------
    n: int
        The number of states to be sampled from the process, not
        including the initial state, ‘x0’
    x0: int
        The initial state of the process, which defaults to zero.
    lamb: float
        The rate at which new individual immigrate into the
        population. Defaults to 1.
    mu: float
        The rate at which individuals within the population die,
        independently of all other individuals. Defaults to 0.1.

    Returns
    -------
    A tuple, `(tvec, xvec)`, where `tvec` is a vector of event times of
    length `n` and `xvec` is a vector of states, of length `n+1`.

    Examples
    --------
    >>> import smfsb
    >>> smfsb.imdeath(100)
    """
    xvec = np.zeros(n + 1)
    tvec = np.zeros(n)
    t = 0
    x = x0
    xvec[0] = x
    for i in range(n):
        t = t + np.random.exponential(lamb + x * mu)
        if np.random.random() < lamb / (lamb + x * mu):
            x = x + 1
        else:
            x = x - 1
        xvec[i + 1] = x
        tvec[i] = t
    return tvec, xvec


def rdiff(a_fun, b_fun, x0=0, t=50, dt=0.01):
    """Simulate a sample path from a univariate diffusion process

    This function simulates a single realisation from a
    time-homogeneous univariate diffusion process.

    Parameters
    ----------
    a_fun: function
        A scalar-valued function representing the infinitesimal mean
        (drift) of the diffusion process. The argument is the current
        state of the process.
    b_fun: function
        A scalar-valued function representing the infinitesimal
        standard deviation of the process. The argument is the current
        state of the process.
    x0: float
        The initial state of the diffusion process.
    t: float
        The length of the time interval over which the diffusion process
        is to be simulated.
    dt: float
        The step size to be used _both_ for the time step of the Euler
        integration method _and_ the recording interval for the output.
        It would probably be better to have separate parameters for
        these two things. Defaults to 0.01 time units.

    Returns
    -------
    A vector of states of the diffusion process on the required time grid.

    Examples
    --------
    >>> import smfsb
    >>> import numpy as np
    >>> smfsb.rdiff(lambda x: 1 - 0.1*x, lambda x: np.sqrt(1 + 0.1*x))
    """
    n = int(t / dt)
    xvec = np.zeros(n)
    x = x0
    sdt = np.sqrt(dt)
    for i in range(n):
        t = i * dt
        x = x + a_fun(x) * dt + b_fun(x) * np.random.normal(0, sdt)
        xvec[i] = x
    return xvec


def simple_euler(rhs, ic, t=50, dt=0.001):
    """Simulate a sample path from an ODE model

    This function integrates an Ordinary Differential Equation (ODE)
    model using a simple first order Euler method. The function is
    pedagogic and not intended for serious use. See scipy.integrate.solve_ivp
    for better, more robust ODE solvers.

    Parameters
    ----------
    rhs: function
        A vector-valued function representing the right hand side of
        the ODE model.  The first argument is a vector representing
        the current state of the model, ‘x’.  The second argument of
        ‘rhs’ is the current simulation time, ‘t’. In the case of a
        homogeneous ODE model, this argument will be unused within
        the function. The output of ‘rhs’ should be a vector of the
        same dimension as ‘x’.
    ic: array
        The initial conditions for the ODE model. This should be a
        vector of the same dimensions as the output from ‘rhs’, and
        the first argument of ‘rhs’.
    t: float
        The length of the time interval over which the ODE model is
        to be integrated. Defaults to 50 time units.
    dt: float
        The step size to be used both for the time step of the Euler
        integration method and the recording interval for the output.
        It would probably be better to have separate parameters for
        these two things. Defaults to 0.001 time units.

    Returns
    -------
    A matrix with rows representing the states at each time step.

    Examples
    --------
    >>> import smfsb
    >>> import numpy as np
    >>> smfsb.simple_euler(lambda x,t: 1-0.1*x[0], np.array([0]))
    """
    p = len(ic)
    n = int(t / dt)
    x_mat = np.zeros((n, p))
    x = ic
    t = 0
    x_mat[0, :] = x
    for i in range(1, n):
        t = t + dt
        x = x + rhs(x, t) * dt
        x_mat[i, :] = x
    return x_mat


def discretise(times, states, dt=1, start=0):
    """Discretise output from a discrete event simulation algorithm

    This function discretises output from a discrete event simulation
    algorithm such as ‘gillespie’ onto a regular time grid, and
    returns the results as a matrix.

    Parameters
    ----------
    times: array
        A vector of event times.
    states: array
        A matrix of states. There should be one more row than the length of times.
    dt: float
        The time step required for the output of the discretisation
        process. Defaults to one time unit.
    start: float
        The start time for the output. Defaults to zero.

    Returns
    -------
    A matrix with rows corresponding to the state of the system on a regular
    grid.

    Examples
    --------
    >>> import smfsb
    >>> import smfsb.models
    >>> lv = smfsb.models.lv()
    >>> times, states = lv.gillespie(1000)
    >>> smfsb.discretise(times, states, 0.1)
    """
    events = len(times)
    end = times[events - 1]
    length = int((end - start) // dt) + 1
    x = np.zeros((length, states.shape[1]))
    target = 0
    j = 0
    for i in range(events):
        while times[i] >= target:
            x[j, :] = states[i, :]
            j = j + 1
            target = target + dt
    return x


# Misc utility functions


def show_source(fun):
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
    >>> smfsb.show_source(smfsb.Spn.step_gillespie)
    """
    print(inspect.getsource(fun))


# eof
