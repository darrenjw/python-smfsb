# inference.py
# Code relating to Chapters 10 and 11

import numpy as np







# Some illustrative functions not intended for serious use...


def normgibbs(N, n, a, b, c, d, xbar, ssquared):
    """A simple Gibbs sampler for Bayesian inference for the mean and
    precision of a normal random sample

    This function runs a simple Gibbs sampler for the Bayesian
    posterior distribution of the mean and precision given a normal
    random sample.

    Parameters
    ----------
    N : int
      The number of iterations of the Gibbs sampler
    n : int
      The sample size of the normal random sample
    a : float
      The shape parameter of the gamma prior on the sample precision.
    b : float
      The scale parameter of the gamma prior on the sample precision.
    c : float
      Th mean of the normal prior on the sample mean.
    d : float
      The precision of the normal prior on the sample mean.
    xbar : float
      The sample mean of the data.
    ssquared : float
      The sample variance of the data.

    Returns
    -------
    A matrix containing the samples of the Gibbs sampler in rows.

    Examples
    --------
    >>> import smfsb
    >>> postmat = smfsb.normgibbs(N=1100, n=15, a=3, b=11, c=10, d=1/100,
    >>>   xbar=25, ssquared=20)
    >>> postmat = postmat[range(100,1100),:]
    """
    mat = np.zeros((N, 2))
    mu = c
    tau = a/b
    mat[1,:] = [mu, tau]
    for i in range(1, N):
        muprec = n*tau + d
        mumean = (d*c + n*tau*xbar)/muprec
        mu = np.random.normal(mumean, np.sqrt(1/muprec))
        taub = b + 0.5*((n-1)*ssquared + n*(xbar-mu)*(xbar-mu))
        tau = np.random.gamma(a + n/2, 1/taub)
        mat[i,:] = [mu, tau]
    return mat

from scipy.stats import norm

def metrop(n, alpha):
    """Run a simple Metropolis sampler with standard normal target and uniform
    innovations

    This function runs a simple Metropolis sampler with standard
    normal target distribution and uniform innovations.

    Parameters
    ----------
    n : int
      The number of iterations of the Metropolis sampler.
    alpha: float
      The tuning parameter of the sampler. The innovations of the sampelr are
      of the form U(-alpha, alpha).

    Returns
    -------
    A vector containing the output of the sampler.

    Examples
    --------
    >>> import smfsb
    >>> smfsb.metrop(100, 1)
    """
    vec = np.zeros((n))
    x = 0
    vec[0] = x
    for i in range(1, n):
        can = x + np.random.uniform(-alpha, alpha)
        aprob = norm.pdf(can)/norm.pdf(x)
        u = np.random.uniform()
        if (u < aprob):
            x = can
        vec[i] = x
    return vec



# eof

