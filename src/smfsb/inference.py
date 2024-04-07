# inference.py
# Code relating to Chapters 10 and 11

import numpy as np
import math



def metropolisHastings(init, logLik, rprop,
                       ldprop=lambda n, o: 1, ldprior=lambda x: 1,
                       iters=10000, thin=10, verb=True, debug=False):
    """Run a Metropolis-Hastings MCMC algorithm for the parameters of a
    Bayesian posterior distribution

    Run a Metropolis-Hastings MCMC algorithm for the parameters of a
    Bayesian posterior distribution. Note that the algorithm carries
    over the old likelihood from the previous iteration, making it
    suitable for problems with expensive likelihoods, and also for
    "exact approximate" pseudo-marginal or particle marginal MH
    algorithms.
    
    Parameters
    ----------
    init : vector
      A parameter vector with which to initialise the MCMC algorithm.
    logLik : function
      A function which takes a parameter (such as `init`) as its
      only required argument and returns the log-likelihood of the
      data. Note that it is fine for this to return the log of an
      unbiased estimate of the likelihood, in which case the
      algorithm will be an "exact approximate" pseudo-marginal MH
      algorithm.
    rprop : stochastic function
      A function which takes a parameter as its only required
      argument and returns a single sample from a proposal
      distribution.
    ldprop : function
      A function which takes a new and old parameter as its first
      two required arguments and returns the log density of the
      new value conditional on the old. Defaults to a flat function which
      causes this term to drop out of the acceptance probability.
      It is fine to use the default for _any_ _symmetric_ proposal,
      since the term will also drop out for any symmetric proposal.
    ldprior : function
      A function which take a parameter as its only required
      argument and returns the log density of the parameter value
      under the prior. Defaults to a flat function which causes this 
      term to drop out of the acceptance probability. People often use 
      a flat prior when they are trying to be "uninformative" or
      "objective", but this is slightly naive. In particular, what
      is "flat" is clearly dependent on the parametrisation of the
      model.
    iters : int
      The number of MCMC iterations required (_after_ thinning).
    thin : int
      The required thinning factor. eg. only store every `thin`
      iterations.
    verb : boolean
      Boolean indicating whether some progress information should
      be printed to the console. Defaults to `True`.
    debug : boolean
      Boolean indicating whether debugging information is required.
      Prints information about each iteration to console, to, eg.,
      debug a crashing sampler.

    Returns
    -------
    A matrix with rows representing samples from the posterior
    distribution.

    Examples
    --------
    >>> import smfsb
    >>> import numpy as np
    >>> import scipy as sp
    >>> data = np.random.normal(5, 2, 250)
    >>> llik = lambda x: np.sum(sp.stats.norm.logpdf(data, x[0], x[1]))
    >>> prop = lambda x: np.random.normal(x, 0.1, 2)
    >>> smfsb.metropolisHastings([1,1], llik, prop)
    """
    p = len(init)
    ll = -math.inf
    mat = np.zeros((iters, p))
    x = init
    if (verb):
        print(f"{iters} iterations")
    for i in range(iters):
        if (verb):
            print(f"{i} ", end='', flush=True)
        for j in range(thin):
            prop = rprop(x)
            if (ldprior(prop) > -math.inf):
                llprop = logLik(prop)
                a = (llprop - ll + ldprior(prop) -
                     ldprior(x) + ldprop(x, prop) - ldprop(prop, x))
                if (debug):
                    print(f"x={x}, prop={prop}, ll={ll}, llprop={llprop}, a={a}")
                if (np.log(np.random.uniform()) < a):
                    x = prop
                    ll = llprop
        mat[i,:] = x
    if (verb):
        print("Done.")
    return mat
    

def abcRun(n, rprior, rdist, verb=False):
    """Run a set of simulations initialised with parameters sampled from a
    given prior distribution, and compute statistics required for an ABC
    analaysis

    Run a set of simulations initialised with parameters sampled from
    a given prior distribution, and compute statistics required for an
    ABC analaysis. Typically used to calculate "distances" of
    simulated synthetic data from observed data.
    
    Parameters
    ----------
    n : int
      An integer representing the number of simulations to run.
    rprior : function
      A function without arguments generating a single parameter
      (vector) from prior distribution.
    rdist : function
      A function taking a parameter (vector) as argument and
      returning the required statistic of interest. This will
      typically be computed by first using the parameter to run a
      forward model, then computing required summary statistics,
      then computing a distance. See the example for details.
    verb : boolean
      Print progress information to console?
    
    Returns
    -------
    A tuple with first component a list of parameters and second component
    a list of corresponding distances.
 
    Examples
    --------
    >>> import smfsb
    >>> import numpy as np
    >>> import scipy as sp
    >>> data = np.random.normal(5, 2, 250)
    >>> def rpr():
    >>>   return np.exp(np.random.uniform(-3, 3, 2))
    >>>
    >>> def rmod(th):
    >>>   return np.random.normal(th[0], th[1], 250)
    >>>
    >>> def sumStats(dat):
    >>>   return np.array([np.mean(dat), np.std(dat)])
    >>>
    >>> ssd = sumStats(data)
    >>> def dist(ss):
    >>>   diff = ss - ssd
    >>>   return np.sqrt(np.sum(diff*diff))
    >>>
    >>> def rdis(th):
    >>>   return dist(sumStats(rmod(th)))
    >>>
    >>> smfsb.abcRun(100, rpr, rdis)
    """
    p = list()
    d = list()
    for i in range(n):
        if (verb):
            print(n-i, end=' ', flush=True)
        pi = rprior()
        di = rdist(pi)
        p.append(pi)
        d.append(di)
    if (verb):
        print(" - Done.")
    return (p, d)


def pfMLLik(n, simX0, t0, stepFun, dataLLik, data, debug=False):
    """Create a function for computing the log of an unbiased estimate of
    marginal likelihood of a time course data set

    Create a function for computing the log of an unbiased estimate of
    marginal likelihood of a time course data set using a simple
    bootstrap particle filter.

    Parameters
    ----------
    n :  int
      An integer representing the number of particles to use in the
      particle filter.
    simX0 : function
      A function with arguments `t0` and `th`, where ‘t0’ is a time 
      at which to simulate from an initial distribution for the state of the
      particle filter and `th` is a vector of parameters. The return value 
      should be a state vector randomly sampled from the prior distribution.
      The function therefore represents a prior distribution on the initial
      state of the Markov process.
    t0 : float
      The time corresponding to the starting point of the Markov
      process. Can be no bigger than the smallest observation time.
    stepFun : function
      A function for advancing the state of the Markov process, with
      arguments `x`, `t0`, `deltat` and `th`, with `th` representing a
      vector of parameters.
    dataLLik : function
      A function with arguments `x`, `t`, `y`, `th`,
      where `x` and `t` represent the true state and time of the
      process, `y` is the observed data, and `th` is a parameter vector. 
      The return value should be the log of the likelihood of the observation. The
      function therefore represents the observation model.
    data : matrix
      A matrix with first column an increasing set of times. The remaining
      columns represent the observed values of `y` at those times.

    Returns
    -------
    A function with single argument `th`, representing a parameter vector, which
    evaluates to the log of the particle filters unbiased estimate of the
    marginal likelihood of the data (for parameter `th`).

    Examples
    --------
    >>> import numpy as np
    >>> import scipy as sp
    >>> import smfsb
    >>> def obsll(x, t, y, th):
    >>>     return np.sum(sp.stats.norm.logpdf(y-x, scale=10)
    >>> 
    >>> def simX(t0, th):
    >>>     return np.array([np.random.poisson(50), np.random.poisson(100)])
    >>> 
    >>> def step(x, t, dt, th):
    >>>     sf = smfsb.models.lv(th).stepGillespie()
    >>>     return sf(x, t, dt)
    >>> 
    >>> mll = smfsb.pfMLLik(80, simX, 0, step, obsll, smfsb.data.LVnoise10)
    >>> mll(np.array([1, 0.005, 0.6]))
    >>> mll(np.array([2, 0.005, 0.6]))
    """
    no = data.shape[1]
    times = np.concatenate(([t0], data[:,0]))
    deltas = np.diff(times)
    obs = data[:,range(1,no)]
    if (debug):
        print(data.shape)
        print(times[range(5)])
        print(deltas[range(5)])
        print(len(deltas))
        print(obs[range(5),:])
    def go(th):
        ll = 0
        xmat = np.zeros((n,1))
        xmat = np.apply_along_axis(lambda x: simX0(t0, th), 1, xmat)
        sh = xmat.shape
        if (debug):
            print(xmat.shape)
            print(xmat[range(5),:])
        for i in range(len(deltas)):
            xmat = np.apply_along_axis(lambda x: stepFun(
                x, times[i], deltas[i], th), 1, xmat)
            lw = np.apply_along_axis(lambda x: dataLLik(
                x, times[i+1], obs[i,], th), 1, xmat)
            m = np.max(lw)
            sw = np.exp(lw - m)
            ssw = np.sum(sw)
            ll = ll + m + np.log(ssw/n)
            rows = np.random.choice(n, n, p=sw/ssw)
            xmat = xmat[rows,:]
            assert(xmat.shape == sh)
        return ll
    return go


def abcSmcStep(dprior, priorSample, priorLW, rdist, rperturb,
               dperturb, factor):
    """Carry out one step of an ABC-SMC algorithm

    Not meant to be directly called by users. See abcSmc.
    """
    n = priorSample.shape[0]
    mx = np.max(priorLW)
    rw = np.exp(priorLW - mx)
    #print(priorSample.shape)
    #print(len(rw))
    priorInd = np.random.choice(range(n), n*factor, p=rw/np.sum(rw))
    prior = priorSample[priorInd,:]
    #print(prior.shape)
    prop = np.apply_along_axis(rperturb, 1, prior)
    #print(prop.shape)
    dist = np.apply_along_axis(rdist, 1, prop)
    #print(dist.shape)
    qCut = np.nanquantile(dist, 1/factor)
    new = prop[dist < qCut,:]
    def logWeight(th):
        terms = priorLW + np.apply_along_axis(lambda x: dperturb(th, x),
                                              1, priorSample)
        mt = np.max(terms)
        denom = mt + np.log(np.sum(np.exp(terms-mt)))
        return dprior(th) - denom
    lw = np.apply_along_axis(logWeight, 1, new)
    mx = np.max(lw)
    rw = np.exp(lw - mx)
    nlw = np.log(rw/np.sum(rw))
    #print(f"new: {new.shape}")
    #print(f"nlw: {nlw.shape}")
    #print(nlw)
    return new, nlw


def abcSmc(N, rprior, dprior, rdist, rperturb, dperturb,
           factor=10, steps=15, verb=False, debug=False):
    """Run an ABC-SMC algorithm for infering the parameters of a forward model

    Run an ABC-SMC algorithm for infering the parameters of a forward
    model. This sequential Monte Carlo algorithm often performs better
    than simple rejection-ABC in practice.

    Parameters
    ----------
    N : int
      An integer representing the number of simulations to pass on
      at each stage of the SMC algorithm. Note that the TOTAL
      number of forward simulations required by the algorithm will
      be (roughly) 'N*steps*factor'.
    rprior : function
      A function without arguments generating single parameter
      (vector) from the prior.
    dprior : function
      A function taking a parameter vector as argumnent and returning
      the log of the prior density.
    rdist : function
      A function taking a parameter (vector) as argument and
      returning a scalar "distance" representing a measure of how
      good the chosen parameter is. This will typically be computed
      by first using the parameter to run a forward model, then
      computing required summary statistics, then computing a
      distance. See the example for details.
    rperturb : function
      A function which takes a parameter as its argument and
      returns a perturbed parameter from an appropriate kernel.
    dperturb : function
      A function which takes a pair of parameters as its first two
      arguments (new first and old second), and returns the log of the density
      associated with this perturbation kernel.
    factor : int
      At each step of the algorithm, 'N*factor' proposals are
      generated and the best 'N' of these are weighted and passed
      on to the next stage. Note that the effective sample size of
      the parameters passed on to the next step may be (much)
      smaller than 'N', since some of the particles may be assigned
      small (or zero) weight.
    steps : int
      The number of steps of the ABC-SMC algorithm. Typically,
      somewhere between 5 and 100 steps seems to be used in
      practice.
    verb : boolean
      Boolean indicating whether some progress should be printed to
      the console (the number of steps remaining).
    
    Returns
    -------
    A matrix with rows representing samples from the approximate posterior
    distribution.

    Examples
    --------
    TODO: do after writing test
    
    """
    priorLW = np.log(np.zeros((N)) + 1/N)
    priorSample = np.zeros((N,1))
    priorSample = np.apply_along_axis(lambda x: rprior(), 1, priorSample)
    for i in range(steps):
        if (verb):
            print(steps-i, end=' ', flush=True)
        priorSample, priorLW = abcSmcStep(dprior, priorSample, priorLW,
                                          rdist, rperturb, dperturb, factor)
        if (debug):
            print(priorSample.shape)
            print(priorLW.shape)
    if (verb):
        print("Done.")
    if (debug):
        print(priorSample.shape)
        print(priorLW.shape)
    #print(priorLW)
    ind = np.random.choice(range(priorLW.shape[0]), N, p = np.exp(priorLW))
    #print(ind)
    return priorSample[ind,:]






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

