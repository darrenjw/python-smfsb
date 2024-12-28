The python-smfsb tutorial
-------------------------

This tutorial assumes that the package has already been installed, following the instructions in the `package readme <https://pypi.org/project/smfsb/>`__.

We begin with non-spatial stochastic simulation.

Non-spatial simulation
----------------------

Standard algorithms for simulating the (stochastic) dynamics of biochemical networks assume that the system is well-mixed, and that spatial effects can be reasonably ignored.

Using a model built-in to the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's see how to simulate a built-in (Lotka-Volterra predator-prey)
model:

.. code:: python

   import smfsb

   lvmod = smfsb.models.lv()
   step = lvmod.step_gillespie()
   out = smfsb.sim_time_series(lvmod.m, 0, 30, 0.1, step)
   assert(out.shape == (300, 2))

Here we used the ``lv`` model. Other built-in models include ``id`` (immigration-death), ``bd`` (birth-death), ``dimer`` (dimerisation kinetics), ``mm`` (Michaelis-Menten enzyme kinetics) and ``sir`` (SIR epdiemic model). The models are of class ``Spn`` (stochastic Petri net), the main data type used in the package. Note the use of the ``step_gillespie`` method, defined on all ``Spn`` models, which returns a function for simulating from the transition kernel of the model, using the Gillespie algorithm. This function can be used with the ``sim_time_series`` function for simulating model trajectories on a regular time grid. Alternative simulation algorithms include ``step_poisson`` (Poisson time-stepping), ``step_cle`` (Euler-Maruyama simulation from the associated chemical Langevin equation) and ``step_euler`` (Euler simulation from the continuous deterministic approximation to the model).

If you have ``matplotlib`` installed (``pip install matplotlib``), then
you can also plot the results with:

.. code:: python

   import matplotlib.pyplot as plt
   fig, axis = plt.subplots()
   for i in range(2):
       axis.plot(range(out.shape[0]), out[:,i])

   axis.legend(lvmod.n)
   fig.savefig("lv.pdf")

Standard python docstring documentation is available. Usage information
can be obtained from the python REPL with commands like
``help(smfsb.Spn)``, ``help(smfsb.Spn.step_gillespie)`` or
``help(smfsb.sim_time_series)``. This documentation is also available
on `ReadTheDocs <https://python-smfsb.readthedocs.io/>`__. The API
documentation contains minimal usage examples.

Creating and simulating a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, let’s create and simulate our own (SIR epidemic) model by
specifying a stochastic Petri net ``Spn`` object explicitly. We must provide species and reaction names, stoichiometry matrices, reaction rates and initial conditions. This time we use approximate Poisson simulation rather than exact simulation via the Gillespie algorithm.

.. code:: python

   import numpy as np
   sir = smfsb.Spn(["S", "I", "R"], ["S->I", "I->R"],
       [[1,1,0], [0,1,0]], [[0,2,0], [0,0,1]],
       lambda x, t: np.array([0.3*x[0]*x[1]/200, 0.1*x[1]]),
       [197.0, 3, 0])
   step_sir = sir.step_poisson()
   sample = smfsb.sim_sample(500, sir.m, 0, 20, step_sir)
   fig, axis = plt.subplots()
   axis.hist(sample[:,1], 30)
   axis.set_title("Infected at time 20")
   plt.savefig("sIr.pdf")

Here, rather than simulating a time series trajectory, we instead simulate a sample of 500 values from the transition kernel at time 20 using ``sim_sample``.


Reading and parsing models in SBML and SBML-shorthand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that you can read in `SBML <https://sbml.org/>`__ or `SBML-shorthand <https://pypi.org/project/sbmlsh/>`__ models that have been
designed for discrete stochastic simulation into a stochastic Petri net
directly. To read and parse an SBML model, use

.. code:: python

   m = smfsb.file_to_spn("myModel.xml")

Note that if you are working with SBML models in Python using
`libsbml <https://pypi.org/project/python-libsbml/>`__, then there is
also a function ``model_to_spn`` which takes a libsbml model object.

To read and parse an SBML-shorthand model, use

.. code:: python

   m = smfsb.mod_to_spn("myModel.mod")

There is also a function ``shorthand_to_spn`` which expects a python
string containing a shorthand model. This is convenient for embedding
shorthand models inside python scripts, and is particularly convenient
when working with things like Jupyter notebooks. Below follows a
complete session to illustrate the idea by creating and simulating a
realisation from a discrete stochastic SEIR model.

.. code:: python

   import smfsb
   import numpy as np

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

   import matplotlib.pyplot as plt
   fig, axis = plt.subplots()
   for i in range(len(seir.m)):
       axis.plot(np.arange(0, 40, 0.05), out[:,i])

   axis.legend(seir.n)
   fig.savefig("seir.pdf")

A `collection of appropriate
models <https://github.com/darrenjw/smfsb/tree/master/models>`__ is
associated with the book.

Spatial simulation
------------------

In addition to methods such as ``step_gillespie`` and ``step_cle`` for well-mixed simulation, ``Spn`` objects also have methods such as ``step_gillespie_1d`` and ``step_cle_2d`` for 1d and 2d spatially explicit simulation of reaction-diffusion processes on a regular grid. These functions expect to be passed an array containing the diffusion coefficient for each species.

1d simulation
~~~~~~~~~~~~~

For 1d simulation, the state is a matrix with rows representing the levels of a given species on a 1d grid. The 1d transition kernels will update such a state. The function ``sim_time_series_1d`` will return a 3d array, with 2d slices representing the state at each time point. Slicing on the first index shows the spatio-temporal evolution of a given species.

.. code:: python

   import smfsb
   import numpy as np
   import matplotlib.pyplot as plt

   N = 20
   T = 25
   x0 = np.zeros((2, N))
   lv = smfsb.models.lv()
   x0[:, int(N / 2)] = lv.m
   step_lv_1d = lv.step_gillespie_1d(np.array([0.6, 0.6]))
   x1 = step_lv_1d(x0, 0, 1)
   print(x1)
   out = smfsb.sim_time_series_1d(x0, 0, T, 1, step_lv_1d, True)

   fig, axis = plt.subplots()
   for i in range(2):
       axis.imshow(out[i, :, :])
       axis.set_title(lv.n[i])
       fig.savefig(f"step_gillespie_1d{i}.pdf")


2d simulation
~~~~~~~~~~~~~

For 2d simulation, the state is a 3d array containing the levels of each species on a 2d grid. The 2d transition kernels will update such a state. Slicing on the first index will show the 2d spatial distribution of a given species.

.. code:: python

   import smfsb
   import numpy as np
   import matplotlib.pyplot as plt

   M = 50
   N = 60
   T = 25
   x0 = np.zeros((2, M, N))
   lv = smfsb.models.lv()
   x0[:, int(M / 2), int(N / 2)] = lv.m
   step_lv_2d = lv.step_cle_2d(np.array([0.6, 0.6]), 0.1)
   x1 = step_lv_2d(x0, 0, T)

   fig, axis = plt.subplots()
   for i in range(2):
       axis.imshow(x1[i, :, :])
       axis.set_title(lv.n[i])
       fig.savefig(f"step_cle_2df{i}.pdf")

Note that on fine 2d grids, approximate simulation using ``step_cle_2d`` is much typically much faster than exact simulation from the reaction diffusion master equation (RDME) using ``step_gillespie_2d``.
    

Bayesian parameter inference
----------------------------

In addition to providing tools for forward-simulation from stochastic kinetic models, the library also provides tools for conducting Bayesian parameter inference for stochastic kinetic models based on observed time course data. eg. given an observed (noisy) trajectory of one or more species from a given model, find rate constants that are most consistent with the observed data. The methods provided are simulation-based, or likelihood-free, based on either `approximate Bayesian computation <https://en.wikipedia.org/wiki/Approximate_Bayesian_computation>`__ (ABC) or (bootstrap) `particle marginal Metropolis-Hastings <https://darrenjw.wordpress.com/2011/05/17/the-particle-marginal-metropolis-hastings-pmmh-particle-mcmc-algorithm/>`__ (PMMH) particle MCMC.

ABC
~~~

In a very basic version of ABC, a candidate parameter vector is drawn from a prior distribution. This parameter vector is used in conjunction with a forward-simulation algorithm for the model of interest in order to generate a synthetic data set. This synthetic data set is compared against the real data set. If they are sufficiently "close", the originally sampled parameter vector will be kept as a sample from the posterior distribution, otherwise it will be rejected, and the process will start again. The function ``abc_run`` helps to scaffold this process. A complete example using simple euclidean distance between the real and synthetic trajectories is presented below.

.. code:: python

   import smfsb
   import numpy as np
   import matplotlib.pyplot as plt

   data = smfsb.data.lv_perfect[:, 1:3]

   def rpr():
       return np.exp(
	   np.array(
	       [
		   np.random.uniform(-3, 3),
		   np.random.uniform(-8, -2),
		   np.random.uniform(-4, 2),
	       ]
	   )
       )

   def rmod(th):
       return smfsb.sim_time_series(
	   np.array([50.0, 100.0]), 0, 30, 2, smfsb.models.lv(th).step_cle(0.1)
       )

   def sum_stats(dat):
       return dat

   ssd = sum_stats(data)

   def dist(ss):
       diff = ss - ssd
       return np.sqrt(np.sum(diff * diff))

   def rdis(th):
       return dist(sum_stats(rmod(th)))

   p, d = smfsb.abc_run(100000, rpr, rdis, verb=False)

   q = np.nanquantile(d, 0.02)
   prmat = np.vstack(p)
   postmat = prmat[d < q, :]
   its, var = postmat.shape
   print(its, var)

   postmat = np.log(postmat)  # look at posterior on log scale

   fig, axes = plt.subplots(2, 3)
   axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
   axes[0, 1].scatter(postmat[:, 0], postmat[:, 2], s=0.5)
   axes[0, 2].scatter(postmat[:, 1], postmat[:, 2], s=0.5)
   axes[1, 0].hist(postmat[:, 0], bins=30)
   axes[1, 1].hist(postmat[:, 1], bins=30)
   axes[1, 2].hist(postmat[:, 2], bins=30)
   fig.savefig("abc.pdf")

Using simple euclidean distance between the trajectories is probably not a great idea. See the file ``abc-cal.py`` in the `demo directory <https://github.com/darrenjw/python-smfsb/tree/main/demos>`__ for an example using more sophisticated summary statistics, calibrated via a pilot run to be on a consistent scale.

   
ABC-SMC
~~~~~~~

Even using well-tuned summary statistics, naive rejection-based ABC is a rather inefficient algorithm. By combining ideas of ABC with those of `sequential Monte Carlo <https://en.wikipedia.org/wiki/Particle_filter>`__ (SMC) one can develop an ABC-SMC algorithm which gradually "zooms in" on promising parts of the parameter space using a sequence of updates in conjunction with a parameter purturbation kernel. The precise details are beyond the scope of this tutorial, but below is a complete example, using calibrated summary statistics from a pilot run. The function ``abc_smc`` performs the Bayesian update.

.. code:: python

   import smfsb
   import numpy as np
   import scipy as sp
   import matplotlib.pyplot as plt

   data = smfsb.data.lv_perfect[:, 1:3]

   def rpr():
       return np.array(
	   [
	       np.random.uniform(-2, 2),
	       np.random.uniform(-7, -3),
	       np.random.uniform(-3, 1),
	   ]
       )

   def dpr(th):
       return np.sum(
	   np.log(
	       np.array(
		   [
		       ((th[0] > -2) & (th[0] < 2)) / 4,
		       ((th[1] > -7) & (th[1] < -3)) / 4,
		       ((th[2] > -3) & (th[2] < 1)) / 4,
		   ]
	       )
	   )
       )

   def rmod(th):
       return smfsb.sim_time_series(
	   [50.0, 100], 0, 30, 2, smfsb.models.lv(np.exp(th)).step_cle(0.1)
       )

   print("Pilot run...")

   def ss1d(vec):
       n = len(vec)
       mean = np.nanmean(vec)
       v0 = vec - mean
       var = np.nanvar(v0)
       acs = [
	   np.corrcoef(v0[0 : (n - 1)], v0[1:n])[0, 1],
	   np.corrcoef(v0[0 : (n - 2)], v0[2:n])[0, 1],
	   np.corrcoef(v0[0 : (n - 3)], v0[3:n])[0, 1],
       ]
       return np.array([np.log(mean + 1), np.log(var + 1), acs[0], acs[1], acs[2]])

   def ssi(ts):
       return np.concatenate(
	   (
	       ss1d(ts[:, 0]),
	       ss1d(ts[:, 1]),
	       np.array([np.corrcoef(ts[:, 0], ts[:, 1])[0, 1]]),
	   )
       )

   p, d = smfsb.abc_run(20000, rpr, lambda th: ssi(rmod(th)), verb=False)
   prmat = np.vstack(p)
   dmat = np.vstack(d)
   print(prmat.shape)
   print(dmat.shape)
   dmat[dmat == np.inf] = np.nan
   sds = np.nanstd(dmat, 0)
   print(sds)

   def sum_stats(dat):
       return ssi(dat) / sds

   ssd = sum_stats(data)

   print("Main ABC-SMC run")

   def dist(ss):
       diff = ss - ssd
       return np.sqrt(np.sum(diff * diff))

   def rdis(th):
       return dist(sum_stats(rmod(th)))

   def rper(th):
       return th + np.random.normal(0, 0.5, 3)

   def dper(ne, ol):
       return np.sum(sp.stats.norm.logpdf(ne, ol, 0.5))

   postmat = smfsb.abc_smc(
       5000, rpr, dpr, rdis, rper, dper, factor=5, steps=6, verb=True
   )

   its, var = postmat.shape
   print(its, var)

   fig, axes = plt.subplots(2, 3)
   axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
   axes[0, 1].scatter(postmat[:, 0], postmat[:, 2], s=0.5)
   axes[0, 2].scatter(postmat[:, 1], postmat[:, 2], s=0.5)
   axes[1, 0].hist(postmat[:, 0], bins=30)
   axes[1, 1].hist(postmat[:, 1], bins=30)
   axes[1, 2].hist(postmat[:, 2], bins=30)
   fig.savefig("abc_smc.pdf")



PMMH particle MCMC
~~~~~~~~~~~~~~~~~~

PMMH is in many ways the "gold standard" likelihood free inference strategy (at least in the case of noisy observations). By combining an unbiased estimate of the model's marginal likelihood (computed using a particle filter) with a Metropolis-Hastings MCMC algorithm, it is possible to generate a Markov chain with equilibrium distribution equal to the exact posterior distribution of the parameters given the observations. Again, the technical details are beyond the scope of this tutorial, but a complete example is given below. The key functions are ``pf_marginal_ll`` and ``metropolis_hastings``.

.. code:: python

   import smfsb
   import mcmc  # extra functions in the demo directory
   import scipy as sp
   import numpy as np

   def obsll(x, t, y, th):
       return np.sum(sp.stats.norm.logpdf(y - x, scale=10))

   def sim_x(t0, th):
       return np.array([np.random.poisson(50), np.random.poisson(100)])

   def step(x, t, dt, th):
       sf = smfsb.models.lv(th).step_cle(0.1)
       return sf(x, t, dt)

   mll = smfsb.pf_marginal_ll(100, sim_x, 0, step, obsll, smfsb.data.lv_noise_10)

   def prop(th, tune=0.01):
       return np.exp(np.random.normal(0, tune, (3))) * th

   thmat = smfsb.metropolis_hastings(
	  np.array([1, 0.005, 0.6]), mll, prop, iters=5000, thin=1, verb=True
   )

   mcmc.mcmc_summary(thmat, "pmmh.pdf")

Note that the summary stats and plots are produced using some additional functions defined in the file ``mcmc.py`` in the demo directory.


Further information
-------------------

For further information, see the `demo
directory <https://github.com/darrenjw/python-smfsb/tree/main/demos>`__ and
the `API
documentation <https://python-smfsb.readthedocs.io/en/latest/index.html>`__.


The ``jax-smfsb`` python package
--------------------------------

If you like this package, but find it to be too slow for serious work, then you may be interested in the `jax-smfsb <https://github.com/darrenjw/jax-smfsb/>`__ package. This is a port of the main simulation and inference functions from this library to the `JAX <https://jax.readthedocs.io/>`__ machine learning framework, offering JIT compilation and parallelisation. The API for the library is very similar to that of this one. The main difference is that non-deterministic (random)
functions have an extra argument (typically the first argument) that
corresponds to a JAX random number key. The functions in the JAX port can often be two orders of magnitude faster than those in this package for non-trivial simulation or inference algorithms.





