# smfsb for python

Python library for the book, [Stochastic modelling for systems biology, third edition](https://github.com/darrenjw/smfsb/). This library is a Python port of the R package associated with the book.

## Install

Latest stable version:
```bash
pip install smfsb
```
To upgrade already installed package:
```bash
pip install --upgrade smfsb
```

**Note** that a number of breaking syntax changes (more pythonic names) were introduced in version 1.1.0. If you upgrade to a version >= 1.1.0 from a version prior to 1.1.0 you will have to update syntax to the new style.

You can test your installation by typing
```python
import smfsb
```
at a python prompt. If it returns silently, then it is probably installed correctly.

## Documentation

Note that **the book**, and its associated [github repo](https://github.com/darrenjw/smfsb) is the main source of documentation for this library. The code in the book is in R, but the code in this library is supposed to mirror the R code, but in Python.

For an introduction to this library, see the [python-smfsb tutorial](https://python-smfsb.readthedocs.io/en/latest/source/tutorial.html).

## Further information

For further information, see the [demo directory](https://github.com/darrenjw/python-smfsb/tree/main/demos) and the [API documentation](https://python-smfsb.readthedocs.io/en/latest/index.html). Within the demos directory, see [sbmlsh-demo.py](https://github.com/darrenjw/python-smfsb/tree/main/demos/sbmlsh-demo.py) for an example of how to specify a (SEIR epidemic) model using SBML-shorthand and [step_cle_2df.py](https://github.com/darrenjw/python-smfsb/tree/main/demos/step_cle_2df.py) for a 2-d reaction-diffusion simulation. For parameter inference (from time course data), see [abc-cal.py](https://github.com/darrenjw/python-smfsb/tree/main/demos/abc-cal.py) for ABC inference, [abc_smc.py](https://github.com/darrenjw/python-smfsb/tree/main/demos/abc_smc.py) for ABC-SMC inference and [pmmh.py](https://github.com/darrenjw/python-smfsb/tree/main/demos/pmmh.py) for particle marginal Metropolis-Hastings MCMC-based inference. There are many other demos besides these.


You can see this package on [PyPI](https://pypi.org/project/smfsb/) or [GitHub](https://github.com/darrenjw/python-smfsb).


## Fast simulation and inference

If you like this library but find it a little slow, you should know that there is a [JAX](https://jax.readthedocs.io/) port of this package: [jax-smfsb](https://github.com/darrenjw/jax-smfsb). It requires a JAX installalation, and the API is (very) slightly modified, but it has state-of-the-art performance for simulation and inference.


**Copyright (2023-2024) Darren J Wilkinson**


