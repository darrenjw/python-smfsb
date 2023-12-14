# smfsb for python

Python library for the book, Stochastic modelling for systems biology, third edition

**Please note that this package is still in the early stages of development. There will be bugs, there is very limited documentation, and the coverage of the book is still quite incomplete.**

## Install

Latest stable version:

```bash
pip install smfsb
```

## Basic usage

Note that **the book**, and its associated [github repo](https://github.com/darrenjw/smfsb) is the main source of documentation for this library. The code in the book is in R, but the code in this library is supposed to mirror the R code, but in Python.

### Using a model built-in to the library

First, see how to simulate a built-in model:
```python
import smfsb

print(smfsb.lv)
stepLv = smfsb.lv.stepGillespie()
out = smfsb.simTs(smfsb.lv.m, 0, 100, 0.1, stepLv)
```
Now, if `matplotlib` is installed, you can plot the output with
```python
import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(2):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(smfsb.lv.n)
fig.savefig("lv.pdf")
```

### Creating and simulating a model

Next, let's create and simulate our own model by specifying a stochastic Petri net explicitly.
```python
import numpy as np
sir = smfsb.Spn(["S", "I", "R"], ["S->I", "I->R"],
	[[1,1,0],[0,1,0]], [[0,2,0],[0,0,1]],
	lambda x, t: np.array([0.3*x[0]*x[1]/200, 0.1*x[1]]),
	[197, 3, 0])
stepSir = sir.stepPTS()
sample = smfsb.simSample(500, sir.m, 0, 20, stepSir)
fig, axis = plt.subplots()
axis.hist(sample[:,1],30)
axis.set_title("Infected at time 20")
plt.savefig("sIr.pdf")
```

### Reading and parsing models in SBML and SBML-shorthand

Note that you can read in SBML or SBML-shorthand models that have been designed for discrete stochastic simulation into a stochastic Petri net directly. To read and parse an SBML model, use
```python
m = smfsb.file2Spn("myModel.xml")
```
To read and parse an SBML-shorthand model, use
```python
m = smfsb.mod2Spn("myModel.mod")
```
A [collection of appropriate models](https://github.com/darrenjw/smfsb/tree/master/models) is associated with the book.



You can see this package on [PyPI](https://pypi.org/project/smfsb/) or [GitHub](https://github.com/darrenjw/python-smfsb).


**Copyright (2023) Darren J Wilkinson**


