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

## Basic usage

Note that **the book**, and its associated [github repo](https://github.com/darrenjw/smfsb) is the main source of documentation for this library. The code in the book is in R, but the code in this library is supposed to mirror the R code, but in Python.

### Using a model built-in to the library

First, see how to simulate a built-in model:
```python
import smfsb

lv = smfsb.models.lv()
print(lv)
stepLv = lv.stepGillespie()
out = smfsb.simTs(lv.m, 0, 100, 0.1, stepLv)
```
Now, if `matplotlib` is installed, you can plot the output with
```python
import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(2):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(lv.n)
fig.savefig("lv.pdf")
```
Standard python docstring documentation is available. Usage information can be obtained from the python REPL with commands like `help(smfsb.Spn)`, `help(smfsb.Spn.stepGillespie)` or `help(smfsb.simTs)`. This documentation is also available on [ReadTheDocs](https://python-smfsb.readthedocs.io/). The API documentation contains very minimal usage examples. For more interesting examples, see the [demos directory](https://github.com/darrenjw/python-smfsb/tree/main/demos).

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
Note that if you are working with SBML models in Python using [libsbml](https://pypi.org/project/python-libsbml/), then there is also a function `model2Spn` which takes a libsbml model object.

To read and parse an SBML-shorthand model, use
```python
m = smfsb.mod2Spn("myModel.mod")
```
There is also a function `sh2Spn` which expects a python string containing a shorthand model. This is convenient for embedding shorthand models inside python scripts, and is particularly convenient when working with things like Jupyter notebooks. Below follows a complete session to illustrate the idea by creating and simulating a realisation from a discrete stochastic SEIR model.
```python
import smfsb
import numpy as np

seirSH = """
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

seir = smfsb.sh2Spn(seirSH)
stepSeir = seir.stepGillespie()
out = smfsb.simTs(seir.m, 0, 40, 0.05, stepSeir)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(len(seir.m)):
	axis.plot(np.arange(0, 40, 0.05), out[:,i])

axis.legend(seir.n)
fig.savefig("seir.pdf")
```


A [collection of appropriate models](https://github.com/darrenjw/smfsb/tree/master/models) is associated with the book.



You can see this package on [PyPI](https://pypi.org/project/smfsb/) or [GitHub](https://github.com/darrenjw/python-smfsb).


**Copyright (2023-2024) Darren J Wilkinson**


