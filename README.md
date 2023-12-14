# smfsb for python

Python library for the book, Stochastic modelling for systems biology, third edition

**Please note that this package is still in the early stages of development. There will be bugs, there is very limited documentation, and the coverage of the book is still quite incomplete.**

## Install

Latest stable version:

```bash
pip install smfsb
```

## Basic usage

```python
import smfsb

print(smfsb.lv)
stepLv = smfsb.lv.stepGillespie()
out = smfsb.simTs(smfsb.lv.m, 0, 100, 0.1, stepLv)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(2):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(smfsb.lv.n)
fig.savefig("lv.pdf")
```


Also see [smfsb on PyPI](https://pypi.org/project/smfsb/)


**Copyright (2023) Darren J Wilkinson**


