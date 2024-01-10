# models.py
# some example Spn models

from smfsb import *



def lv(th=[1, 0.005, 0.6]):
    """Create a Lotka-Volterra model

    Create and return a Spn object representing a discrete stochastic
    Lotka-Volterra model.
    
    Parameters
    ----------
    th: array
        array of length 3 containing the rates of the three governing reactions

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import smfsb
    >>> lv = smfsb.models.lv()
    >>> step = lv.stepGillespie()
    >>> smfsb.simTs(lv.m, 0, 50, 0.1, step)
    """
    return Spn(["Prey", "Predator"], ["Prey rep", "Inter", "Pred death"],
               [[1,0],[1,1],[0,1]], [[2,0],[0,2],[0,0]],
               lambda x, t: np.array([th[0]*x[0], th[1]*x[0]*x[1], th[2]*x[1]]),
               [50,100])



def sir(th=[0.0015, 0.1]):
    """Create a basic SIR compartmental epidemic model

    Create and return a Spn object representing a discrete stochastic
    SIR model.
    
    Parameters
    ----------
    th: array
        array of length 2 containing the rates of the two governing transitions

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import smfsb
    >>> sir = smfsb.models.sir()
    >>> step = sir.stepGillespie()
    >>> smfsb.simTs(sir.m, 0, 50, 0.1, step)
    """
    return Spn(["S", "I", "R"], ["S->I", "I->R"], [[1,1,0],[0,1,0]], [[0,2,0],[0,0,1]],
               lambda x, t: np.array([th[0]*x[0]*x[1], th[1]*x[1]]),
               [197, 3, 0])



def dimer(th=[0.00166, 0.2]):
    """Create a dimerisation kinetics model

    Create and return a Spn object representing a discrete stochastic
    dimerisation kinetics model.
    
    Parameters
    ----------
    th: array
        array of length 2 containing the rates of the bind and unbind reactions

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import smfsb
    >>> dimer = smfsb.models.dimer()
    >>> step = dimer.stepGillespie()
    >>> smfsb.simTs(dimer.m, 0, 50, 0.1, step)
    """
    return Spn(["P", "P2"], ["Dim", "Diss"], [[2,0],[0,1]], [[0,1],[2,0]],
               lambda x, t: np.array([th[0]*x[0]*(x[0]-1)/2, th[1]*x[1]]),
               [301, 0])









# eof

