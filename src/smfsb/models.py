# models.py
# some example Spn models

from smfsb import *



def lv(th=[1, 0.005, 0.6]):
    return Spn(["Prey", "Predator"], ["Prey rep", "Inter", "Pred death"],
               [[1,0],[1,1],[0,1]], [[2,0],[0,2],[0,0]],
               lambda x, t: np.array([th[0]*x[0], th[1]*x[0]*x[1], th[2]*x[1]]),
               [50,100])



def sir(th=[0.0015, 0.1]):
    return Spn(["S", "I", "R"], ["S->I", "I->R"], [[1,1,0],[0,1,0]], [[0,2,0],[0,0,1]],
               lambda x, t: np.array([th[0]*x[0]*x[1], th[1]*x[1]]),
               [197, 3, 0])



def dimer(th=[0.00166, 0.2]):
    return Spn(["P", "P2"], ["Dim", "Diss"], [[2,0],[0,1]], [[0,1],[2,0]],
               lambda x, t: np.array([th[0]*x[0]*(x[0]-1)/2, th[1]*x[1]]),
               [301, 0])









# eof

