# models.py
# some example Spn models

from smfsb import *

lv = Spn(["Prey", "Predator"], ["Prey rep", "Inter", "Pred death"],
         [[1,0],[1,1],[0,1]], [[2,0],[0,2],[0,0]],
         lambda x, t: np.array([x[0], 0.005*x[0]*x[1], 0.6*x[1]]),
         [50,100])

sir = Spn(["S", "I", "R"], ["S->I", "I->R"], [[1,1,0],[0,1,0]], [[0,2,0],[0,0,1]],
          lambda x, t: np.array([0.3*x[0]*x[1]/200, 0.1*x[1]]),
          [197, 3, 0])

dimer = Spn(["P", "P2"], ["Dim", "Diss"], [[2,0],[0,1]], [[0,1],[2,0]],
          lambda x, t: np.array([0.00166*x[0]*(x[0]-1)/2, 0.2*x[1]]),
          [301, 0])





# eof

