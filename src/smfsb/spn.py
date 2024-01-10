#!/usr/bin/env python3
# smfsb.py

import numpy as np


# Spn class definition, including methods

class Spn:
    """Class for stochastic Petri net models.
    """

    
    def __init__(self, n, t, pre, post, h, m):
        """Constructor method for Spn objects

        Create a Spn object for representing a stochastic Petri net model that
        can be simulated using a variety of algorithms.

        Parameters
        ----------
        n : list of strings
            Names of the species/tokens in the model
        t : list of strings
            Names of the reactions/transitions in the model
        pre : matrix
            Matrix representing the LHS stoichiometries
        post: matrix
            Matrix representing the RHS stoichiometries
        h: function
            A function to compute the rates of the reactions from the current state and time of
            the system. The function should return a numpy array of rates.
        m: list of integers
            The intial state/marking of the model/net

        Returns
        -------
        A object of class Spn.

        Examples
        --------
        >>> import smfsb
        >>> import numpy as np
        >>> sir = smfsb.Spn(["S", "I", "R"], ["S->I", "I->R"],
              [[1,1,0],[0,1,0]], [[0,2,0],[0,0,1]],
	      lambda x, t: np.array([0.3*x[0]*x[1]/200, 0.1*x[1]]),
	      [197, 3, 0])
        >>> stepSir = sir.stepPTS()
        >>> smfsb.simSample(10, sir.m, 0, 20, stepSir)
        """
        self.n = n # species names
        self.t = t # reaction names
        self.pre = np.matrix(pre)
        self.post = np.matrix(post)
        self.h = h # hazard function
        self.m = np.array(m) # initial marking


        
    def __str__(self):
        """A very simple string representation of the Spn object, mainly for debugging.
        """
        return "n: {}\n t: {}\npre: {}\npost: {}\nh: {}\nm: {}".format(str(self.n),
                str(self.t), str(self.pre), str(self.post), str(self.h), str(self.m))


    
    def stepGillespie(self):
        """Create a function for advancing the state of a SPN by using the
        Gillespie algorithm

        This method returns a function for advancing the state of an SPN
        model using the Gillespie algorithm. The resulting function
        (closure) can be used in conjunction with other functions (such as
        ‘simTs’) for simulating realisations of SPN models.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using the Gillespie algorithm. The function closure
        has interface ‘function(x0, t0, deltat)’, where ‘x0’ and ‘t0’
        represent the initial state and time, and ‘deltat’ represents the
        amount of time by which the process should be advanced. The
        function closure returns a vector representing the simulated state
        of the system at the new time.

        Examples
        --------
        >>> import smfsb.models
        >>> lv = smfsb.models.lv()
        >>> stepLv = lv.stepGillespie()
        >>> stepLv([50, 100], 0, 1)
        """
        S = (self.post - self.pre).T
        u, v = S.shape
        def step(x0, t0, deltat):
            t = t0
            x = x0
            termt = t0 + deltat
            while(True):
                h = self.h(x, t)
                h0 = h.sum()
                if (h0 > 1e07):
                    print("WARNING: hazard too large - terminating!")
                    return(x)
                if (h0 < 1e-10):
                    t = 1e99
                else:
                    t = t + np.random.exponential(1.0/h0)
                if (t > termt):
                    return(x)
                j = np.random.choice(v, p=h/h0)
                x = np.add(x, S[:,j].A1)
        return step


    
    def stepPTS(self, dt = 0.01):
        """Create a function for advancing the state of an SPN by using a 
        simple approximate Poisson time stepping method

        This method returns a function for advancing the state of an SPN
        model using a simple approximate Poisson time stepping method. The
        resulting function (closure) can be used in conjunction with other
        functions (such as ‘simTs’) for simulating realisations of SPN
        models.

        Parameters
        ----------
        dt : float
            The time step for the time-stepping integration method. Defaults to 0.01.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using a Poisson time stepping method with step size
        ‘dt’. The function closure has interface
        ‘function(x0, t0, deltat)’, where ‘x0’ and ‘t0’ represent the
        initial state and time, and ‘deltat’ represents the amount of time
        by which the process should be advanced. The function closure
        returns a vector representing the simulated state of the system at
        the new time.

        Examples
        --------
        >>> import smfsb.models
        >>> lv = smfsb.models.lv()
        >>> stepLv = lv.stepPTS(0.001)
        >>> stepLv([50, 100], 0, 1)
        """
        S = (self.post - self.pre).T
        u, v = S.shape
        def step(x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            while(True):
                h = self.h(x, t)
                r = np.random.poisson(h * dt)
                x = np.add(x, S.dot(r).A1)
                t = t + dt
                if (t > termt):
                    return x
        return step


    
    def stepEuler(self, dt = 0.01):
        """Create a function for advancing the state of an SPN by using a simple
        continuous deterministic Euler integration method

        This method returns a function for advancing the state of an SPN
        model using a simple continuous deterministic Euler integration
        method. The resulting function (closure) can be used in
        conjunction with other functions (such as ‘simTs’) for simulating
        realisations of SPN models.

        Parameters
        ----------
        dt : float
            The time step for the time-stepping integration method. Defaults to 0.01.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using an Euler method with step size ‘dt’. The
        function closure has interface ‘function(x0, t0, deltat)’, where
        ‘x0’ and ‘t0’ represent the initial state and time, and ‘deltat’
        represents the amount of time by which the process should be
        advanced. The function closure returns a vector representing the
        simulated state of the system at the new time.

        Examples
        --------
        >>> import smfsb.models
        >>> lv = smfsb.models.lv()
        >>> stepLv = lv.stepEuler(0.001)
        >>> stepLv([50, 100], 0, 1)
        """
        S = (self.post - self.pre).T
        def step(x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            while(True):
                h = self.h(x, t)
                x = np.add(x, S.dot(h*dt).A1)
                t = t + dt
                if (t > termt):
                    return x
        return step


    def stepCLE(self, dt = 0.01):
        """Create a function for advancing the state of an SPN by using a simple
        Euler-Maruyama integration method for the associated CLE

        This method returns a function for advancing the state of an SPN
        model using a simple Euler-Maruyama integration method
        method for the chemical Langevin equation form of the model.The 
        resulting function (closure) can be used in
        conjunction with other functions (such as ‘simTs’) for simulating
        realisations of SPN models.

        Parameters
        ----------
        dt : float
            The time step for the time-stepping integration method. Defaults to 0.01.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using an Euler-Maruyama method with step size ‘dt’. The
        function closure has interface ‘function(x0, t0, deltat)’, where
        ‘x0’ and ‘t0’ represent the initial state and time, and ‘deltat’
        represents the amount of time by which the process should be
        advanced. The function closure returns a vector representing the
        simulated state of the system at the new time.

        Examples
        --------
        >>> import smfsb.models
        >>> lv = smfsb.models.lv()
        >>> stepLv = lv.stepCLE(0.001)
        >>> stepLv([50, 100], 0, 1)
        """
        S = (self.post - self.pre).T
        v = S.shape[1]
        sdt = np.sqrt(dt)
        def step(x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            while(True):
                h = self.h(x, t)
                dw = np.random.normal(scale=sdt, size=v)
                x = np.add(x, S.dot(h*dt + np.sqrt(h)*dw).A1)
                x[x<0] = -x[x<0]
                t = t + dt
                if (t > termt):
                    return x
        return step








# eof

