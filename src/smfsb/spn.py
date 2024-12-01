#!/usr/bin/env python3
# smfsb.py

import numpy as np


# Spn class definition, including methods


class Spn:
    """Class for stochastic Petri net models."""

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
        >>> stepSir = sir.step_poisson()
        >>> smfsb.sim_sample(10, sir.m, 0, 20, stepSir)
        """
        self.n = n  # species names
        self.t = t  # reaction names
        self.pre = np.array(pre)
        self.post = np.array(post)
        self.h = h  # hazard function
        self.m = np.array(m)  # initial marking

    def __str__(self):
        """A very simple string representation of the Spn object, mainly for debugging."""
        return "n: {}\n t: {}\npre: {}\npost: {}\nh: {}\nm: {}".format(
            str(self.n),
            str(self.t),
            str(self.pre),
            str(self.post),
            str(self.h),
            str(self.m),
        )

    def step_gillespie(self, min_haz=1e-10, max_haz=1e07):
        """Create a function for advancing the state of a SPN by using the
        Gillespie algorithm

        This method returns a function for advancing the state of an SPN
        model using the Gillespie algorithm. The resulting function
        (closure) can be used in conjunction with other functions (such as
        ‘sim_time_series’) for simulating realisations of SPN models.

        Parameters
        ----------
        min_haz : float
          Minimum hazard to consider before assuming 0. Defaults to 1e-10.
        max_haz : float
          Maximum hazard to consider before assuming an explosion and
          bailing out. Defaults to 1e07.

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
        >>> stepLv = lv.step_gillespie()
        >>> stepLv([50, 100], 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape

        def step(x0, t0, deltat):
            t = t0
            x = x0
            termt = t0 + deltat
            while True:
                h = self.h(x, t)
                h0 = h.sum()
                if h0 > max_haz:
                    print("WARNING: hazard too large - terminating!")
                    return x
                if h0 < min_haz:
                    t = 1e99
                else:
                    t = t + np.random.exponential(1.0 / h0)
                if t > termt:
                    return x
                j = np.random.choice(v, p=h / h0)
                x = np.add(x, sto[:, j])

        return step

    def step_first(self):
        """Create a function for advancing the state of an SPN by using
        Gillespie's first reaction method

        This function creates a function for advancing the state of an SPN
        model using Gillespie's first reaction method. The resulting
        function (closure) can be used in conjunction with other functions
        (such as ‘sim_time_series’) for simulating realisations of SPN models.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using Gillespie's first reaction method. The
        function closure returns a vector representing the simulated state
        of the system at the new time.

        Examples
        --------
        >>> import smfsb.models
        >>> lv = smfsb.models.lv()
        >>> stepLv = lv.step_first()
        >>> stepLv([50, 100], 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape

        def step(x0, t0, deltat):
            t = t0
            x = x0
            termt = t0 + deltat
            while True:
                h = self.h(x, t)
                h[h == 0.0] = 1e-10
                pu = np.random.exponential(1.0 / h)
                j = np.argmin(pu)
                t = t + pu[j]
                if t > termt:
                    return x
                x = np.add(x, sto[:, j])

        return step

    def step_poisson(self, dt=0.01):
        """Create a function for advancing the state of an SPN by using a
        simple approximate Poisson time stepping method

        This method returns a function for advancing the state of an SPN
        model using a simple approximate Poisson time stepping method. The
        resulting function (closure) can be used in conjunction with other
        functions (such as ‘sim_time_series’) for simulating realisations of SPN
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
        >>> stepLv = lv.step_poisson(0.001)
        >>> stepLv([50, 100], 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape

        def step(x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            while True:
                h = self.h(x, t)
                r = np.random.poisson(h * dt)
                x = np.add(x, sto.dot(r))
                t = t + dt
                if t > termt:
                    return x

        return step

    def step_euler(self, dt=0.01):
        """Create a function for advancing the state of an SPN by using a simple
        continuous deterministic Euler integration method

        This method returns a function for advancing the state of an SPN
        model using a simple continuous deterministic Euler integration
        method. The resulting function (closure) can be used in
        conjunction with other functions (such as ‘sim_time_series’) for simulating
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
        >>> stepLv = lv.step_euler(0.001)
        >>> stepLv([50, 100], 0, 1)
        """
        sto = (self.post - self.pre).T

        def step(x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            while True:
                h = self.h(x, t)
                x = np.add(x, sto.dot(h * dt))
                t = t + dt
                if t > termt:
                    return x

        return step

    def step_cle(self, dt=0.01):
        """Create a function for advancing the state of an SPN by using a simple
        Euler-Maruyama integration method for the associated CLE

        This method returns a function for advancing the state of an SPN
        model using a simple Euler-Maruyama integration method
        method for the chemical Langevin equation form of the model.The
        resulting function (closure) can be used in
        conjunction with other functions (such as ‘sim_time_series’) for simulating
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
        >>> stepLv = lv.step_cle(0.001)
        >>> stepLv([50, 100], 0, 1)
        """
        sto = (self.post - self.pre).T
        v = sto.shape[1]
        sdt = np.sqrt(dt)

        def step(x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            while True:
                h = self.h(x, t)
                dw = np.random.normal(scale=sdt, size=v)
                x = np.add(x, sto.dot(h * dt + np.sqrt(h) * dw))
                x[x < 0] = -x[x < 0]
                t = t + dt
                if t > termt:
                    return x

        return step

    # spatial simulation functions, from chapter 9

    def step_gillespie_1d(self, d, min_haz=1e-10, max_haz=1e07):
        """Create a function for advancing the state of an SPN by using the
        Gillespie algorithm on a 1D regular grid

        This method creates a function for advancing the state of an SPN
        model using the Gillespie algorithm. The resulting function
        (closure) can be used in conjunction with other functions (such as
        `sim_time_series_1d`) for simulating realisations of SPN models in space and
        time.

        Parameters
        ----------
        d : array
          A vector of diffusion coefficients - one coefficient for each
          reacting species, in order. The coefficient is the reaction
          rate for a reaction for a molecule moving into an adjacent
          compartment. The hazard for a given molecule leaving the
          compartment is therefore twice this value (as it can leave to
          the left or the right).
        min_haz : float
          Minimum hazard to consider before assuming 0. Defaults to 1e-10.
        max_haz : float
          Maximum hazard to consider before assuming an explosion and
          bailing out. Defaults to 1e07.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using the Gillespie algorithm. The function closure
        has arguments `x0`, `t0`, `deltat`, where `x0` is a matrix
        with rows corresponding to species and columns corresponding to
        voxels, representing the initial condition, `t0` represent the
        initial state and time, and `deltat` represents the amount of time
        by which the process should be advanced. The function closure
        returns a matrix representing the simulated state of the system at
        the new time.

        Examples
        --------
        >>> import smfsb.models
        >>> import numpy as np
        >>> lv = smfsb.models.lv()
        >>> stepLv1d = lv.step_gillespie_1d(np.array([0.6,0.6]))
        >>> N = 20
        >>> x0 = np.zeros((2,N))
        >>> x0[:,int(N/2)] = lv.m
        >>> stepLv1d(x0, 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape

        def step(x0, t0, deltat):
            t = t0
            x = x0
            n = x.shape[1]
            termt = t0 + deltat
            while True:
                hr = np.apply_along_axis(lambda xi: self.h(xi, t), 0, x)
                hrs = np.apply_along_axis(np.sum, 0, hr)
                hrss = hrs.sum()
                hd = np.apply_along_axis(lambda xi: xi * d * 2, 0, x)
                hds = np.apply_along_axis(np.sum, 0, hd)
                hdss = hds.sum()
                h0 = hrss + hdss
                if h0 > max_haz:
                    print("WARNING: hazard too large - terminating!")
                    return x
                if h0 < min_haz:
                    t = 1e99
                else:
                    t = t + np.random.exponential(1.0 / h0)
                if t > termt:
                    return x
                if np.random.uniform(0, h0) < hdss:
                    # diffuse
                    j = np.random.choice(n, p=hds / hdss)  # pick a box
                    i = np.random.choice(u, p=hd[:, j] / hds[j])  # pick species
                    x[i, j] = x[i, j] - 1  # decrement chosen box
                    if np.random.uniform(0, 1) < 0.5:
                        # left
                        if j > 0:
                            x[i, j - 1] = x[i, j - 1] + 1
                        else:
                            x[i, n - 1] = x[i, n - 1] + 1
                    else:
                        # right
                        if j < n - 1:
                            x[i, j + 1] = x[i, j + 1] + 1
                        else:
                            x[i, 0] = x[i, 0] + 1
                else:
                    # react
                    j = np.random.choice(n, p=hrs / hrss)  # pick a box
                    i = np.random.choice(v, p=hr[:, j] / hrs[j])  # pick a reaction
                    x[:, j] = np.add(x[:, j], sto[:, i])

        return step

    def step_gillespie_2d(self, d, min_haz=1e-10, max_haz=1e07):
        """Create a function for advancing the state of an SPN by using the
        Gillespie algorithm on a 2D regular grid

        This method creates a function for advancing the state of an SPN
        model using the Gillespie algorithm. The resulting function
        (closure) can be used in conjunction with other functions (such as
        `sim_time_series_2d`) for simulating realisations of SPN models in space and
        time.

        Parameters
        ----------
        d : array
          A vector of diffusion coefficients - one coefficient for each
          reacting species, in order. The coefficient is the reaction
          rate for a reaction for a molecule moving into an adjacent
          compartment. The hazard for a given molecule leaving the
          compartment is therefore four times this value (as it can leave in
          one of 4 directions).
        min_haz : float
          Minimum hazard to consider before assuming 0. Defaults to 1e-10.
        max_haz : float
          Maximum hazard to consider before assuming an explosion and
          bailing out. Defaults to 1e07.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using the Gillespie algorithm. The function closure
        has arguments `x0`, `t0`, `deltat`, where `x0` is a 3d array
        with dimensions corresponding to species then two spatial dimensions,
        representing the initial condition, `t0` represent the
        initial state and time, and `deltat` represents the amount of time
        by which the process should be advanced. The function closure
        returns an array representing the simulated state of the system at
        the new time.

        Examples
        --------
        >>> import smfsb.models
        >>> import numpy as np
        >>> lv = smfsb.models.lv()
        >>> stepLv2d = lv.step_gillespie_2d(np.array([0.6, 0.6]))
        >>> N = 20
        >>> x0 = np.zeros((2, N, N))
        >>> x0[:, int(N/2), int(N/2)] = lv.m
        >>> stepLv2d(x0, 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape

        def step(x0, t0, deltat):
            t = t0
            x = x0
            uu, m, n = x.shape
            termt = t0 + deltat
            while True:
                hr = np.apply_along_axis(lambda xi: self.h(xi, t), 0, x)
                hrs = np.sum(hr, axis=(0))
                hrss = hrs.sum()
                hd = np.apply_along_axis(lambda xi: xi * d * 4, 0, x)
                hds = np.sum(hd, axis=(0))
                hdss = hds.sum()
                h0 = hrss + hdss
                if h0 > max_haz:
                    print("WARNING: hazard too large - terminating!")
                    return x
                if h0 < min_haz:
                    t = 1e99
                else:
                    t = t + np.random.exponential(1.0 / h0)
                if t > termt:
                    return x
                if np.random.uniform(0, h0) < hdss:
                    # diffuse
                    r = np.random.choice(m * n, p=hds.flatten() / hdss)  # pick a box
                    i = r // n
                    j = r % n
                    k = np.random.choice(u, p=hd[:, i, j] / hds[i, j])  # pick species
                    x[k, i, j] = x[k, i, j] - 1  # decrement chosen box
                    un = np.random.uniform(0, 1)
                    if un < 0.25:
                        # left
                        if j > 0:
                            x[k, i, j - 1] = x[k, i, j - 1] + 1
                        else:
                            x[k, i, n - 1] = x[k, i, n - 1] + 1
                    elif un < 0.5:
                        # right
                        if j < n - 1:
                            x[k, i, j + 1] = x[k, i, j + 1] + 1
                        else:
                            x[k, i, 0] = x[k, i, 0] + 1
                    elif un < 0.75:
                        # up
                        if i > 0:
                            x[k, i - 1, j] = x[k, i - 1, j] + 1
                        else:
                            x[k, m - 1, j] = x[k, m - 1, j] + 1
                    else:
                        # down
                        if i < m - 1:
                            x[k, i + 1, j] = x[k, i + 1, j] + 1
                        else:
                            x[k, 0, j] = x[k, 0, j] + 1
                else:
                    # react
                    r = np.random.choice(m * n, p=hrs.flatten() / hrss)  # pick a box
                    i = r // n
                    j = r % n
                    k = np.random.choice(
                        v, p=hr[:, i, j] / hrs[i, j]
                    )  # pick a reaction
                    x[:, i, j] = np.add(x[:, i, j], sto[:, k])

        return step

    def step_cle_1d(self, d, dt=0.01):
        """Create a function for advancing the state of an SPN by using a simple
        Euler-Maruyama discretisation of the CLE on a 1D regular grid

        This method creates a function for advancing the state of an SPN
        model using a simple Euler-Maruyama discretisation of the CLE on a
        1D regular grid. The resulting function (closure) can be used in
        conjunction with other functions (such as `sim_time_series_1d`) for
        simulating realisations of SPN models in space and time.

        Parameters
        ----------
        d : array
          A vector of diffusion coefficients - one coefficient for each
          reacting species, in order. The coefficient is the reaction
          rate for a reaction for a molecule moving into an adjacent
          compartment. The hazard for a given molecule leaving the
          compartment is therefore twice this value (as it can leave to
          the left or the right).
        dt : float
          Time step for the Euler-Maruyama discretisation.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using a simple Euler-Maruyama algorithm. The function
        closure has parameters `x0`, `t0`, `deltat`, where `x0` is
        a matrix with rows corresponding to species and columns
        corresponding to voxels, representing the initial condition, `t0`
        represents the initial state and time, and `deltat` represents the
        amount of time by which the process should be advanced. The
        function closure returns a matrix representing the simulated state
        of the system at the new time.

        Examples
        --------
        >>> import smfsb.models
        >>> import numpy as np
        >>> lv = smfsb.models.lv()
        >>> stepLv1d = lv.step_cle_1d(np.array([0.6,0.6]))
        >>> N = 20
        >>> x0 = np.zeros((2,N))
        >>> x0[:,int(N/2)] = lv.m
        >>> stepLv1d(x0, 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape
        sdt = np.sqrt(dt)

        def forward(m):
            return np.roll(m, -1, axis=1)

        def back(m):
            return np.roll(m, +1, axis=1)

        def laplacian(m):
            return forward(m) + back(m) - 2 * m

        def rectify(m):
            m[m < 0] = 0
            return m

        def diffuse(m):
            n = m.shape[1]
            noise = np.random.normal(0, sdt, (u, n))
            m = (
                m
                + (np.diag(d) @ laplacian(m)) * dt
                + np.diag(np.sqrt(d))
                @ (np.sqrt(m + forward(m)) * noise - np.sqrt(m + back(m)) * back(noise))
            )
            m = rectify(m)
            return m

        def step(x0, t0, deltat):
            x = x0
            t = t0
            n = x0.shape[1]
            termt = t0 + deltat
            while True:
                x = diffuse(x)
                hr = np.apply_along_axis(lambda xi: self.h(xi, t), 0, x)
                dwt = np.random.normal(0, sdt, (v, n))
                x = x + sto @ (hr * dt + np.diag(np.sqrt(hr)) @ dwt)
                x = rectify(x)
                t = t + dt
                if t > termt:
                    return x

        return step

    def step_cle_2d(self, d, dt=0.01):
        """Create a function for advancing the state of an SPN by using a simple
        Euler-Maruyama discretisation of the CLE on a 2D regular grid

        This method creates a function for advancing the state of an SPN
        model using a simple Euler-Maruyama discretisation of the CLE on a
        2D regular grid. The resulting function (closure) can be used in
        conjunction with other functions (such as `sim_time_series_2d`) for
        simulating realisations of SPN models in space and time.

        Parameters
        ----------
        d : array
          A vector of diffusion coefficients - one coefficient for each
          reacting species, in order. The coefficient is the reaction
          rate for a reaction for a molecule moving into an adjacent
          compartment. The hazard for a given molecule leaving the
          compartment is therefore four times this value (as it can leave
          in one of 4 directions).
        dt : float
          Time step for the Euler-Maruyama discretisation.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using a simple Euler-Maruyama algorithm. The function
        closure has parameters `x0`, `t0`, `deltat`, where `x0` is
        a 3d array with indices species, then rows and columns
        corresponding to voxels, representing the initial condition, `t0`
        represents the initial state and time, and `deltat` represents the
        amount of time by which the process should be advanced. The
        function closure returns a matrix representing the simulated state
        of the system at the new time.

        Examples
        --------
        >>> import smfsb.models
        >>> import numpy as np
        >>> lv = smfsb.models.lv()
        >>> stepLv2d = lv.step_cle_2d(np.array([0.6,0.6]))
        >>> M = 15
        >>> N = 20
        >>> x0 = np.zeros((2,M,N))
        >>> x0[:,int(M/2),int(N/2)] = lv.m
        >>> stepLv2d(x0, 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape
        sdt = np.sqrt(dt)

        def left(a):
            return np.roll(a, -1, axis=1)

        def right(a):
            return np.roll(a, +1, axis=1)

        def up(a):
            return np.roll(a, -1, axis=2)

        def down(a):
            return np.roll(a, +1, axis=2)

        def laplacian(a):
            return left(a) + right(a) + up(a) + down(a) - 4 * a

        def rectify(a):
            a[a < 0] = 0
            return a

        def diffuse(a):
            uu, m, n = a.shape
            dwt = np.random.normal(0, sdt, (u, m, n))
            dwts = np.random.normal(0, sdt, (u, m, n))
            a = (
                a
                + (np.apply_along_axis(lambda xi: xi * d, 0, laplacian(a))) * dt
                + np.apply_along_axis(
                    lambda xi: xi * np.sqrt(d),
                    0,
                    (
                        np.sqrt(a + left(a)) * dwt
                        - np.sqrt(a + right(a)) * right(dwt)
                        + np.sqrt(a + up(a)) * dwts
                        - np.sqrt(a + down(a)) * down(dwts)
                    ),
                )
            )
            a = rectify(a)
            return a

        def step(x0, t0, deltat):
            x = x0
            t = t0
            uu, m, n = x0.shape
            termt = t0 + deltat
            while True:
                x = diffuse(x)
                hr = np.apply_along_axis(lambda xi: self.h(xi, t), 0, x)
                dwt = np.random.normal(0, sdt, (v, m, n))
                for i in range(m):
                    for j in range(n):
                        x[:, i, j] = x[:, i, j] + sto @ (
                            hr[:, i, j] * dt + np.sqrt(hr[:, i, j]) * dwt[:, i, j]
                        )
                x = rectify(x)
                t = t + dt
                if t > termt:
                    return x

        return step

    # some illustrative functions, not intended for serious use

    def gillespie(self, n):
        """Simulate a sample path from a stochastic kinetic model
        described by a stochastic Petri net

        This function simulates a single realisation from a discrete
        stochastic kinetic model described by a stochastic Petri net
        (SPN).

        Parameters
        ----------
        n: int
        An integer representing the number of events to simulate,
        excluding the initial state

        Returns
        -------
        A tuple consisting of a first component, a vector of length n
        containing the event times, and a second component, a matrix
        with n+1 rows containing the state of the system. The first
        row is the intial state prior to the first event.

        Examples
        --------
        >>> import smfsb.models
        >>> lv = smfsb.models.lv()
        >>> lv.gillespie(1000)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape
        t = 0
        x = self.m
        t_vec = np.zeros(n)
        x_mat = np.zeros((n + 1, u))
        x_mat[0, :] = x
        for i in range(n):
            h = self.h(x, t)
            h0 = h.sum()
            t = t + np.random.exponential(1.0 / h0)
            j = np.random.choice(v, p=h / h0)
            x = np.add(x, sto[:, j])
            x_mat[i + 1, :] = x
            t_vec[i] = t
        return t_vec, x_mat

    def gillespied(self, tt, dt=1):
        """Simulate a sample path from a stochastic kinetic model described by
        a stochastic Petri net

        This function simulates a single realisation from a discrete
        stochastic kinetic model described by a stochastic Petri net and
        discretises the output onto a regular time grid.

        Parameters
        ----------
        tt: float
        The required length of simulation time.
        dt: float
        The grid size for the output. Note that this parameter simply
        determines the volume of output. It has no bearing on the
        correctness of the simulation algorithm. Defaults to one time
        unit.

        Examples
        --------
        >>> import smfsb.models
        >>> lv = smfsb.models.lv()
        >>> lv.gillespied(30, 0.1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape
        t = 0
        n = int(tt / dt)
        x = self.m
        x_mat = np.zeros((n, u))
        i = 0
        target = 0
        while True:
            h = self.h(x, t)
            h0 = h.sum()
            if h0 < 1e-10:
                t = 1e99
            else:
                t = t + np.random.exponential(1.0 / h0)
            while t >= target:
                x_mat[i, :] = x
                i = i + 1
                target = target + dt
                if i >= n:
                    return x_mat
            j = np.random.choice(v, p=h / h0)
            x = np.add(x, sto[:, j])


# eof
