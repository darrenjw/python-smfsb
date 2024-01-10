#!/usr/bin/env python3
# smfsbSbml.py
# Import SBML models into an smfsb Spn

import libsbml
import smfsb
import sys
import numpy as np
import smfsb.mod2sbml as mod2sbml


def mod2Spn(filename, verb=False):
    """Convert an SBML-shorthand model into a Spn object

    Read a file containing a model in SBML-shorthand and convert into
    an Spn object for simulation and analysis.

    Parameters
    ----------
    filename: string
        String name of file containing the model
    verb: boolean
        Output some debugging info

    Returns
    -------
    An Spn object

    Examples
    --------
    >>> import smfsb
    >>> myMod = smfsb.mod2spn("myModel.mod")
    >>> step = myMod.stepGillespie()
    """
    try:
        s = open(filename, "r")
    except:
        sys.stderr.write("Error: failed to open "+filename+"\n")
        sys.exit(1)
    p = mod2sbml.Parser()
    d = p.parseStream(s)
    m = d.getModel()
    if (m == None):
        sys.stderr.write("Error: can't extract SBML model\n")
        sys.exit(1)
    return(model2Spn(m, verb))



def sh2Spn(shString, verb=False):
    """Convert an SBML-shorthand model string into a Spn object

    Parse a string containing a model in SBML-shorthand and convert into
    an Spn object for simulation and analysis.

    Parameters
    ----------
    shString: string
        String containing the model
    verb: boolean
        Output some debugging info

    Returns
    -------
    An Spn object

    Examples
    --------
    >>> import smfsb
    >>> file = open('myModel.mod', 'r')
    >>> myModStr = file.read()
    >>> file.close()
    >>> myMod = smfsb.sh2spn(myModStr)
    >>> step = myMod.stepGillespie()
    """
    p = mod2sbml.Parser()
    d = p.parse(shString)
    m = d.getModel()
    if (m == None):
        sys.stderr.write("Error: couldn't parse the shorthand string\n")
        sys.exit(1)
    return(model2Spn(m, verb))



def file2Spn(filename, verb=False):
    """Convert an SBML model into a Spn object

    Read a file containing a model in SBML and convert into
    an Spn object for simulation and analysis.

    Parameters
    ----------
    filename: string
        String name of file containing the model
    verb: boolean
        Output some debugging info

    Returns
    -------
    An Spn object

    Examples
    --------
    >>> import smfsb
    >>> myMod = smfsb.mod2spn("myModel.xml")
    >>> step = myMod.stepGillespie()
    """
    d = libsbml.readSBML(filename)
    m = d.getModel()
    if (m == None):
        sys.stderr.write("Can't parse SBML file: "+filename+"\n")
        sys.exit(1)
    return(model2Spn(m, verb))



def model2Spn(m, verb=False):
    """Convert a libSBML model into a Spn object

    Convert a libSBML model into a Spn object for simulation and analysis.

    Parameters
    ----------
    m: model
        A libsbml model (not document) object
    verb: boolean
        Output some debugging info

    Returns
    -------
    An Spn object

    Examples
    --------
    >>> import smfsb
    >>> import libsbml
    >>> d = libsbml.readSBML("myModel.xml")
    >>> m = d.getModel()
    >>> myMod = smfsb.model2Spn(m)
    >>> step = myMod.stepGillespie()
    """
    # Species and initial amounts
    ns = m.getNumSpecies()
    if (verb):
        print(str(ns)+" species")
    ml = []
    nl = []
    for i in range(ns):
        s = m.getSpecies(i)
        nl += [s.getId()]
        ml += [s.getInitialAmount()]
    if (verb):
        print(nl)
        print(ml)
    # Compartments
    nc = m.getNumCompartments()
    cd = {}
    for i in range(nc):
        comp = m.getCompartment(i)
        cd[comp.getId()] = comp.getVolume()
    if (verb):
        print(cd)
    # Global parameters
    ngp = m.getNumParameters()
    gpd = {}
    for i in range(ngp):
        param = m.getParameter(i)
        gpd[param.getId()] = param.getValue()
    if (verb):
        print(gpd)
    # Reactions
    nr = m.getNumReactions()
    if (verb):
        print(str(nr)+" reactions")
    pre = np.zeros((nr, ns))
    post = np.zeros((nr, ns))
    rn = []
    kl = []
    lpl = []
    for i in range(nr):
        r = m.getReaction(i)
        rn += [r.getId()]
        nPre = r.getNumReactants()
        for j in range(nPre):
            sr = r.getReactant(j)
            sto = sr.getStoichiometry()
            pre[i, nl.index(sr.getSpecies())] = sto
        nPost = r.getNumProducts()
        for j in range(nPost):
            sr = r.getProduct(j)
            sto = sr.getStoichiometry()
            post[i, nl.index(sr.getSpecies())] = sto
        kli = r.getKineticLaw()
        kl += [libsbml.formulaToString(kli.getMath())]
        nlp = kli.getNumLocalParameters()
        lpd = {}
        for j in range(nlp):
            param = kli.getLocalParameter(j)
            lpd[param.getId()] = param.getValue()
        lpl += [lpd]
    if (verb):
        print(rn)
        print("Pre:")
        print(pre)
        print("Post:")
        print(post)
        print(kl)
        print(lpl)
    gpd.update(cd)
    def haz(x, t):
        h = np.zeros(nr)
        xd = dict(zip(nl, x))
        glob = gpd.copy()
        glob.update(xd)
        for i in range(nr):
            h[i] = eval(kl[i], glob, lpl[i])
        return(h)
    spn = smfsb.Spn(nl, rn, pre, post, haz, ml)
    spn.comp = cd
    spn.gp = gpd
    spn.kl = kl
    spn.lp = lpl
    return(spn)


# eof


