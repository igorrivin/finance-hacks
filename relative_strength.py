import pandas as pd
import numpy as np

def doaverages(df, col, longdur, shortdur, holdtime, suffixshort='_short', suffixlong='_long', suffixgain='_gain'):
    df1 = df.copy()
    shortname = col+suffixshort
    longname = col + suffixlong
    gainname = col + suffixgain
    df1[shortname] = df1[col].rolling(shortdur).mean()
    df1[longname] = df1[col].rolling(longdur).mean()
    df1[gainname] = - df1[col].diff(-holdtime)
    return df1, col, shortname, longname, gainname


def move_eval(df, shortcol, longcol, gaincol, thebound, thesign):
    filtered = df[df[longcol] - df[shortcol] * thesign > thebound * thesign]
    totgain = filtered[gaincol].sum()
    return totgain

class r_evaluator(object):
    def __init__(self, df, col):
        self.data = df
        self.col = col
    def __call__(self, vardict):
        longdur = vardict['longdur']
        shortdur = vardict['shortdur']
        thebound= vardict['thebound']
        thesign = vardict['thesign']
        holdtime = vardict['holdtime']
        newdata, _, shortname, longname, gainname = doaverages(self.data, self.col, longdur, shortdur, holdtime)
        return move_eval(newdata, shortname, longname, gainname, thebound, thesign)/holdtime
        

        