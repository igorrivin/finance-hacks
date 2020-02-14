from pandashacks import *
import pandas as pd
import numpy as np
from pymssa2 import MSSA

def doforecast(df, cols, howmany, start=None, end=None, winsize=None, indices=None):
    newdf = df[cols]
    if start is None and end is None:
        newdf = newdf.iloc[:-howmany]
    elif start is None:
        newdf = newdf.iloc[:end-howmany]
    else:
        newdf = newdf.iloc[start:end-howmany]
    mssa = MSSA(n_components='parallel_analysis', 
                pa_percentile_threshold=95, 
                window_size=winsize,
                verbose=True)
    mssa.fit(newdf)
    fc = mssa.forecast(howmany, timeseries_indices=indices)
    return fc

def fullpred(df, cols, howmany=1, beg=None, end=None):
    newcols = detrend(df, cols, beg, end)
    mssa = MSSA(n_components='parallel_analysis',pa_percentile_threshold=95,window_size=None, verbose=True)
    X_train = mssa.fit(df[newcols].iloc[beg:end])
    fc = mssa.forecast(howmany)
    return fc

def dorange(df, cols, howmany=1, beg=None, end=None, iters=1):
    ar = []
    for i in range(iters):
        res = fullpred(df, cols, howmany, beg, end+i)
        ar.append(res[:, -1])
    return np.vstack(ar)
