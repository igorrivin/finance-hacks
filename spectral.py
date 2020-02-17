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
    if beg is None:
        beg = 0
    if end is None:
        end = len(df)
    newcols, idict, cdict = detrend(df, cols, beg, end)
    #mssa = MSSA(n_components='parallel_analysis',pa_percentile_threshold=95,window_size=None, verbose=True)
    mssa = MSSA(n_components=11,pa_percentile_threshold=95,window_size=None, verbose=True,svd_method='randomized_gpu')
    X_train = mssa.fit(df[newcols].iloc[beg:end])
    fc = mssa.forecast(howmany)
    return fc, idict, cdict

def dorange(df, cols, howmany=1, beg=None, end=None, iters=1):
    ar = []
    ilist = []
    clist = []
    indlist = []
    if beg is None:
        beg=0
    if end is None:
        end = len(df)
    for i in range(iters):
        res, idict, cdict = fullpred(df, cols, howmany, beg, end+i)
        ar.append(res[:, -1])
        ilist.append(idict)
        clist.append(cdict)
        indlist.append(end+howmany+i-1)
    indser = pd.Series(indlist, name='ordnum')
    #print(indser)
    idf = pd.DataFrame(ilist)[cols]
    #print(idf)
    cdf = pd.DataFrame(clist)[cols]
    #print(cdf)
    cdf2 = cdf.multiply(indser, axis=0)
    #print(cdf2)
    resdf = pd.DataFrame(np.vstack(ar), columns=cols)
    #print(resdf)
    newdf = resdf * cdf2 + idf
    #print(newdf)
    newdf = pd.concat([indser, newdf], axis=1)
    return newdf, resdf


