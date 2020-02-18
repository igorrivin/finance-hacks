from pandashacks import *
import pandas as pd
import numpy as np
from pymssa2 import MSSA
from multiprocessing import Pool

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
    mssa = MSSA(n_components=11,pa_percentile_threshold=95,window_size=None, verbose=True)
    X_train = mssa.fit(df[newcols].iloc[beg:end])
    fc = mssa.forecast(howmany)
    return fc, idict, cdict

def processpred(indlist, ilist, clist, ar, cols):
    indser = pd.Series(indlist, name='ordnum')
    idf = pd.DataFrame(ilist)[cols]
    cdf = pd.DataFrame(clist)[cols]
    cdf2 = cdf.multiply(indser, axis=0)
    resdf = pd.DataFrame(np.vstack(ar), columns=cols)
    newdf = resdf * cdf2 + idf
    newcols = [i+"_pred" for i in newdf.columns]
    newdf.columns = newcols
    newdf = pd.concat([indser, newdf], axis=1)
    return newdf, resdf
    
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
    # indser = pd.Series(indlist, name='ordnum')
    # idf = pd.DataFrame(ilist)[cols]
    # cdf = pd.DataFrame(clist)[cols]
    # cdf2 = cdf.multiply(indser, axis=0)
    # resdf = pd.DataFrame(np.vstack(ar), columns=cols)
    # newdf = resdf * cdf2 + idf
    # newcols = [i+"_pred" for i in newdf.columns]
    # newdf.columns = newcols
    # newdf = pd.concat([indser, newdf], axis=1)
    # return newdf, resdf
    return processpred(indlist, ilist, clist, ar, cols)

class oneprediction(object):
    def __init__(self, df, cols, howmany, beg, end):
        self.df = df
        self.cols = cols
        self.howmany =howmany
        self.beg = beg
        self.end = end

    def __call__(self, i):
        return fullpred(self.df, self.cols, self.howmany, self.beg, self.end + i)


def dorangemulti(df, cols, howmany=1, beg=None, end=None, iters=1, poolsize=16):
    if beg is None:
        beg=0
    if end is None:
        end = len(df)
    pool=Pool(poolsize)
    predatomic=oneprediction(df, cols, howmany, beg, end)
    tmp = pool.map(predatomic, range(iters))
    reslist = [i[0] for i in tmp]
    ar = [res[:, -1] for res in reslist]
    ilist = [i[1] for i in tmp]
    clist = [i[2] for i in tmp]
    indlist = [end + howmany + i -1 for i in range(iters)]
    return processpred(indlist, ilist, clist, ar, cols)
