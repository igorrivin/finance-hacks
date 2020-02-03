from pandashacks import *
import pandas
import numpy
from pymssa import MSSA

def doforecast(df, cols, howmany, start=None, end=None, winsize=None, indices=None):
    newdf = df[[cols]]
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

