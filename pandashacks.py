import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def detrend(df, cols, inplace=False, suffix="_dt"):
    oldindex = df.index.name
    df.reset_index(inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index':'ordnum'})
    ordf = df[['ordnum']]
    df.set_index(oldindex, inplace=True)
    for c in cols:
        curcol = df[c]
        thereg = LinearRegression()
        thereg.fit(ordf, curcol)
        residuals = curcol - thereg.predict(ordf)
        if inplace is True:
            df[c] = residuals
        else:
            newname = c + suffix
            df[newname] = residuals
    df.drop('ordnum', axis=1, inplace=True)
    return df
    
def dodrawdown(df):
    """computes the drawdown of a time series."""
    dfsum = df.cumsum()
    dfmax = dfsum.cummax()
    drawdown  = - min(dfsum-dfmax)
    return drawdown

#lots of fun rolling statistics

def rollingnormalize(df, window):
    """
    Takes a dataframe with per stock and per date data, with stock as level 1 in    the multiindex, and some signal values, and does a rolling normalization. Do    es not drop the nans, since you might need them for posterity. 
    """
    gr = df.groupby(level=1).apply(lambda x:
    (x-x.rolling(window).mean())/x.rolling(window).std()) 
    return gr

def getbetas(df, market, window = 45):
    """ given an unstacked pandas dataframe (columns instruments, rows
    dates), compute the rolling betas vs the market.
    """
    nmarket = market/market.rolling(window).var()
    thebetas = df.rolling(window).cov(other=nmarket)
    return thebetas

def adjrets(df, market, window = 45, thebetas = None):
    """beta-adjusted returns"""
    if thebetas is None:
        thebetas = getbetas(df, market, window)
    newvals = df.values - thebetas.values * market[:, np.newaxis]
    newdf = pd.DataFrame(newvals, columns = df.columns, index = df.index)
    return newdf.dropna()

from numpy.linalg import pinv, svd
import cvxopt

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P),cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G),cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    theans = np.array(sol['x']).reshape(-1)
    return theans

def dokelly(df, ir=0, G=None, h=None, alpha=0):
    """ given a dataframe of return time series, computes the Kelly
    portfolio composition. If constraints are given, give the constrained Kelly.
    """
    cmat= df.cov().values
    means = df.mean().values
    if G is None:
        adjmeans = means - ir
        pseu = pinv(cmat)
        prekelly = pseu.dot(adjmeans)
    else:
        adjmeans = means
        if alpha is 0:
            prekelly = cvxopt_solve_qp(cmat, -adjmeans, G, h)
        else:
            U, s, V = svd(cmat)
            tops = s.max()
            theeye = np.identity(cmat.shape[0])
            themat = cmat + alpha * tops * theeye
            prekelly = cvxopt_solve_qp(themat, -adjmeans, G, h)
            
    return (1+ir)*prekelly

# Correct Sharpe computation

def realvol(s, n):
    themean = s.mean()+1
    thevar = s.var()
    m2 = themean * themean
    t1 = thevar + m2
    thestd =  math.sqrt(math.pow(t1, n) - math.pow(m2, n))
    return thestd

def realmean(s, n):
    themean = s.mean()+1
    totalmean = math.pow(themean, n) - 1
    return totalmean

def realsharpe(s, n):
    return realmean(s, n)/realvol(s, n)
