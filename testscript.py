from spectral import *
import pandas as pd
import numpy as np
import scipy

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime
import pandas_datareader.data as web
from pymssa2 import MSSA

spy = web.DataReader("SPY", "av-daily-adjusted", start=datetime(2000, 2, 9), api_key=os.getenv('ALPHAVANTAGE_API_KEY')).close.map(np.log)
iwm = web.DataReader("IWM", "av-daily-adjusted", start=datetime(2000, 2, 9), api_key=os.getenv('ALPHAVANTAGE_API_KEY')).close.map(np.log)
dia = web.DataReader("DIA", "av-daily-adjusted", start=datetime(2000, 2, 9), api_key=os.getenv('ALPHAVANTAGE_API_KEY')).close.map(np.log)
qqq = web.DataReader("QQQ", "av-daily-adjusted", start=datetime(2000, 2, 9), api_key=os.getenv('ALPHAVANTAGE_API_KEY')).close.map(np.log)

mydf = pd.concat([spy, iwm, dia, qqq], axis=1)
mydf.index=pd.to_datetime(mydf.index)
mydf.columns = ['spy', 'iwm', 'dia', 'qqq']
mydf.dropna(inplace=True)
foor = dorange(mydf, ['spy', 'iwm', 'dia', 'qqq'], 10, 0, 1000, 1000)
foordf = pd.DataFrame(foor)
foordf.to_csv('foordf.csv')
