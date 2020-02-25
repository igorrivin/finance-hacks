from spectral import *
import pandas as pd
import numpy as np
import scipy
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime
import pandas_datareader.data as web
from pymssa2 import MSSA

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

spy = web.DataReader("SPY", "av-daily-adjusted", start=datetime(2000, 2, 9), api_key=os.getenv('ALPHAVANTAGE_API_KEY')).close
dia = web.DataReader("DIA", "av-daily-adjusted", start=datetime(2000, 2, 9), api_key=os.getenv('ALPHAVANTAGE_API_KEY')).close
qqq = web.DataReader("QQQ", "av-daily-adjusted", start=datetime(2000, 2, 9), api_key=os.getenv('ALPHAVANTAGE_API_KEY')).close

mydf = pd.concat([spy, dia, qqq], axis=1)
mydf.index=pd.to_datetime(mydf.index)
mydf.columns = ['spy', 'dia', 'qqq']
mydf['ordnum'] = mydf.reset_index().index
mydf.dropna(inplace=True)
mydf.to_csv('hist.csv')
foor = dorangemulti_simple(mydf, ['spy', 'dia', 'qqq'], 40, 0, 1000, 3923, 16, 15, None)
foor.to_csv('preds.csv')

