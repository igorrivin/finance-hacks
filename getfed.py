#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np

fname=sys.argv[1]


fedout = pd.read_json('https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key=91c6e96d21f86b0932368859c1f6bc3f&file_type=json')



fedvals = pd.DataFrame(list(fedout.observations))

fedvals.date = pd.to_datetime(fedvals.date)

fedvals = fedvals.set_index('date')

alldays = pd.date_range(start='1954-07-01', end=pd.Timestamp.today())

fedvals = fedvals.reindex(alldays, method='ffill')

fedvals.value.to_csv(fname)





