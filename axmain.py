import pandas as pd
import numpy as np
import argparse
import pandas_datareader.data as web
from datetime import datetime
import os
import sys

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.metrics.branin import branin
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax import Experiment, save
from ax import load

from spectral import *

def main(args):
    begbeg = int(args.begbeg)
    begend = int(args.begend)
    endbeg = int(args.endbeg)
    endend = int(args.endend)
    howmany = int(args.howmany)
    poolsize = int(args.poolsize)
    fname = args.outfile
    shortname=args.shortout

    spy = web.DataReader("SPY", "av-daily-adjusted", start=datetime(2000, 2, 9), api_key=os.getenv('ALPHAVANTAGE_API_KEY')).close
    dia = web.DataReader("DIA", "av-daily-adjusted", start=datetime(2000, 2, 9), api_key=os.getenv('ALPHAVANTAGE_API_KEY')).close
    qqq = web.DataReader("QQQ", "av-daily-adjusted", start=datetime(2000, 2, 9), api_key=os.getenv('ALPHAVANTAGE_API_KEY')).close

    mydf = pd.concat([spy, dia, qqq], axis=1)
    mydf.index=pd.to_datetime(mydf.index)
    mydf.columns = ['spy', 'dia', 'qqq']
    mydf['ordnum'] = mydf.reset_index().index
    mydf.dropna(inplace=True)

    ssaeval = trialscored(mydf, ['spy', 'dia', 'qqq'], (begbeg, begend), (endbeg, endend), howmany, poolsize, doscorelog)
    best_parameters, values, experiment, model = optimize(
        parameters = [
            {
                "name": "howmany",
                "type": "range",
                "bounds" : [1, 40],
                "value_type" : "int",
            },
            {
                "name": "components",
                "type": "range",
                "bounds": [1, 40],
                "value_type" : "int",
            },
            {
                "name": "winsize",
                "type": "fixed",
                "value": True,
            },
        ],
        experiment_name="ssa_test",
        objective_name="ssa",
<<<<<<< HEAD
        total_trials = 100,
        evaluation_function = ssaeval,
=======
        total_trials=1000,
        evaluation_function = ssaeval
>>>>>>> a34dcd1c768aa04a92efa91cc62a0365e9235134
    )
    shortdesc = open(shortname, "a")
    print(best_parameters, file=shortdesc)
    print(values, file=shortdesc)
    shortdesc.close()
    save(experiment, fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('begbeg', help='beginning of the begin range')
    parser.add_argument('begend', help='end of begin range')
    parser.add_argument('endbeg', help='beginning of the end range')
    parser.add_argument('endend', help='end of the end range')
    parser.add_argument('howmany', help='how many days')
    parser.add_argument('poolsize', help='how many processes to spawn')
    parser.add_argument('outfile', help='output file path')
    parser.add_argument('shortout', help='brief summary')
    args=parser.parse_args()
    sys.exit(main(args))
