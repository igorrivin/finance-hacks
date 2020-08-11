import os
import glob
import pandas as pd
def make_csvs(base, holdtime, pref=None):
    if pref is None:
        paths = os.path.join(base + "*_" + str(holdtime) + ".csv")
    else:
        paths = os.path.join(pref, base + "*_" + str(holdtime) + ".csv")
    flist = glob.glob(paths)
    dflist = []
    for i in flist:
        dflist.append(pd.read_csv(i, index_col=0, parse_dates=True))
    thedf = pd.concat(dflist).sort_index()
    thedf.to_csv(base + "new_" + str(holdtime) + ".csv")
    #thedf[thedf.index.year > 2012].to_csv(base + "2018_" + str(holdtime) + ".csv")
