# Author: Pranav Mahajan, 2021

import numpy as np
import scipy.io
import json
import time
import os

from toolbox.utils import save_obj, load_obj
from toolbox.Oinfo import exhaustive_loop_zerolag
from toolbox.dOinfo import exhaustive_loop_lagged

## Load .mat dataset
# ts = scipy.io.loadmat(os.path.join('data','ts.mat'))
# ts = np.array(ts['ts'])
# ts = ts[:, :10].T # Change this to include more variables, currently including 5 or 10 variables

## Load fmri dataset processed by brainlife pipeline
from numpy import genfromtxt
import pandas as pd
df = pd.read_csv("data/timeseries.tsv.gz", compression='gzip', delimiter='\t')
df = df.loc[:, (df != 0.0).any(axis=0)]
df.to_csv('data/cleaned_timeseries.tsv', sep='\t',index=False)
ts = genfromtxt('data/cleaned_timeseries.tsv', delimiter='\t', )
ts = ts[1:,:10].T # ideally has 101 variables, 152 timepoints; but using only first 10


configFilename = "config.json"
outputDirectory = "output"
if(not os.path.exists(outputDirectory)):
		os.makedirs(outputDirectory)

with open(configFilename, "r") as fd:
		config = json.load(fd)

if("metric" in config):
	metric = config["metric"]
else:
	metric = "Oinfo"

if metric == "Oinfo":
    t = time.time()
    Odict = exhaustive_loop_zerolag(ts, config)
    elapsed = time.time() - t
    print("Elapsed time is ", elapsed, " seconds.")
    save_obj(Odict, 'Odict_Oinfo_higher_order')
    Odict_Oinfo = load_obj('Odict_Oinfo_higher_order')
    print(Odict_Oinfo)

if metric == "dOinfo":
    t = time.time()
    Odict = exhaustive_loop_lagged(ts, config)
    elapsed = time.time() - t
    print("Elapsed time is ", elapsed, " seconds.")
    save_obj(Odict, 'Odict_dOinfo_higher_order')
    Odict_dOinfo = load_obj('Odict_dOinfo_higher_order')
    print(Odict_dOinfo)

# Examples on how to see the final output dict (equivalent of struct in MATLAB)
# Odict_Oinfo[3]
# Odict_dOinfo[0][2]


# Examples on how the new combinations manager works
# from toolbox.utils import combinations_manager

# H = combinations_manager(10,5)
# for i in range(10):
#     comb = H.nextchoose()
#     num = H.combination2number(comb)
#     comb_ret = H.number2combination(num) 
#     print(comb)
#     print(num)
#     print(comb_ret)


