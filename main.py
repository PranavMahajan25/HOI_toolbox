# Author: Pranav Mahajan, 2021

import numpy as np
from numpy import genfromtxt
import scipy.io
import pandas as pd
import time
import json
import sys
import os

from toolbox.utils import save_obj, load_obj
from toolbox.Oinfo import exhaustive_loop_zerolag
from toolbox.dOinfo import exhaustive_loop_lagged


configFilename = "config.json"
argCount = len(sys.argv)
if(argCount > 1):
		configFilename = sys.argv[1]

outputDirectory = "output"
if(not os.path.exists(outputDirectory)):
		os.makedirs(outputDirectory)

with open(configFilename, "r") as fd:
		config = json.load(fd)

if("metric" in config):
	metric = config["metric"]
else:
	metric = "Oinfo"

if("input" in config):
	timeseriesFilename = config["input"]
else:
	print("Please provide input data location in config")
	sys.exit()

if("input_type" in config):
	input_type = config["input_type"]
else:
	input_type = "tsv"


if input_type=="tsv":
	df = pd.read_csv(timeseriesFilename, compression='gzip', delimiter='\t')
	df = df.loc[:, (df != 0.0).any(axis=0)]
	df.to_csv('data/cleaned_timeseries.tsv', sep='\t', index=False)
	ts = genfromtxt('data/cleaned_timeseries.tsv', delimiter='\t', )
	ts = ts[1:,:].T # 101 variables, 152 timepoints
	# print(ts.shape)
elif input_type=="mat":
	ts = scipy.io.loadmat(timeseriesFilename)
	ts = np.array(ts['ts'])
	ts = ts.T
	# print(ts.shape)
else:
	print("Unknown input type")
	sys.exit()

if metric == "Oinfo":
    t = time.time()
    Odict = exhaustive_loop_zerolag(ts, config)
    elapsed = time.time() - t
    print("Elapsed time is ", elapsed, " seconds.")
    save_obj(Odict, 'Odict_Oinfo')
    Odict_Oinfo = load_obj('Odict_Oinfo')
    print(Odict_Oinfo)

if metric == "dOinfo":
    t = time.time()
    Odict = exhaustive_loop_lagged(ts, config)
    elapsed = time.time() - t
    print("Elapsed time is ", elapsed, " seconds.")
    save_obj(Odict, 'Odict_dOinfo')
    Odict_dOinfo = load_obj('Odict_dOinfo')
    print(Odict_dOinfo)
