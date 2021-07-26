# Author: Pranav Mahajan, 2021

import numpy as np
import scipy.io
import time

from utils import save_obj, load_obj
from Oinfo import exhaustive_loop_zerolag
from dOinfo import exhaustive_loop_lagged


ts = scipy.io.loadmat('ts.mat')
ts = np.array(ts['ts'])
ts = ts[:, :10].T # Change this to include more variables, currently including 5 or 10 variables
# ts = ts.T #This includes all the variables in the data

metric = 'dOinfo'

if metric == 'Oinfo':
    t = time.time()
    Odict = exhaustive_loop_zerolag(ts, frites=True)
    elapsed = time.time() - t
    print("Elapsed time is ", elapsed, " seconds.")
    save_obj(Odict, 'Odict_Oinfo')
    Odict_Oinfo = load_obj('Odict_Oinfo')
    print(Odict_Oinfo)

if metric == 'dOinfo':
    t = time.time()
    Odict = exhaustive_loop_lagged(ts, frites=True)
    elapsed = time.time() - t
    print("Elapsed time is ", elapsed, " seconds.")
    save_obj(Odict, 'Odict_dOinfo')
    Odict_dOinfo = load_obj('Odict_dOinfo')
    print(Odict_dOinfo)

# Examples on how to see the final output dict (equivalent of struct in MATLAB)
# Odict_Oinfo[3]
# Odict_dOinfo[0][2]


# Examples on how the new combinations manager works
# from utils import combinations_manager

# H = combinations_manager(10,5)
# for i in range(10):
#     comb = H.nextchoose()
#     num = H.combination2number(comb)
#     comb_ret = H.number2combination(num) 
#     print(comb)
#     print(num)
#     print(comb_ret)


