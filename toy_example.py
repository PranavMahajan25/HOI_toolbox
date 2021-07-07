# Author: Pranav Mahajan, 2021

import numpy as np
import scipy.io

from Oinfo import exhaustive_loop_zerolag
from dOinfo import exhaustive_loop_lagged


dataPatient = np.array([
    [0.5377,   -1.3077,   -1.3499,   -0.2050,    0.6715,    1.0347,    0.8884,    1.4384,   -0.1022,   -0.0301],
    [1.8339,   -0.4336,    3.0349,   -0.1241,   -1.2075,    0.7269,   -1.1471,    0.3252,   -0.2414,   -0.1649],
    [-2.2588,    0.3426,    0.7254,    1.4897,    0.7172,   -0.3034,   -1.0689,   -0.7549,    0.3192,    0.6277],
    [0.8622,    3.5784,   -0.0631,    1.4090,    1.6302,    0.2939,   -0.8095,    1.3703,    0.3129,    1.0933],
    [0.3188,    2.7694,    0.7147,    1.4172,    0.4889,   -0.7873,   -2.9443,   -1.7115,   -0.8649,    1.1093]
])

# print(dataPatient.shape)  

ts = scipy.io.loadmat('ts.mat')
ts = np.array(ts['ts'])
ts = ts[:, :5].T

exhaustive_loop_zerolag(ts)
# exhaustive_loop_lagged(ts)

