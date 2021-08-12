import numpy as np
import scipy.io
import os
import time
from toolbox.gcmi import copnorm, gccmi_ccc_nocopnorm, ent_g
from toolbox.lin_est import lin_ent, lin_cmi_ccc


## Check timing for entropy computation

N = 10000 # 817

# M = 3 to 6 variables
M = 100

X = np.random.randn(M,N) # note the difference - 
# the matlab code has it (N,M), its just the way the function expects the input

# ts = scipy.io.loadmat(os.path.join('data','ts.mat'))
# ts = np.array(ts['ts'])
# X = ts[:,:].T

t = time.time()
e = lin_ent(X)
elapsed = time.time() - t
print("Covariance matrix based: Elapsed time is ", elapsed, " seconds.")
print(e)

t = time.time()
X = copnorm(X)
entropy = ent_g(X)
elapsed = time.time() - t
print("GCMI based: Elapsed time is ", elapsed, " seconds.")
print(entropy)



#####################
## Check timing for conditional mutual information computation

# N = 653

# A = np.random.randn(1,N)
# B = np.random.randn(6,N)
# C = np.random.randn(3,N)

# t = time.time()
# mi = lin_cmi_ccc(A.T, B.T, C.T)
# elapsed = time.time() - t
# print("Covariance matrix based: Elapsed time is ", elapsed, " seconds.")
# print(mi)

# t = time.time()
# A = copnorm(A)
# B = copnorm(B)
# C = copnorm(C)
# mi = gccmi_ccc_nocopnorm(A,B,C)
# elapsed = time.time() - t
# print("GCMI based: Elapsed time is ", elapsed, " seconds.")
# print(mi)