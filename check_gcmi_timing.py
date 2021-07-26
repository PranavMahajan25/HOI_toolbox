import numpy as np
import time
from gcmi import copnorm, gccmi_ccc_nocopnorm, ent_g
from frites.estimator.est_gcmi import GCMIEstimator
from frites.core.gcmi_1d import ent_1d_g


N = 100000 # 817

# M = 3 to 6 variables
M = 20

X = np.random.randn(M,N) # note the difference - 
# the matlab code has it (N,M), its just the way the function expects the input
X = copnorm(X)

t = time.time()
entropy = ent_1d_g(X)
elapsed = time.time() - t
print("Frites - Elapsed time is ", elapsed, " seconds.")
print(entropy)

t = time.time()
entropy = ent_g(X)
elapsed = time.time() - t
print("Rob Ince - Elapsed time is ", elapsed, " seconds.")
print(entropy)


# N = 6530

# frites_estimator = GCMIEstimator(mi_type='ccc', copnorm=False)

# A = np.random.randn(1,N)
# B = np.random.randn(6,N)
# C = np.random.randn(3,N)

# A = copnorm(A)
# B = copnorm(B)
# C = copnorm(C)

# t = time.time()
# Af = np.expand_dims(A,axis=0)
# Bf = np.expand_dims(B,axis=0)
# Cf = np.expand_dims(C,axis=0)
# mi = frites_estimator.estimate(Af,Bf,Cf)
# elapsed = time.time() - t
# print("Frites - Elapsed time is ", elapsed, " seconds.")
# print(mi[0][0])

# t = time.time()
# mi = gccmi_ccc_nocopnorm(A,B,C)
# elapsed = time.time() - t
# print("Rob Ince - Elapsed time is ", elapsed, " seconds.")
# print(mi)