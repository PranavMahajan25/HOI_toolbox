import numpy as np
import time
from toolbox.gcmi import copnorm, gccmi_ccc_nocopnorm, ent_g


N = 10000 # 817

# M = 3 to 6 variables
M = 10

X = np.random.randn(M,N) # note the difference - 
# the matlab code has it (N,M), its just the way the function expects the input
t = time.time()
X = copnorm(X)
entropy = ent_g(X)
elapsed = time.time() - t
print("Elapsed time is ", elapsed, " seconds.")
print(entropy)


N = 653

# A = np.random.randn(1,N)
# B = np.random.randn(6,N)
# C = np.random.randn(3,N)
# t = time.time()
# A = copnorm(A)
# B = copnorm(B)
# C = copnorm(C)
# mi = gccmi_ccc_nocopnorm(A,B,C)
# elapsed = time.time() - t
# print("Elapsed time is ", elapsed, " seconds.")
# print(mi)