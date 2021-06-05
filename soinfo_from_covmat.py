# Original code in MATLAB (https://github.com/brincolab/High-Order-interactions)
# Translated to Python by Pranav Mahajan, 2021


# author = RubÃ©n Herzog, rubenherzog@postgrado.uv.cl, modify by Marilyn Gatica, marilyn.gatica@postgrado.uv.cl
# Computes the O-Information and S-Information of gaussian data given their covariance
# matrix 'covmat'.
#
# INPUTS
# covmat = N x N covariance matrix 
# T = length data
#
# OUTPUT
# oinfo (O-Information) 
# sinfo (S-Information) of the system with covariance matrix covmat.

import numpy as np
from scipy.special import psi as psi

def ent_fun(x,y):
    # function to compute the entropy of multivariate gaussian distribution, where x is dimensionsionality
    # and y is the variables variance of the covariance matrix determinant.
    return 0.5*np.log(((2*np.pi*np.exp(1))**(x))*y)

def gaussian_ent_biascorr(N, T):
    psiterms = psi((T - (np.arange(1,N+1)))/2)
    biascorr =  0.5*(N*np.log(2/(T-1)) + sum(psiterms))
    return biascorr

def reduce_x(x, covmat, N):
    covmat = covmat[np.arange(1,N+1) != x, :]
    covmat = covmat[:, np.arange(1,N+1) != x]
    return covmat

def soinfo_from_covmat(covmat, T):
    N = len(covmat)
    emp_det = np.linalg.det(covmat) # determinant
    single_vars = np.diag(covmat); # variance of single variables

    # bias corrector for N,(N-1) and one gaussian variables
    biascorrN = gaussian_ent_biascorr(N,T)
    biascorrNmin1 = gaussian_ent_biascorr(N-1,T)
    biascorr_1 = gaussian_ent_biascorr(1,T)

    tc = np.sum(ent_fun(1,single_vars)-biascorr_1) - (ent_fun(N,emp_det)-biascorrN) # tc=Total Correlation
    print(tc) 

    Hred = 0
    for red in range(1,N+1):
        Hred = Hred + ent_fun((N-1), np.linalg.det(reduce_x(red, covmat, N))) - biascorrNmin1
    dtc = Hred - (N-1)*(ent_fun(N,emp_det) - biascorrN) # dtc = Dual Total Correlation
    print(dtc)

    oinfo = tc - dtc
    sinfo = tc + dtc

    return oinfo, sinfo