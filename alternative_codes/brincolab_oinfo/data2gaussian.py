# Original code in MATLAB (https://github.com/brincolab/High-Order-interactions)
# Translated to Python by Pranav Mahajan, 2021

# Transforms 'data' (T samples x N dimensionmatrix) to Gaussian with 0 mean and 1 sd
# using empirical copulas
# 
# INPUT
# data = T samples x N variables matrix
#
# OUTPUT
# gaussian_data = T samples x N variables matrix with the gaussian copula
# transformed data
# covmat = N x N covariance matrix of gaussian copula transformed data.
# Author: Rubén Herzog.

import numpy as np
from scipy.stats import norm

def data2gaussian(data):
    T = data.shape[0]
    sortid = np.argsort(data, axis=0) # sort data and keep sorting indexes
    copdata = np.argsort(sortid, axis=0) # sorting sorting indexes #but is indexed from 0 !

    copdata = copdata + np.ones(copdata.shape) # to make it like matlab, indexed from 1 ****
    copdata = copdata/(T+1) #can T work? # normalization to have data in [0,1]


    gaussian_data = norm.ppf(copdata)
    gaussian_data[np.isinf(gaussian_data)] = 0
    # print("gaussian_data", gaussian_data)   
    covmat = np.matmul(gaussian_data.T, gaussian_data)/(T-1)
    # print("covmat", covmat)   
    return gaussian_data, covmat