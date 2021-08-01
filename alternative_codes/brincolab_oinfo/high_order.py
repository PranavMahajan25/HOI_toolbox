# Original code in MATLAB (https://github.com/brincolab/High-Order-interactions)
# Translated to Python by Pranav Mahajan, 2021

# Code description: Function to compute S-Information, O-Information, and characterize the High-order interactions among n variables governed by Redundancy or Synergy.
# Input: - 'data': Matrix with dimensionality (N,T), where N is the number of brain regions or modules, and T is the number of samples.
#        - 'n': number of interactions or n-plets. n must be greater or equal, and if n=3, then the interactions is among triplets. 
# Output: - 'Red': Matrix with dimension (Npatients,Modules), with the redundancy values per patient and per module
#         - 'Syn': Matrix with dimension (Npatients,Modules), with the synergy values per patient and per module
#         - 'Oinfo': O-Information for all the n-plets.
#         - 'Sinfo': S-Information for all the n-plets
# @author: Marilyn Gatica, marilyn.gatica@postgrado.uv.cl
# Acknowledgment: data2gaussian(), soinfo_from_covmat() created by Ruben
# Herzog, ruben.herzog@postgrado.uv.cl; modify by Marilyn Gatica.

import numpy as np
import itertools

from data2gaussian import data2gaussian
from soinfo_from_covmat import soinfo_from_covmat

def high_order(data, n):
    Modules = data.shape[0]
    Red = np.zeros((1, Modules)) # matrix to save the redundant values, per patient and per module
    Syn = np.zeros((1, Modules)) # matrix to save the synergistic values, per patient and per module
    nplets_iter=itertools.combinations(range(1,Modules+1),n)
    nplets = []
    for nplet in nplets_iter:
        nplets.append(nplet)
    nplets = np.array(nplets) # n-tuples without repetition over N modules
    Oinfo=np.zeros((nplets.shape[0],1)) # vector Oinfo to save the O-Information value for each n-tuple
    Sinfo=np.zeros((nplets.shape[0],1)) # vector Sinfo to save the S-Information value for each n-tuple
    i=0
    data_mean = data.mean(axis=1)
    data_mean = data_mean[:, np.newaxis]
    dataNorm = data - data_mean # normalize the time series (per module: BOLD signal- mean(BOLD signal))
    _, est_covmat = data2gaussian(dataNorm.T) # Transformation to Copulas and Covariance Matrix Estimation
    print(est_covmat)
    for nplet in nplets: # moving in each interaction of n-tuples: npletIndex
        npletIndex = nplet - 1
        print(npletIndex)
        sub_covmat = est_covmat[npletIndex, :]
        sub_covmat = sub_covmat[:, npletIndex] # create a sub covariance matrix with only the particular values npletIndex
        # print(sub_covmat)
        est_oinfo, est_sinfo = soinfo_from_covmat(sub_covmat, dataNorm.shape[1]) # Estimating O-Information and S-Info with bias
        Oinfo[i] = est_oinfo # O-Information for the n-plet with index 'i'
        Sinfo[i] = est_sinfo # S-Information for the n-plet with index 'i'
        i+=1
    
    print(nplets)
    for module in range(1, Modules+1):
        print("module ", module)
        modRow, _ = np.where(nplets == module) # find the interactions where the module 'module' belong
        Oinfo_module=Oinfo[modRow] # compute the Oinfo per module
        Red[0, module-1] = np.mean(Oinfo_module[Oinfo_module>0]) # if Oinfo is positive, the interaction is governed by redundancy
        Syn[0, module-1] = np.mean(np.abs(Oinfo_module[Oinfo_module<0])) # if Oinfo is negative, the interaction is governed by synergy 

    Syn[np.isnan(Syn)]=0

    return Oinfo, Sinfo, Red, Syn