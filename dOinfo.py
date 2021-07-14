import numpy as np
import itertools
from tqdm import tqdm
from gcmi import copnorm, gccmi_ccc_nocopnorm
from utils import bootci

def o_information_lagged_boot(Y,X,m,indstart,chunklength,indvar):
    # evaluates the o_information flow
    # Y Nx1 target vector .  X NxM drivers
    # m order of the model

    if chunklength == 0:
        indsample = np.arange(len(Y))
    else:
        nchunks = int(np.floor(len(Y)/chunklength))
        indstart = indstart[0:nchunks]
        indsample = np.zeros(nchunks*chunklength)
        for istart in range(nchunks):
            indsample[(istart)*chunklength:(istart+1)*chunklength] = np.arange(indstart[istart],indstart[istart]+chunklength)
            
    indsample = indsample.astype('int32')
    # print(indsample)
    # print(indvar)
    # print(X.shape, Y.shape, indsample.shape)
    
    Y = Y[indsample]
    X = X[indsample, :]
    X = X[:, indvar]
    N, M = X.shape
    
    n = N - m
    X0 = np.zeros((n,m,M))
    Y0 = np.zeros((n,m))
    y = np.array([Y[m:]])
    for i in range(n):
        for j in range(m):
            Y0[i,j] = Y[m-j+i-1]
            for k in range(M):
                X0[i,j,k] = X[m-j+i-1, k]

    X0_reshaped = np.reshape(np.ravel(X0, order='F'), (n, m*M), order='F').T 
    Y0 = Y0.T
    # print(X0_reshaped.shape)
    # print(y.shape)
    # print(Y0.shape)
    o = - (M-1) * gccmi_ccc_nocopnorm(y, X0_reshaped, Y0)
    
    for k in range(M):
        X = np.delete(X0, k, axis=2)
        X_reshaped = np.reshape(np.ravel(X, order='F'), (n, m*(M-1)), order='F').T
        o=o+gccmi_ccc_nocopnorm(y, X_reshaped, Y0)
    # print(o)
    return o


def exhaustive_loop_lagged(ts):
    print("ts.shape: ", ts.shape)
    Xfull = copnorm(ts)
    # print(Xfull)
    nvartot, N = Xfull.shape
    # print(nvartot, N)
    X = Xfull.T
    modelorder = 3 # check this
    maxsize = 6 # max number of variables in the multiplet
    n_best = 10 # number of most informative multiplets retained
    nboot = 100 # number of bootstrap samples
    chunklength = round(N/5); #can play around with this
    alphaval = 0.05
    o_b = np.zeros((nboot,1))

    # this section is for the expansion of redundancy, so maximizing the O

    Odict = {}
    bar_length = nvartot*(maxsize+1-2)
    with tqdm(total=bar_length) as pbar:
        pbar.set_description("Outer loops, maximizing O")
        for itarget in tqdm(range(nvartot), disable=True):
            Otarget = {} 
            t = X[:, itarget]
            for isize in tqdm(range(2,maxsize+1), disable=True):
                Otot = {}
                var_arr = np.setdiff1d(np.arange(1,nvartot+1), itarget+1)
                nplets_iter=itertools.combinations(var_arr,isize)
                nplets = []
                for nplet in nplets_iter:
                    nplets.append(nplet)
                C = np.array(nplets) # n-tuples without repetition over N modules
                # print(C)
                ncomb = C.shape[0]
                # print(ncomb)
                Osize = np.zeros(ncomb)

                for icomb in tqdm(range(ncomb), desc="Inner loop, computing O", leave=False):
                    Osize[icomb] = o_information_lagged_boot(t, X, modelorder, np.arange(N), 0, C[icomb, :] - 1)

                ind_pos = np.argwhere(Osize>0)
                ind_neg = np.argwhere(Osize<0)
                O_pos = Osize[Osize>0]
                O_neg = Osize[Osize<0]
                Osort_pos , ind_pos_sort = np.sort(O_pos)[::-1], np.argsort(O_pos)[::-1]
                Osort_neg , ind_neg_sort = np.sort(O_neg), np.argsort(O_neg)
                
                if Osort_pos.size != 0:
                    n_sel = min(n_best, len(Osort_pos))
                    boot_sig = np.zeros((n_sel, 1))
                    for isel in range(n_sel):
                        indvar=np.squeeze(C[ind_pos[ind_pos_sort[isel]],:])
                        # print(indvar)
                        f = lambda xsamp: o_information_lagged_boot(t, X, modelorder, xsamp, chunklength, indvar - 1)
                        ci_lower, ci_upper = bootci(nboot, f, np.arange(N-chunklength+1), alphaval)
                        # print(ci_lower, ci_upper)
                        boot_sig[isel] = not(ci_lower<=0 and ci_upper > 0) # bias correction?
                    Otot['sorted_red'] = Osort_pos[0:n_sel]
                    Otot['index_red'] = ind_pos[ind_pos_sort[0:n_sel]].flatten()
                    Otot['bootsig_red'] = boot_sig
                if Osort_neg.size != 0:
                    n_sel = min(n_best, len(Osort_neg))
                    boot_sig = np.zeros((n_sel, 1))
                    for isel in range(n_sel):
                        indvar=np.squeeze(C[ind_neg[ind_neg_sort[isel]],:])
                        # print(indvar)
                        f = lambda xsamp: o_information_lagged_boot(t, X, modelorder, xsamp, chunklength, indvar - 1)
                        ci_lower, ci_upper = bootci(nboot, f, np.arange(N-chunklength+1), alphaval)
                        # print(ci_lower, ci_upper)
                        boot_sig[isel] = not(ci_lower<=0 and ci_upper > 0) # bias correction?
                    Otot['sorted_syn'] = Osort_neg[0:n_sel]
                    Otot['index_syn'] = ind_neg[ind_neg_sort[0:n_sel]].flatten()
                    Otot['bootsig_syn'] = boot_sig
                Otarget[isize] = Otot
                pbar.update(1)
            Odict[itarget] = Otarget
    return Odict