import numpy as np
import itertools
from gcmi import copnorm, ent_g

def o_information_boot(X, indsample, indvar):
    # this function takes the whole X as input, and additionally the indices
    # convenient for bootstrap
    # X size is M(variables) x N (samples)
    print(X.shape)
    X = X[indvar,:]
    X = X[:, indsample]

    M,N = X.shape
    o = (M-2) * ent_g(X)
    for j in range(M):
        X1 = np.delete(X, j, axis=0)
        o = o + ent_g(X[j,:]) - ent_g(X1)
    return o


def exhaustive_loop_zerolag(ts):
    print(ts.shape)
    Xfull = copnorm(ts)
    print(Xfull)
    nvartot, N = Xfull.shape
    print(nvartot, N)
    X = Xfull
    maxsize = 3 # max number of variables in the multiplet
    n_best = 10 # number of most informative multiplets retained
    nboot = 100 # number of bootstrap samples
    alphaval = 0.05
    o_b = np.zeros((nboot,1))

    for isize in range(3,maxsize+1):
        nplets_iter=itertools.combinations(range(1,nvartot+1),isize)
        nplets = []
        for nplet in nplets_iter:
            nplets.append(nplet)
        C = np.array(nplets) # n-tuples without repetition over N modules
        print(C)
        ncomb = C.shape[0]
        print(ncomb)
        Osize = np.zeros(ncomb)

        for icomb in range(ncomb):
            Osize[icomb] = o_information_boot(X, range(N), C[icomb, :] - 1)

        ind_pos = np.argwhere(Osize>0)
        ind_neg = np.argwhere(Osize<0)
        O_pos = Osize[Osize>0]
        O_neg = Osize[Osize<0]
        Osort_pos , ind_pos_sort = np.sort(O_pos), np.argsort(O_pos)
        Osort_neg , ind_neg_sort = np.sort(O_neg), np.argsort(O_neg)
        
        if Osort_pos.size != 0:
            n_sel = min(n_best, len(Osort_pos))
            sorted_red = Osort_pos[::-1][0:n_sel]
            index_red = ind_pos[ind_pos_sort[::-1][0:n_sel]].flatten()
        if Osort_neg.size != 0:
            n_sel = min(n_best, len(Osort_neg))
            sorted_syn = Osort_neg[0:n_sel]
            index_syn = ind_neg[ind_neg_sort[0:n_sel]].flatten()

        print(Osize, sorted_red, index_red, sorted_syn, index_syn)

    return Osize, sorted_red, index_red, sorted_syn, index_syn