import numpy as np
import itertools
from tqdm import tqdm
from gcmi import copnorm, ent_g
from utils import bootci


def o_information_boot(X, indsample, indvar):
    # this function takes the whole X as input, and additionally the indices
    # convenient for bootstrap
    # X size is M(variables) x N (samples)
    
    # print(X.shape)
    X = X[indvar,:]
    X = X[:, indsample]

    M,N = X.shape
    o = (M-2) * ent_g(X)
    for j in range(M):
        X1 = np.delete(X, j, axis=0)
        o = o + ent_g(X[j,:]) - ent_g(X1)
    return o


def exhaustive_loop_zerolag(ts):
    print("ts.shape: ", ts.shape)
    Xfull = copnorm(ts)
    # print(Xfull)
    nvartot, N = Xfull.shape
    # print(nvartot, N)
    X = Xfull
    maxsize = 6 # max number of variables in the multiplet
    n_best = 10 # number of most informative multiplets retained
    nboot = 100 # number of bootstrap samples
    alphaval = 0.05
    o_b = np.zeros((nboot,1))

    Odict = {}

    # this section is for the expansion of redundancy, so maximizing the O
    # there's no need to fix the target here
    bar_length = (maxsize+1-3)
    with tqdm(total=bar_length) as pbar:
        pbar.set_description("Outer loops, maximizing O")
        for isize in tqdm(range(3,maxsize+1), disable=True):
            Otot = {}
            nplets_iter=itertools.combinations(range(1,nvartot+1),isize)
            nplets = []
            for nplet in nplets_iter:
                nplets.append(nplet)
            C = np.array(nplets) # n-tuples without repetition over N modules
            # print(C)
            ncomb = C.shape[0]
            # print(ncomb)
            Osize = np.zeros(ncomb)

            for icomb in tqdm(range(ncomb), desc="Inner loop, computing O", leave=False):
                Osize[icomb] = o_information_boot(X, range(N), C[icomb, :] - 1)

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
                    f = lambda xsamp: o_information_boot(X, xsamp, indvar-1)
                    ci_lower, ci_upper = bootci(nboot, f, range(N), alphaval)
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
                    f = lambda xsamp: o_information_boot(X, xsamp, indvar-1)
                    ci_lower, ci_upper = bootci(nboot, f, range(N), alphaval)
                    # print(ci_lower, ci_upper)
                    boot_sig[isel] = not(ci_lower<=0 and ci_upper > 0) # bias correction?
                Otot['sorted_syn'] = Osort_neg[0:n_sel]
                Otot['index_syn'] = ind_neg[ind_neg_sort[0:n_sel]].flatten()
                Otot['bootsig_syn'] = boot_sig
            Odict[isize] = Otot
            pbar.update(1)
            
    return Odict