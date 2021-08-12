import sys
import numpy as np
import itertools
from tqdm.auto import tqdm
from toolbox.gcmi import copnorm, ent_g
from toolbox.lin_est import lin_ent
from toolbox.utils import bootci, combinations_manager, ncr


def get_ent(X, estimator):
    if estimator=='lin_est':
        entropy = lin_ent(X)
    elif estimator=='gcmi':
        entropy = ent_g(X)
    else:
        print("Please use estimator out of the following - 'lin_est' or 'gcmi'")
        sys.exit()
    return entropy

def o_information_boot(X, indsample, indvar, estimator):
    # this function takes the whole X as input, and additionally the indices
    # convenient for bootstrap
    # X size is M(variables) x N (samples)
    
    # print(X.shape)
    X = X[indvar,:]
    X = X[:, indsample]

    M,N = X.shape
    o = (M-2) * get_ent(X, estimator)
    
    for j in range(M):
        X1 = np.delete(X, j, axis=0)
        o = o + get_ent(X[j,:], estimator) - get_ent(X1, estimator)
    return o


def exhaustive_loop_zerolag(ts, config):
    higher_order = config["higher_order"]
    estimator = config["estimator"]
    Xfull = copnorm(ts)
    nvartot, N = Xfull.shape
    print("Timeseries details - Number of variables: ", str(nvartot),", Number of timepoints: ", str(N))
    print("Computing Oinfo using " + estimator + " estimator")
    X = Xfull
    maxsize = config["maxsize"] #5 # max number of variables in the multiplet
    n_best = config["n_best"] #10 # number of most informative multiplets retained
    nboot = config["nboot"] #100 # number of bootstrap samples
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
            if higher_order:
                H = combinations_manager(nvartot, isize)
                ncomb = ncr(nvartot, isize)
                O_pos = np.zeros(n_best)
                O_neg = np.zeros(n_best)
                ind_pos = np.zeros(n_best)
                ind_neg = np.zeros(n_best)
            else:
                nplets_iter=itertools.combinations(range(1,nvartot+1),isize)
                nplets = []
                for nplet in nplets_iter:
                    nplets.append(nplet)
                C = np.array(nplets) # n-tuples without repetition over N modules
                ncomb = C.shape[0]
            Osize = np.zeros(ncomb)

            for icomb in tqdm(range(ncomb), desc="Inner loop, computing O", leave=False):
                if higher_order:
                    comb = H.nextchoose()
                    Osize= o_information_boot(X, range(N), comb - 1, estimator)
                    valpos, ipos = np.min(O_pos), np.argmin(O_pos)
                    valneg, ineg = np.max(O_neg), np.argmax(O_neg)
                    if Osize>0 and Osize>valpos:
                        O_pos[ipos] = Osize
                        ind_pos[ipos] = H.combination2number(comb)
                    if Osize<0 and Osize<valneg:
                        O_neg[ineg] = Osize
                        ind_neg[ineg] = H.combination2number(comb)
                else:
                    comb = C[icomb, :]
                    Osize[icomb] = o_information_boot(X, range(N), comb - 1, estimator)   

            if higher_order:
                Osort_pos , ind_pos_sort = np.sort(O_pos)[::-1], np.argsort(O_pos)[::-1]
                Osort_neg , ind_neg_sort = np.sort(O_neg), np.argsort(O_neg)
            else:
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
                    if higher_order:
                        indvar=H.number2combination(ind_pos[ind_pos_sort[isel]])
                    else:
                        indvar=np.squeeze(C[ind_pos[ind_pos_sort[isel]],:])
                    f = lambda xsamp: o_information_boot(X, xsamp, indvar-1, estimator)
                    ci_lower, ci_upper = bootci(nboot, f, range(N), alphaval)
                    boot_sig[isel] = not(ci_lower<=0 and ci_upper > 0) 
                Otot['sorted_red'] = Osort_pos[0:n_sel]
                Otot['index_red'] = ind_pos[ind_pos_sort[0:n_sel]].flatten()
                Otot['bootsig_red'] = boot_sig
            if Osort_neg.size != 0:
                n_sel = min(n_best, len(Osort_neg))
                boot_sig = np.zeros((n_sel, 1))
                for isel in range(n_sel):
                    if higher_order:
                        indvar=H.number2combination(ind_neg[ind_neg_sort[isel]])
                    else:
                        indvar=np.squeeze(C[ind_neg[ind_neg_sort[isel]],:])
                    f = lambda xsamp: o_information_boot(X, xsamp, indvar-1, estimator)
                    ci_lower, ci_upper = bootci(nboot, f, range(N), alphaval)
                    boot_sig[isel] = not(ci_lower<=0 and ci_upper > 0) 
                Otot['sorted_syn'] = Osort_neg[0:n_sel]
                Otot['index_syn'] = ind_neg[ind_neg_sort[0:n_sel]].flatten()
                Otot['bootsig_syn'] = boot_sig
            Odict[isize] = Otot
            pbar.update(1)
            
    return Odict
