import os
import sys
import pickle
import numpy as np
from sklearn.utils import resample
import operator as op
from functools import reduce

def bootci(nboot, oinfo_func, xsamp_range, alpha):
    stats = list()
    for i in range(nboot):
        xsamp = resample(xsamp_range, n_samples=len(xsamp_range))
        oinfo = oinfo_func(xsamp)
        stats.append(oinfo)
    # confidence intervals
    p = ((alpha)/2.0) * 100
    lower = np.percentile(stats, p)
    p = (1-(alpha)/2.0) * 100
    upper = np.percentile(stats, p)
    return lower, upper


def save_obj(obj, name):
    if not os.path.exists('output'):
        os.makedirs('output')
    with open('output/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('output/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def ncr(n, r):
    if n<r:
        return 0
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  

class combinations_manager:
    def __init__(self, N, K):
        if K>N:
            print("error: K can't be greater than N in N choose K")
            sys.exit()
        self.N = N
        self.K = K
        self.lim = K
        self.inc = 1
        if K > N/2:
            WV = N-K
        else:
            WV = K
        
        self.BC = ncr(N,WV) - 1
        self.CNT = 0
        self.WV = []

    def nextchoose(self):
        if self.CNT == 0 or self.K == self.N:
            self.WV = np.arange(1, self.K+1)
            self.B = self.WV
            self.CNT += 1
            return self.B
        
        if self.CNT == self.BC:
            self.B =  np.arange(self.N-self.K+1, self.N+1)
            self.CNT = 0
            self.inc = 1
            self.lim = self.K
            return self.B

        for jj in range(self.inc):
            self.WV[self.K + jj - self.inc] = self.lim + jj + 1 
        
        if self.lim < (self.N-self.inc):
            self.inc = 0

        self.inc += 1
        self.lim = self.WV[self.K-self.inc]
        self.CNT += 1
        self.B = self.WV
        return self.B

    def combination2number(self, comb):
        num = 0
        k = len(comb)
        for i in range(1, k+1):
            c = comb[i-1] - 1
            num += ncr(c,i)
        return num

    def number2combination(self, num):
        comb = []
        k = self.K
        num_red = num
        while k>0:
            m = k-1
            while True:
                mCk = ncr(m,k) 
                if mCk > num_red:
                    break
                if comb.count(m) > 0:
                    break
                m+=1
            comb.append(m)
            num_red -= ncr(m-1, k)
            k -=1
        comb.reverse()
        comb = np.array(comb)
        return comb
