import numpy as np
from sklearn.utils import resample

def bootci(nboot, oinfo_func, xsamp_range, alpha):
    samplesize = int(len(xsamp_range)//2)
    stats = list()
    for i in range(nboot):
        xsamp = resample(xsamp_range, n_samples=samplesize)
        xsamp = np.sort(xsamp)
        oinfo = oinfo_func(xsamp)
        stats.append(oinfo)
    # confidence intervals
    p = ((alpha)/2.0) * 100
    lower = np.percentile(stats, p)
    p = (1-(alpha)/2.0) * 100
    upper = np.percentile(stats, p)
    return lower, upper