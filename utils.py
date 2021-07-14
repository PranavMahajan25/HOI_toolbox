import pickle
import numpy as np
from sklearn.utils import resample

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
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)