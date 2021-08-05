import numpy as np

def lin_ent(X):
    # X is of shape (num var, num timepoints)
    covX = np.cov(X)
    if covX.ndim == 0:
        covX = np.var(X)
        det_covX = covX
        N = 1
    else:
        det_covX = np.linalg.det(covX) 
        N = X.shape[0]
    e=0.5*np.log(det_covX) + 0.5*N*np.log(2*np.pi*np.exp(1))    
    return e

def lin_CE(Yb, Z):
    # Yb (output), Z (input) are of shape (num timepoints, num variables)
    Am = np.linalg.lstsq(Z, Yb, rcond=None)[0]
    Yp = Z@Am
    Up = Yb - Yp
    S = np.cov(Up.T)
    if S.ndim == 0:
        S = np.var(Up.T)
        detS = S
    else:
        detS = np.linalg.det(S)
    N = Yb.shape[1]
    ce = 0.5*np.log(detS)+0.5*N*np.log(2*np.pi*np.exp(1))
    return ce

def lin_cmi_ccc(Y, X0, Y0):
    H_Y_Y0 = lin_CE(Y, Y0)
    X0Y0 = np.concatenate((X0, Y0), axis=1)
    H_Y_X0Y0 = lin_CE(Y, X0Y0)
    cmi = H_Y_Y0 - H_Y_X0Y0
    # print(cmi, H_Y_Y0, H_Y_X0Y0)
    return cmi