import numpy as np, pandas as pd
import scipy.optimize as opt
from sklearn.decomposition import NMF, PCA, FastICA
    
def censored_nnlstsq(A, B, M, maxiter = -1):
    """Solves least squares problem subject to missing data.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # Solve via tensor representation
    rhs = np.dot(A.T, M * B) # r x n x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    
    res = np.random.rand(A.shape[1], B.shape[1])
    
    if maxiter == -1:
        maxiter = 3 * A.shape[1]
    
    for j in range(rhs.shape[1]):
        r = opt.nnls(T[j,:,:], rhs[:,j], maxiter = maxiter)
        res[:,j] = r[0]
    
    return res

def cv_nmf(data, rank, n_iter = 200, p_holdout=.1):
    """Fit NMF while holding out a fraction of the dataset.
    """

    # create masking matrix
    M = np.random.choice([0, 1], size = data.shape, p = [p_holdout, 1 - p_holdout])
    
    # Initialize the factor matrices
    W = np.random.randn(data.shape[0], rank)
    H = np.random.randn(rank, data.shape[1])
    
    for itr in range(n_iter):
        H = censored_nnlstsq(W, data, M)
        W = censored_nnlstsq(H.T, data.T, M.T).T

    # return result and test/train error
    mask = np.nonzero(M)
    inverted_mask = np.nonzero(1 - M)
    resid = np.dot(W, H) - data
    
    train_err = np.mean(resid[mask]**2) 
    test_err = np.mean(resid[inverted_mask]**2)
    net_err = np.mean(resid**2)
    
    return train_err, test_err, net_err, M, W, H
