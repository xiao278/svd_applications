import numpy as np
from scipy.stats import norm

def sum_from_diag(A:np.ndarray):
    '''Returns a matrix where each (i,j) is the sum of all cells of the **square** input matrix between and including (i,j) and (j,i)'''
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    # base case d=0
    S = np.zeros(A.shape)
    np.fill_diagonal(S, np.diagonal(A))

    for d in range(1, n):
        for k in range(0,n-d):
            r = k
            c = d + k
            total = None
            if (d == 1): # base case d=1
                total = A[r+1,c] + A[r,c-1] + A[r,c] + A[c,r]
            else:
                total = S[r+1,c] + S[r,c-1] - S[r+1,c-1] + A[r,c] + A[c,r]
            S[r,c] = total
            S[c,r] = total
    return S

def mean_from_diag(A:np.ndarray):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    M = np.zeros(A.shape)
    np.fill_diagonal(M, 1)
    for r in range(n):
        for c in range(r+1,n):
            size = (c - r + 1) ** 2
            M[r,c] = size
            M[c,r] = size
    S = sum_from_diag(A)
    return S / M

def calc_from_diag(mat:np.ndarray, function):
    '''Returns a matrix where each (i,j) is the specified function run on all cells of the input matrix between and including (i,j) and (j,i). WARNING: SLOW'''
    n = mat.shape[0]
    assert mat.shape[0] == mat.shape[1]
    calc_mat = np.zeros((n,n))
    for r in range(0,n):
        for c in range(r,n):
            lo_diag = min(r,c)
            hi_diag = max(r,c) + 1
            temp = function(mat[lo_diag:hi_diag,lo_diag:hi_diag])
            calc_mat[r,c] = temp
            calc_mat[c,r] = temp
    return calc_mat

def calc_cutoff_likelihoods(svs:list[float]):
    svs = np.array(svs)
    p = svs.shape[0] # how many total singular values are we dealing with
    samp_means = np.zeros((p + 1, 2))
    samp_vars = np.zeros((p + 1, 2))
    # calculate sample mean and variance for pre and post cutoff segments assuming q is the cutoff
    for q in range(1, p + 1):
        samp_means[q,0] = np.mean(svs[0:q])
        samp_means[q,1] = np.mean(svs[q:p])

        samp_vars[q,0] = np.var(svs[0:q])
        samp_vars[q,1] = np.var(svs[q:p])

    def total_log_likelihood(x:np.ndarray, mean:float, std:float):
        return np.sum(
            np.log(
                norm.pdf(x, loc=mean, scale=std)
            )
        )

    LL = np.zeros((p,))
    # calculate log likelihood
    for q in range(1, p):
        combined_var = (
            (q - 1) * samp_vars[q,0] + (p - q - 1) * samp_vars[q,1]
        ) / (p - 2)
        combined_std = np.sqrt(combined_var)
        LL[q-1] = (
            total_log_likelihood(svs[0:q], mean=samp_means[q,0], std=combined_std)
            +
            total_log_likelihood(svs[q:p], mean=samp_means[q,1], std=combined_std)
        )
    LL[p-1] = total_log_likelihood(svs, mean=samp_means[p-1,0], std=np.sqrt(samp_vars[p-1,0]))

    return LL

# likelihoods_test = np.array([
#     10, 9, 3, 2, 1
# ])
# should be [-12.27178365  -5.36541382 -11.51601249 -12.90901428 -13.90901428]