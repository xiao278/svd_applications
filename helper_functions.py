import numpy as np

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