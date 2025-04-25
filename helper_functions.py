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

def size_from_diag(A:np.ndarray):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    N = np.zeros(A.shape)
    np.fill_diagonal(N, 1)
    for r in range(n):
        for c in range(r+1,n):
            size = (c - r + 1) ** 2
            N[r,c] = size
            N[c,r] = size
    return N

def mean_from_diag(A:np.ndarray):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    S = sum_from_diag(A)
    N = size_from_diag(A)
    return S / N

def var_from_diag(A:np.ndarray, diag_zeroes=False):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    sum_squared = sum_from_diag(np.square(A))
    S = sum_from_diag(A)
    N = size_from_diag(A)
    np.fill_diagonal(N, 2)
    M = S / N
    V = (
        (sum_squared) - (N * np.square(M))
    ) / (N - 1)
    if not diag_zeroes:
        np.fill_diagonal(V,np.diag(V,k=1) / 2)
    else:
        np.fill_diagonal(V,0)
    V = np.clip(V, a_min=0, a_max=None)
    return V

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
        samp_vars[q,0] = np.var(svs[0:q])

        if q < p:
            samp_means[q,1] = np.mean(svs[q:p])
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

def calc_cutoff_index(svs:list[float]):
    distinct_sv_idx = [] # multiple singular value are the same this will only include the index of the last one
    def is_same(a,b,tol=1e-8):
        return abs(a - b) / max(abs(a), abs(b)) < tol
    group_starting_value = svs[0]
    for i in range(1, len(svs)):
        if is_same(group_starting_value, svs[i]):
            continue
        group_starting_value = svs[i]
        distinct_sv_idx.append(i - 1)
    if distinct_sv_idx[len(distinct_sv_idx) - 1] != len(svs) - 1:
        distinct_sv_idx.append(len(svs) - 1)
    distinct_sv = [svs[idx] for idx in distinct_sv_idx]
    LL = calc_cutoff_likelihoods(distinct_sv)
    best_cutoff_idx = np.argmax(LL)
    return distinct_sv_idx[best_cutoff_idx]

# likelihoods_test = np.array([
#     10, 9, 3, 2, 1
# ])
# should be [-12.27178365  -5.36541382 -11.51601249 -12.90901428 -13.90901428]

def detect_cluster_structure_old(A:np.ndarray, sigma_tolerance):
    '''deprecated'''
    # start with n - 1 'active' cells one above the diagonal
    n = A.shape[0]
    C = np.zeros(A.shape)
    M = mean_from_diag(A)
    V = np.sqrt(var_from_diag(A))
    np.fill_diagonal(C, 1)
    # TODO: implement weighted update for mean and variance
    # CUM_M = np.zeros(A.shape) # cumulative mean
    # CUM_V = np.zeros(A.shape) # cumulative variance
    # np.fill_diagonal(CUM_M, np.diag(M))
    # np.fill_diagonal(CUM_V, np.diag(V))

    def should_join_clusters(r,c):
        combined_std = np.sqrt((V[r+1,c] + V[r,c-1]) / 2)
        combined_mean = (M[r+1,c] + M[r,c-1]) / 2
        return (abs(combined_mean - A[r,c]) < combined_std * sigma_tolerance) and A[r,c] > A[0,-1]

    for d in range(1, n):
        for k in range(0,n - d):
            r = k
            c = d + k
            if should_join_clusters(r,c) and (C[r+1,c] == 1) and (C[r,c-1] == 1):
                C[r,c] = 1
                C[c,r] = 1
            else:
                C[r,c] = 0
                C[c,r] = 0
    return C

def find_clusters(structure:np.ndarray):
    clusters = []
    n = structure.shape[0]
    cursor_row = 0
    cursor_col = 0
    prev_move_right = False
    while(cursor_col < n - 1):
        if (structure[cursor_row, cursor_col + 1] > 0.5):
            prev_move_right = True
            cursor_col += 1
        else:
            if prev_move_right:
                clusters.append(range(cursor_row, cursor_col + 1))
            prev_move_right = False
            if cursor_row == cursor_col:
                cursor_col += 1
            cursor_row += 1
    if prev_move_right:
        clusters.append(range(cursor_row, cursor_col + 1))
    return clusters

    

            
            
        