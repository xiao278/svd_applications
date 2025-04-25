import numpy as np
from helper_functions import mean_from_diag, var_from_diag

def update_aggregates(existing_aggregate, new_value, new_weight):
    (cu_weight, mean, moo2) = existing_aggregate
    cu_weight += new_weight
    delta = new_value - mean
    mean += delta * new_weight / cu_weight
    delta2 = new_value - mean
    moo2 += new_weight * delta * delta2
    return (cu_weight, mean, moo2)

def infer_aggregates(matrices,r,c):
    (W, M, M2) = matrices
    w_a = W[r,c-1]
    w_b = W[r+1,c]
    w_c = W[r+1,c-1]
    
    m_a = M[r,c-1]
    m_b = M[r+1,c]
    m_c = M[r+1,c-1]

    m2_a = M2[r,c-1]
    m2_b = M2[r+1,c]
    m2_c = M2[r+1,c-1]
    if (abs(r - c) == 1):
        w_c = 0
        m_c = 0
        m2_c = 0

    cu_weight = w_a + w_b - w_c
    mean = (w_a * m_a + w_b * m_b - w_c * m_c) / cu_weight
    correction = w_a * np.square(m_a - mean) + w_b * np.square(m_b - mean) - w_c * np.square(m_c - mean)
    moo2 = m2_a + m2_b - m2_c + correction
    return (cu_weight, mean, moo2)

def init_agg_matrices(A:np.ndarray):
    W = np.zeros(A.shape) # replaces count in welfords
    np.fill_diagonal(W, 1) # diagonals should have all sum weight of one
    M = np.zeros(A.shape) # replaces mean in welfords
    np.fill_diagonal(M, np.diag(A))
    M2 = np.zeros(A.shape)
    return (W, M, M2)

def weighted_welford_example(A:np.ndarray):
    '''implemented from welfords algorithm on wikipedia'''
    # start with n - 1 'active' cells one above the diagonal
    n = A.shape[0]
    (W, M, M2) = init_agg_matrices(A)
    matrices = (W, M, M2)

    # def update(existing_aggregate, new_value):
    #     (count, mean, M2) = existing_aggregate
    #     count += 1
    #     delta = new_value - mean
    #     mean += delta / count
    #     delta2 = new_value - mean
    #     m2 += delta * delta2
    #     return (count, mean, m2)
    
    # # Retrieve the mean, variance and sample variance from an aggregate
    # def finalize(existing_aggregate):
    #     (count, mean, M2) = existing_aggregate
    #     if count < 2:
    #         return float("nan")
    #     else:
    #         (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
    #         return (mean, variance, sample_variance)

    for d in range(1, n):
        for k in range(0,n - d):
            r = k
            c = d + k
            existing_agg = infer_aggregates(matrices, r, c)
            new_value = A[r,c]
            (cu_weight, mean, moo2) = update_aggregates(existing_agg, new_value, 2)
            W[r,c] = cu_weight
            W[c,r] = cu_weight
            M[r,c] = mean
            M[c,r] = mean
            M2[r,c] = moo2
            M2[c,r] = moo2
    true_mean = mean_from_diag(A)
    true_var = var_from_diag(A, diag_zeroes=True)
    print(np.sum(np.square(M - true_mean)))
    variance = M2 / (W - 1 + np.identity(M2.shape[0]))
    print(np.sum(np.square(variance - true_var)))

def detect_cluster_structure(A:np.ndarray, sigma_tolerance = 0.05, debug=False):
    n = A.shape[0]
    (W, M, M2) = init_agg_matrices(A)
    matrices = (W, M, M2)
    total_mean = np.mean(A)
    total_var = np.var(A, ddof=1)

    def eval_local_stats(r,c):
        '''evaluate the local mean and variance'''
        padding = 1
        bound_r = max(r-padding, 0)
        bound_c = min(c+padding, n-1)
        diag_start = min(bound_r, bound_c)
        diag_end = max(bound_r, bound_c) + 1
        local_region = A[diag_start:diag_end,diag_start:diag_end]
        local_mean = np.mean(local_region)
        local_var = np.var(local_region)
        loc_min = np.min(local_region)
        return (local_mean, local_var, loc_min)

    for d in range(1, n):
        for k in range(0,n - d):
            r = k
            c = d + k
            if (W[r+1,c] > 0.5 and W[r,c-1] > 0.5):
                existing_agg = infer_aggregates(matrices, r, c)
                (ex_cu_weight, ex_mean, ex_moo2) = existing_agg
                variance = ex_moo2 / ex_cu_weight
                if d == 1:
                    (loc_mean, loc_var, loc_min) = eval_local_stats(r,c)
                    loc_size = 16
                    should_include = A[r,c] > (
                        (loc_min * loc_size + total_mean * n) / (loc_size + n)
                    ) + np.sqrt(loc_var)  # if very noisy threshold is higher, if not noisy threshold is lower
                else:
                    distance = abs(ex_mean - A[r,c])
                    should_include = distance < np.sqrt(1 / variance) * sigma_tolerance # 1 / var should be used rather than variance otherwise very uniform groups would have tight thresholds and messy groups have very loose thresholds. This creates clusters exactly where they shouldnt be. (between two squares)
                if should_include:
                    if d == 1:
                        weight = 1
                    else:
                        weight = np.exp(- distance / (2 * total_var)) # fixed variance to prevent runaway-pickups. 1/var makes it too restrictive on larger graphs
                    (cu_weight, mean, moo2) = update_aggregates(existing_agg, A[r,c], 2 * weight)
                    W[r,c] = cu_weight
                    W[c,r] = cu_weight
                    M[r,c] = mean
                    M[c,r] = mean
                    M2[r,c] = moo2
                    M2[c,r] = moo2
    return (W, M, M2) if debug else W > 0.5

def detect_cluster_basecase_test(A:np.ndarray, sigma_tolerance):
    n = A.shape[0]
    C = np.zeros(A.shape)
    np.fill_diagonal(C, 1)
    def eval_local_stats(r,c):
        '''evaluate the local mean and variance'''
        padding = 1
        bound_r = max(r-padding, 0)
        bound_c = min(c+padding, n-1)
        diag_start = min(bound_r, bound_c)
        diag_end = max(bound_r, bound_c) + 1
        local_region = A[diag_start:diag_end,diag_start:diag_end]
        local_mean = np.mean(local_region)
        local_var = np.var(local_region)
        return (local_mean, local_var)
    
    total_mean = np.mean(A)
    total_var = np.var(A)

    for d in range(1, 2):
        for k in range(0,n - d):
            r = k
            c = d + k
            (loc_mean, loc_var) = eval_local_stats(r,c)
            if np.abs(A[r,c] - loc_mean) < 1 / np.sqrt(loc_var) * sigma_tolerance:
                C[r,c] = 1
                C[c,r] = 1
    return C
            

