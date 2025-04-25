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
    '''implemented from welfords algorithm'''
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
    total_var = np.var(A)

    for d in range(1, n):
        for k in range(0,n - d):
            r = k
            c = d + k
            if (W[r+1,c] > 0.5 and W[r,c-1] > 0.5):
                existing_agg = infer_aggregates(matrices, r, c)
                (ex_cu_weight, ex_mean, ex_moo2) = existing_agg
                variance = ex_moo2 / ex_cu_weight
                distance = abs(ex_mean - A[r,c])
                if d == 1:
                    should_include = A[r,c] > total_mean + np.sqrt(total_var)
                else:
                    # weighted_std = np.sqrt(
                    #     (
                    #         (variance * ex_cu_weight ** 2) + (total_var * n ** 2)
                    #     ) / (ex_cu_weight ** 2 + n ** 2)
                    # )
                    should_include = distance < np.sqrt(1 / variance) * sigma_tolerance
                if should_include:
                    if d == 1:
                        weight = 1
                    else:
                        weight = np.exp(- distance / (2 * total_var))
                    (cu_weight, mean, moo2) = update_aggregates(existing_agg, A[r,c], 2 * weight)
                    W[r,c] = cu_weight
                    W[c,r] = cu_weight
                    M[r,c] = mean
                    M[c,r] = mean
                    M2[r,c] = moo2
                    M2[c,r] = moo2
    return (W, M, M2) if debug else W > 0.5

