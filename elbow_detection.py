import numpy as np

def calc_lambda_max(svs:np.ndarray):
    assert svs.ndim == 1
    n = svs.shape[0]
    lambda_max = 0
    for k in range(1, n):
        lambda_new = (svs[0] - svs[k]) / k
        lambda_max = max(lambda_max, lambda_new)
    return lambda_max

def calc_weights(v_term, lambda_max, M=1e5):
    lambdas = np.random.uniform(0, lambda_max, int(M))
    k_frequency = np.zeros(v_term.shape)
    baseline_penalty = np.array(range(0,v_term.shape[0]))
    for l in lambdas:
        k = np.argmin(v_term + baseline_penalty * l)
        k_frequency[k] += 1
    return k_frequency / M

def calc_elbow(svs, conf=0.90):
    '''Implemented using https://arxiv.org/abs/2308.09108'''
    assert conf < 1
    assert conf > 0
    l_max = calc_lambda_max(svs)
    weights = calc_weights(svs, l_max)
    cusum_weights = np.cumsum(weights)
    for k in range(cusum_weights.shape[0]):
        if cusum_weights[k] >= conf:
            return k
    return -1