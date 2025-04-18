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

def determine_rank_cutoff(svs:list[float]):
    angle_list = []
    sv_pair_list = []

    angle_diff_list = []
    line_seg_pair_list = [] # line segment number 0 consists of points (0,1), line segment number 1 consists of points (1,2)
    def calc_angle(y0, y1): # assumes x1 - x0 = 1
        x1 = 1
        x0 = 0
        delta_x = x1 - x0
        delta_y = y1 - y0
        return np.arctan(delta_y / delta_x) / np.pi * 180
    for i in range(len(svs) - 1):
        angle_list.append(calc_angle(svs[i], svs[i+1]))
        sv_pair_list.append((i, i+1))
    line_0_index = 0
    for i in range(1, len(angle_list)):
        line_1_angle = angle_list[i]
        if (np.abs(line_1_angle) < 0.001): # if next sv is flat
            continue
        angle_diff_list.append(line_1_angle - angle_list[line_0_index])
        line_seg_pair_list.append((line_0_index, i))
        line_0_index = i
    return ({'angles': angle_diff_list, 'points': line_seg_pair_list})