import numpy as np
import heapq
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

def find_clusters(structure:np.ndarray, min_cluster_size = 3):
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
            if prev_move_right and np.abs(cursor_row - cursor_col) + 1 >= min_cluster_size:
                clusters.append(range(cursor_row, cursor_col + 1))
            prev_move_right = False
            if cursor_row == cursor_col:
                cursor_col += 1
            cursor_row += 1
    if prev_move_right and np.abs(cursor_row - cursor_col) + 1 >= min_cluster_size:
        clusters.append(range(cursor_row, cursor_col + 1))
    return clusters

def reorder_cosine_matrix(CM:np.ndarray, index=False):
    '''
    Takes a cosine matrix and returns a reordered copy of it. if index=True, returns a tuple (CM_REORDERED, INDEX) instead
    implmented from https://www.researchgate.net/publication/51152953_Community_detection_in_graphs_using_singular_value_decomposition
    '''
    INDEX = [] # global reordering index
    CM_COPY = CM.copy()
    indices = list(range(0,CM.shape[1]))
    while(len(indices) > 1):
        # sort by cosine similarity
        similarity_list = CM_COPY[0,:]
        new_indices = np.argsort(similarity_list)[::-1]
        
        # rearrange matrix
        indices = [indices[i] for i in new_indices]
        CM_COPY = CM_COPY[np.ix_(new_indices,new_indices)]

        # remove and update
        INDEX.append(indices[0])
        indices.pop(0)
        CM_COPY = CM_COPY[1:,1:]
    INDEX.append(indices[0])
    CM_REORDERED = CM[np.ix_(INDEX, INDEX)]
    if index:
        return (CM_REORDERED, INDEX)
    else:
        return CM_REORDERED

def compute_cosine_matrix(S:np.ndarray,V:np.ndarray,m,n):
    CM = np.zeros((m,n))
    vectors = S @ V.T
    for i in range(0,vectors.shape[1]):
        for j in range(i+1, vectors.shape[1]):
            u = vectors[:,i]
            v = vectors[:,j]
            cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            CM[i,j] = cosine
            CM[j,i] = cosine
    np.fill_diagonal(CM, 1)
    return CM

class Partition:
    def __init__(self):
        self.last_node = 0
        self.span = 0
        self.clusters:list[range] = []

    def __lt__(self, other:'Partition'):
        return self.last_node < other.last_node
    
    def add_cluster(self, cluster:range):
        assert self.can_add(cluster)
        self.clusters.append(cluster)
        self.last_node = cluster.stop
        self.span += cluster.stop - cluster.start
    
    def can_add(self, cluster:range):
        return self.last_node <= cluster.start

def partition_clusters(clusters:list[range]):
    # cluster already sorted by end time
    heap = [Partition()]
    for cluster in clusters:
        best_partition = heap[0]
        if best_partition.can_add(cluster):
            heapq.heappop(heap)
            best_partition.add_cluster(cluster)
            heapq.heappush(heap, best_partition)
        else: 
            new_partition = Partition()
            new_partition.add_cluster(cluster)
            heapq.heappush(heap, new_partition)
    heap.sort(key = lambda item: item.span, reverse=True)
    for h in heap:
        print(h.clusters, h.span)
    cluster_partitions = [p.clusters for p in heap]
    
    num_clusters = [len(cp) for cp in cluster_partitions]
    prev_clusters = np.roll(num_clusters, 1)
    prev_clusters[0] = 0
    prev_clusters = np.cumsum(prev_clusters)
    return (cluster_partitions, prev_clusters)

