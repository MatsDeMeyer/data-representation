import numpy as np

#k-means clustering
def euclidean_vectorized(A, B):
    n, d = A.shape
    m, d1 = B.shape
    if not isinstance(A, np.ndarray):
        A = A.toarray()
    if not isinstance(B, np.ndarray):
        B = B.toarray()

    assert d == d1, 'Incompatible shape'
    A_squared = np.sum(np.square(A), axis=1, keepdims=True)
    B_squared = np.sum(np.square(B), axis=1, keepdims=True)
    AB = np.matmul(A, B.T)
    distances = np.sqrt(A_squared - 2 * AB + B_squared.T)
    return distances

# X: data matrix of size (n_samples,n_features)
# n_clusters: number of clusters
# output 1: labels of X with size (n_samples,)
# output 2: centroids of clusters
def kmeans(X,n_clusters):
    # initialize labels and prev_labels. prev_labels will be compared with labels to check if the stopping condition
    # have been reached.
    prev_labels = np.zeros(X.shape[0])
    labels = np.zeros(X.shape[0])
    
    # init random indices
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    
    # assign centroids using the indices
    centroids = X[indices]
    
    # the interative algorithm goes here
    while (True):
        # calculate the distances to the centroids
        distances = euclidean_vectorized(X,centroids)
        
        # assign labels
        labels = np.argmin(distances,axis=1)
        
        # stopping condition
        if np.array_equal(labels, prev_labels):
            break
        
        # calculate new centroids
        for cluster_indx in range(centroids.shape[0]):
            members = X[labels == cluster_indx]
            centroids[cluster_indx,:] = np.mean(members,axis=0)
        
        # keep the labels for next round's usage
        prev_labels = np.argmin(distances,axis=1)
    
    return labels,centroids