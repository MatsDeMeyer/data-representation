
import numpy as np
import copy

#euclidean distance
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

#find eps-neighborhood of a point
def find_eps_neighborhood(distances, ind, eps):
    '''
    Input arguments:
        - distances: a matrix containing distances between all pairs of points in the dataset
        - ind: index of the point of interest
        - eps: the epsilon parameter
    Output:
        - Retun a set of points in the neighborhood.
        (Note: Use Set data structure in Python)
    '''
    eps_neighbors = np.nonzero(distances[ind] <= eps)[0]
    return set(eps_neighbors.tolist())

#find all reachable points of a given point w.r.t eps
def find_reachable_pts(distances, eps, ind):
    eps_neighbors = find_eps_neighborhood(distances, ind, eps)
    reachables = eps_neighbors
    new_pts = copy.deepcopy(eps_neighbors)
    if ind in new_pts:
        new_pts.remove(ind)
    while len(new_pts) > 0:
        pt = new_pts.pop()
        pt_neighbors = find_eps_neighborhood(distances, pt, eps)
        additional_pts = pt_neighbors.difference(reachables)
        reachables.update(additional_pts)
        new_pts.update(additional_pts)
    return reachables


def dbscan(X, eps, minPts):
    ''' a simple implementation of DBSCAN algorithm
    In this implementation, a point is represented by its index in the dataset.
    In this function, except for the step to calculate the Euclidean distance,
    we will only work with the points't indices.

    Input arguments:
        - X: the dataset
        - eps: the epsilon parameter
        - minPts: the minimum number of points for a cluster
    Output:
        - core_points: a list containing the indices of the core points
        - cluster_labels: a Numpy array containing labels for each point in X
        - outliers: a list containing the indices of the outlier points
    '''
    # a list to keep track of the unvisited points
    unvisited = list(range(X.shape[0]))
    # list of core points (or cluster centroids)
    core_points = list([])
    # list of clusters, each cluster is a set of points
    clusters = list([])
    # list of outlier points (or noises)
    outliers = list([])
    #distances = euclidean_distances(X, X)
    distances = euclidean_vectorized(X, X)

    while True:
        # randomly choose a point, p, from the list of unvisited points
        ind = np.random.choice(unvisited, size=1)[0]

        # find the eps-neighborhood of the chosen point p
        eps_neighbors = find_eps_neighborhood(distances, ind, eps)

        # check if p is a core point or not
        is_core_pt = len(eps_neighbors) >= minPts

        if is_core_pt:
            # add the chosen index to the core_points list
            core_points.append(ind)

            # find all reachable points from p w.r.t eps and form a new cluster
            new_cluster = find_reachable_pts(distances, eps, ind)

            # add the newly formed cluster to the list of cluster
            clusters.append(new_cluster)

            # remove the indices in the new_cluster from the unvisited list and the outlier list,
            # if they were added to either those list before
            for ind in new_cluster:
                if ind in unvisited:
                    unvisited.remove(ind)
                if ind in outliers:
                    outliers.remove(ind)

        else:
            # if not core point, add p to the list of outlier points
            outliers.append(ind)

        # remove the chosen index from the unvisited list (if it is still inside this list)
        if ind in unvisited:
            unvisited.remove(ind)

        # if there is no point left in the unvisited list, stop the loop
        if len(unvisited) == 0:
            break

    # convert the resulting cluster list to cluster_labels
    cluster_labels = np.zeros(X.shape[0])
    for i in range(len(clusters)):
        for j in clusters[i]:
            cluster_labels[j] = i

    return core_points, cluster_labels, outliers