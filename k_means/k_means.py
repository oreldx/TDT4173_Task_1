import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self,
                 K=2,
                 max_iterations=1000,
                 init_type="random",
                 tolerance=0.0001,
                 normalize=False,
                 multiple_init=False,
                 ):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.max_iterations = max_iterations

        self.X = []
        self.K = K
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.n_samples, self.n_features = (0, 0)

        self.sse = None
        self.tolerance = tolerance
        self.init_type = init_type
        self.normalize = normalize
        self.multiple_init = multiple_init

    def fit(self, X):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """

        # Normalizing dataframe
        if self.normalize:
            for column in X.columns:
                X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min())

        # Converto from a Dataframe to a Numpy array
        self.X = X.to_numpy()
        self.n_samples, self.n_features = np.shape(X)

        # Initialize centroids with a method
        # Doing multiple initialization to find the one with lowest SSE/Distortion
        if self.multiple_init:
            possible_centroids = {}
            for _ in range(15):
                self.centroids = self.initialize_centroids(X)
                self.clusters = self.get_clusters()
                possible_centroids[self.get_sse()] = self.centroids
            self.centroids = possible_centroids[min(possible_centroids.keys())]
        else:
            self.centroids = self.initialize_centroids(X)

        for i in range(self.max_iterations):
            # Foreach point assign it to the closet cluster centroid
            self.clusters = self.get_clusters()

            # Compute the new centroid for each cluster
            old_centroids = self.centroids
            self.centroids = self.get_centroids()

            # Check if centroids are the same
            if np.all((old_centroids - self.centroids) == 0):
                break

            # Check if SSE is still having important variation
            elif i > self.max_iterations/2:
                old_sse = self.sse
                self.sse = self.get_sse()
                if old_sse is not None:
                    if old_sse - self.sse < self.tolerance:
                        break

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """

        # Foreach point assign it ot the closet cluster centroid
        self.clusters = self.get_clusters()

        # Putting in the correct form in 1D-array and clusters assignment [0, self.K-1]
        z = np.zeros(self.n_samples, int)
        for idx in range(self.n_samples):
            for idx_cluster, cluster in enumerate(self.clusters):
                if idx in cluster:
                    z[idx] = idx_cluster
                    break
        return z

    def initialize_centroids(self, X):
        """
        Args:
            X: data as Dataframe

        Returns:
            (array<self.K, self.n_features>): a matrix of floats with  self.K rows (#first centroids)
            and n columns (#features)
        """
        if self.init_type == "random":
            return np.asarray(X.sample(n=self.K))
        elif self.init_type == "kmeans++":
            # Setting the first centroid randomly
            centroids = [self.X[np.random.randint(0, self.n_samples)]]
            # Selecting next centroids as the maximal distance between all points and their closest centroid
            for idx in range(1, self.K):
                distances = np.asarray([[min([np.linalg.norm(point-centroid) for centroid in centroids])] for point in self.X])
                centroids.append(self.X[np.argmax(distances)])
            return centroids

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm

        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        centroids = np.zeros((self.K, self.n_features))
        for idx, idx_cluster in enumerate(self.clusters):
            cluster = [self.X[i] for i in idx_cluster]
            cluster_mean = np.mean(cluster, axis=0)
            centroids[idx] = cluster_mean
        return centroids

    def get_clusters(self):
        """
        Returns:
            (array<m,n>): a matrix of integers with self.K rows (#clusters)
             and n columns (#points assigned in the clusters)
        """
        clusters = [[] for _ in range(self.K)]
        for idx, point in enumerate(self.X):
            closet_centroid = self.get_closet_centroid(point)
            clusters[closet_centroid].append(idx)
        return clusters

    def get_closet_centroid(self, point):
        dist_euclidean = np.zeros(self.K)
        for idx, centroid in enumerate(self.centroids):
            dist_euclidean[idx] = np.linalg.norm(centroid-point)
        return np.argmin(dist_euclidean)

    def get_sse(self):
        self.sse = [0 for _ in range(self.K)]
        for idx, cluster in enumerate(self.clusters):
            for idx_point in cluster:
                self.sse[idx] += (np.linalg.norm(self.X[idx_point] - self.centroids[idx]) ** 2).sum()
        return np.sum(self.sse)

# --- Some utility functions


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion


def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))


def get_succes_rate(testing_iterations=100, verbose=True):
    """
    To get the success rate of the second data set by checking if the centroids are obtained in the right area.
    Get the average Silhouette and Distortion on the correct prediction.
    Args:
        verbose: boolean allow the print or not about the progress
        testing_iterations: integer that defines the number of execution of the k-means
    """
    data = pd.read_csv('data_2.csv')
    data_X = data[['x0', 'x1']]

    correct_predict = 0
    distortion = []
    silhouette = []
    for i in range(testing_iterations):
        if verbose:
            print(f"Iteration n°{i}/{testing_iterations} - Correct prediction {correct_predict}")
        correct_centroids = [[0.13279129, 0.84815637],
                             [0.58220928, 0.18037957],
                             [0.91298819, 0.75292059],
                             [0.23372082, 0.11692144],
                             [0.6884244, 0.59809779],
                             [0.1741503, 0.65839862],
                             [0.52354169, 0.87667458],
                             [0.85885356, 0.53989269],
                             [0.43386821, 0.28739245],
                             [0.32555331, 0.65838786]]

        # Running the model
        k_means = KMeans(K=10, max_iterations=1000, init_type="kmeans++", normalize=True)
        k_means.fit(data_X)
        z = k_means.predict(data_X)

        # Checking the correct centroids
        threshold = 0.05
        for centroid in k_means.centroids:
            for idx, correct_centroid in enumerate(correct_centroids):
                if np.linalg.norm(correct_centroid - centroid) < threshold:
                    del correct_centroids[idx]
                    break

        # If the model is correct
        if not correct_centroids:
            correct_predict += 1
            distortion.append(euclidean_distortion(data_X, z))
            silhouette.append(euclidean_silhouette(data_X, z))

    print(f"Succes rate : {correct_predict*100/testing_iterations}%")
    print(f"Average Distortion {sum(distortion)/correct_predict:.3f}")
    print(f"Average Silhouette {sum(silhouette)/correct_predict:.3f}")


if __name__ == '__main__':
    get_succes_rate(testing_iterations=1000)
