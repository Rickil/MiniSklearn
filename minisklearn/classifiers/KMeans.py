import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4):
        """
        Parameters:
        - n_clusters (int): Number of clusters (K).
        - max_iters (int): Maximum iterations for convergence.
        - tol (float): Tolerance for centroid movement to declare convergence.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X: np.ndarray):
        """
        Train K-Means on input data X.
        
        Parameters:
        - X (ndarray): Input data of shape (n_samples, n_features).
        """
        self._initialize_centroids(X)
        iter = 0
        while iter < self.max_iters:
            labels = self.predict(X)

            new_centroids = []
            for n in range(self.n_clusters):
                new_centroids.append(X[labels==n].mean(axis=0))
            new_centroids = np.array(new_centroids)

            movement = np.linalg.norm(self.centroids-new_centroids)
            self.centroids = new_centroids

            # Check if the model has converged
            if movement <= self.tol:
                break
        
        predictions = self.predict(X)

        return predictions

    def predict(self, X):
        """
        Assign each sample to the nearest cluster.
        
        Returns:
        - labels (ndarray): Cluster indices for each sample.
        """
        diff = X[:, None, :] - self.centroids[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        labels = np.argmin(distances, axis=-1)

        return labels

    def _initialize_centroids(self, X):
        """
        Helper: Randomly initialize centroids.
        """
        self.centroids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]