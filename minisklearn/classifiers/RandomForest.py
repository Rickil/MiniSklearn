import numpy as np
from .DecisionTree import DecisionTree
from minisklearn import metrics

class RandomForest:  
    def __init__(self, n_estimators: int = 100, max_depth: int = 3, min_samples_split: int = 2, min_samples_leaf: int = 1, criterion: str = "gini", metric="accuracy"):  
        # Initialize forest parameters (number of trees, feature randomness)  
        self.forest = [DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion) for _ in range(n_estimators)]
        self.n_estimators = n_estimators
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:  
        # Train ensemble of decision trees on bootstrapped samples  
        for tree in self.forest:
            indexes = self._bootstrap_sample(len(X))
            tree.fit(X[indexes], y[indexes])

        predictions = self.predict(X)
        return getattr(metrics, self.metric)(predictions, y)

    def predict(self, X: np.ndarray) -> np.ndarray:  
        # Aggregate predictions from all trees (majority vote)
        predictions = []  
        for tree in self.forest:
            predictions.append(tree.predict(X))
        
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def _bootstrap_sample(self, size: int) -> list:  
        # Helper: Generate a random bootstrap sample (with replacement)  
        indexes = np.random.randint(0, size, size=size)
        return indexes