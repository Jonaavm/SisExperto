"""
K-Nearest Neighbors Algorithm Implementation
"""
import numpy as np


class KNN:
    """K-Nearest Neighbors Classifier"""
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the KNN model (just store the data)"""
        self.X = X.astype(float)
        self.y = y

    def _distances(self, row):
        """Calculate Euclidean distances from a row to all training samples"""
        return np.sqrt(np.sum((self.X - row.astype(float))**2, axis=1))

    def predict(self, X: np.ndarray):
        """Predict multiple rows"""
        preds = []
        for r in X:
            dists = self._distances(r)
            idx = np.argsort(dists)[:self.k]
            vals, counts = np.unique(self.y[idx], return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)
