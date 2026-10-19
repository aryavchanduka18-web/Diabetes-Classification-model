"""k-Nearest Neighbours classifier implemented from scratch with NumPy.

Expects standardised features: distance-based, so unscaled columns would dominate.
"""
import numpy as np


class KNNScratch:
    def __init__(self, k=5, weighted=False):
        self.k = k
        self.weighted = weighted  # weight each neighbour by 1 / distance

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y).astype(float)
        if self.k > len(self.y_):
            raise ValueError("k larger than number of training samples")
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        # squared Euclidean distance: |a|^2 - 2 a.b + |b|^2
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ self.X_.T + (self.X_ ** 2).sum(1)[None, :]
        d2 = np.maximum(d2, 0.0)
        idx = np.argpartition(d2, self.k - 1, axis=1)[:, : self.k]
        votes = self.y_[idx]
        if not self.weighted:
            return votes.mean(axis=1)
        w = 1.0 / (np.sqrt(np.take_along_axis(d2, idx, axis=1)) + 1e-9)
        return (w * votes).sum(axis=1) / w.sum(axis=1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
