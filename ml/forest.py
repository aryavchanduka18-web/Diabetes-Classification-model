"""Random forest built on the from-scratch decision tree: bagging + random feature subsets."""
import numpy as np

from .tree import DecisionTreeScratch


class RandomForestScratch:
    def __init__(self, n_trees=100, max_depth=6, min_samples_leaf=1,
                 max_features="sqrt", seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.trees_ = []
        for _ in range(self.n_trees):
            boot = rng.integers(0, n, n)             # bootstrap sample (with replacement)
            tree = DecisionTreeScratch(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features, seed=int(rng.integers(2 ** 31)),
            ).fit(X[boot], y[boot])
            self.trees_.append(tree)
        self.importances_ = np.mean([t.importances_ for t in self.trees_], axis=0)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees_], axis=0)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
