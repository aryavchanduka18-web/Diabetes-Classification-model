"""Logistic Regression implemented from scratch with NumPy (batch gradient descent)."""
import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid: 1 / (1 + e^-z)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def binary_cross_entropy(y, p, w=None, l2=0.0, eps=1e-12):
    """Mean log-loss, plus optional L2 penalty (bias excluded)."""
    p = np.clip(p, eps, 1 - eps)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    if w is not None and l2 > 0:
        loss += (l2 / (2 * len(y))) * np.sum(w ** 2)
    return loss


class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=2000, l2=0.0, class_weight=None):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.class_weight = class_weight  # None or "balanced"
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def _gradients(self, X, y, sample_w):
        n = X.shape[0]
        p = sigmoid(X @ self.w + self.b)
        err = (p - y) * sample_w
        dw = (X.T @ err) / n + (self.l2 / n) * self.w
        db = np.sum(err) / n
        return dw, db, p

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []

        sample_w = np.ones(n)
        if self.class_weight == "balanced":
            n_pos = y.sum()
            n_neg = n - n_pos
            sample_w = np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))

        for _ in range(self.n_iter):
            dw, db, p = self._gradients(X, y, sample_w)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            self.loss_history.append(binary_cross_entropy(y, p, self.w, self.l2))
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, dtype=float) @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
