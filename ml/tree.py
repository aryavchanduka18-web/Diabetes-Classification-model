"""Decision tree (CART, Gini impurity) implemented from scratch with NumPy."""
import numpy as np


def gini(pos, n):
    """Gini impurity of a node with `pos` positives out of `n` samples: 2p(1-p)."""
    p = pos / n
    return 2.0 * p * (1.0 - p)


class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # None, "sqrt" or an int (used by the forest)
        self.seed = seed

    def _n_feats(self, d):
        if self.max_features is None:
            return d
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        return min(d, int(self.max_features))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.n_features_ = X.shape[1]
        self.importances_ = np.zeros(self.n_features_)
        self._rng = np.random.default_rng(self.seed)
        self.root_ = self._build(X, y, 0)
        total = self.importances_.sum()
        if total > 0:
            self.importances_ = self.importances_ / total
        return self

    def _best_split(self, X, y, pos):
        """Scan every threshold of every candidate feature in O(n log n) per feature."""
        n, d = X.shape
        m = self._n_feats(d)
        feats = self._rng.choice(d, m, replace=False) if m < d else np.arange(d)
        parent = gini(pos, n)
        best_gain, best_j, best_t = 0.0, None, None
        msl = self.min_samples_leaf
        for j in feats:
            order = np.argsort(X[:, j], kind="stable")
            xs, ys = X[order, j], y[order]
            left_n = np.arange(1, n)                 # split after sample i -> i+1 on the left
            left_pos = np.cumsum(ys)[:-1]
            right_n = n - left_n
            right_pos = pos - left_pos
            valid = (xs[1:] != xs[:-1]) & (left_n >= msl) & (right_n >= msl)
            if not valid.any():
                continue
            pl, pr = left_pos / left_n, right_pos / right_n
            weighted = (left_n * 2 * pl * (1 - pl) + right_n * 2 * pr * (1 - pr)) / n
            weighted = np.where(valid, weighted, np.inf)
            i = int(np.argmin(weighted))
            gain = parent - weighted[i]
            if gain > best_gain + 1e-12:
                t = (xs[i] + xs[i + 1]) / 2.0
                if t >= xs[i + 1]:                   # float rounding guard
                    t = xs[i]
                best_gain, best_j, best_t = gain, int(j), float(t)
        return best_gain, best_j, best_t

    def _build(self, X, y, depth):
        n = len(y)
        pos = int(y.sum())
        node = {"p": pos / n, "n": n}
        if depth >= self.max_depth or n < self.min_samples_split or pos in (0, n):
            return node
        gain, j, t = self._best_split(X, y, pos)
        if j is None:
            return node
        mask = X[:, j] <= t
        self.importances_[j] += n * gain            # impurity decrease weighted by node size
        node.update(j=j, t=t,
                    left=self._build(X[mask], y[mask], depth + 1),
                    right=self._build(X[~mask], y[~mask], depth + 1))
        return node

    def _route(self, node, X, idx, out):
        if "j" not in node:
            out[idx] = node["p"]
            return
        go_left = X[idx, node["j"]] <= node["t"]
        self._route(node["left"], X, idx[go_left], out)
        self._route(node["right"], X, idx[~go_left], out)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X))
        self._route(self.root_, X, np.arange(len(X)), out)
        return out

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def depth(self):
        def d(node):
            return 0 if "j" not in node else 1 + max(d(node["left"]), d(node["right"]))
        return d(self.root_)

    def export_text(self, feature_names):
        """Human-readable rules, handy for the viva."""
        lines = []

        def walk(node, indent):
            pad = "|   " * indent
            if "j" not in node:
                lines.append(f"{pad}-> p(diabetes)={node['p']:.2f}  (n={node['n']})")
                return
            lines.append(f"{pad}if {feature_names[node['j']]} <= {node['t']:.3f}:")
            walk(node["left"], indent + 1)
            lines.append(f"{pad}else:")
            walk(node["right"], indent + 1)

        walk(self.root_, 0)
        return "\n".join(lines)
