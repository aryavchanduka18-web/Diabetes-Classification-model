"""Sanity tests for the from-scratch code. Run:  python tests.py
Uses small hand-made / synthetic arrays only (not the real dataset)."""
import numpy as np

from ml import preprocess as pp
from ml.logreg import LogisticRegressionScratch, binary_cross_entropy, sigmoid
from ml.metrics import auc, confusion_matrix, evaluate, f1_score, precision, recall, roc_curve


def test_sigmoid():
    assert np.allclose(sigmoid(np.array([0.0])), 0.5)
    s = sigmoid(np.array([-1000.0, 1000.0]))  # must not overflow
    assert np.allclose(s, [0.0, 1.0])


def test_gradient_matches_numerical():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = (rng.random(50) > 0.5).astype(float)
    m = LogisticRegressionScratch(l2=0.5)
    m.w = rng.normal(size=3)
    m.b = 0.3
    dw, db, _ = m._gradients(X, y, np.ones(50))

    def loss(w, b):
        return binary_cross_entropy(y, sigmoid(X @ w + b), w, 0.5)

    eps = 1e-6
    for j in range(3):
        wp, wm = m.w.copy(), m.w.copy()
        wp[j] += eps
        wm[j] -= eps
        assert abs((loss(wp, m.b) - loss(wm, m.b)) / (2 * eps) - dw[j]) < 1e-6
    assert abs((loss(m.w, m.b + eps) - loss(m.w, m.b - eps)) / (2 * eps) - db) < 1e-6


def test_learns_separable_data():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-2, 1, (100, 2)), rng.normal(2, 1, (100, 2))])
    y = np.r_[np.zeros(100), np.ones(100)]
    m = LogisticRegressionScratch(lr=0.1, n_iter=500).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.95
    assert m.loss_history[-1] < m.loss_history[0]


def test_metrics_by_hand():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([1, 1, 0, 0, 0, 1, 0, 0])
    cm = confusion_matrix(y, p)
    assert cm.tolist() == [[3, 1], [2, 2]]  # TN FP / FN TP
    assert abs(precision(cm) - 2 / 3) < 1e-12
    assert abs(recall(cm) - 0.5) < 1e-12
    assert abs(f1_score(cm) - 4 / 7) < 1e-12


def test_auc_known_values():
    # classic textbook example: AUC = 0.75
    fpr, tpr = roc_curve(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert abs(auc(fpr, tpr) - 0.75) < 1e-12
    fpr, tpr = roc_curve(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert abs(auc(fpr, tpr) - 1.0) < 1e-12
    fpr, tpr = roc_curve(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5]))  # ties
    assert abs(auc(fpr, tpr) - 0.5) < 1e-12


def test_preprocessing_no_leakage():
    X = np.zeros((10, 8))
    X[:, 1] = [100, 0, 120, 110, 0, 130, 90, 95, 105, 115]  # Glucose
    X[:, 2] = 70
    X[:, 3] = 20
    X[:, 4] = 80
    X[:, 5] = 30
    med = pp.fit_medians(X)
    assert med[1] == np.median([100, 120, 110, 130, 90, 95, 105, 115])  # zeros ignored
    Xi = pp.impute(X, med)
    assert (Xi[:, 1] != 0).all()
    y = np.array([0, 1] * 5)
    a, b, ya, yb = pp.stratified_split(X, y, 0.2, seed=0)
    assert len(yb) == 2 and yb.sum() == 1  # class ratio kept


def test_evaluate_runs():
    r = evaluate(np.array([0, 1, 1, 0]), np.array([0.2, 0.9, 0.6, 0.4]))
    assert r["accuracy"] == 1.0 and r["roc_auc"] == 1.0


def test_tree_best_split_by_hand():
    from ml.tree import DecisionTreeScratch
    t = DecisionTreeScratch(max_depth=1).fit(np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1]))
    assert t.root_["j"] == 0 and t.root_["t"] == 2.5          # perfect split between 2 and 3
    assert t.predict(np.array([[1.5], [3.5]])).tolist() == [0, 1]


def test_tree_depth_limit_and_fit():
    from ml.tree import DecisionTreeScratch
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    shallow = DecisionTreeScratch(max_depth=2).fit(X, y)
    deep = DecisionTreeScratch(max_depth=12).fit(X, y)
    assert shallow.depth() <= 2
    assert (deep.predict(X) == y).mean() == 1.0                 # unlimited depth memorises training data
    assert abs(deep.importances_.sum() - 1.0) < 1e-9


def test_knn_by_hand():
    from ml.knn import KNNScratch
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([0, 0, 1, 1])
    assert KNNScratch(k=1).fit(X, y).predict(np.array([[9.0]])).tolist() == [1]
    p = KNNScratch(k=3).fit(X, y).predict_proba(np.array([[0.4]]))   # neighbours: 0, 1, 10
    assert abs(p[0] - 1 / 3) < 1e-12


def test_forest_learns_and_is_reproducible():
    from ml.forest import RandomForestScratch
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)                    # XOR-like: a linear model cannot learn this
    a = RandomForestScratch(n_trees=30, max_depth=6, seed=1).fit(X, y)
    b = RandomForestScratch(n_trees=30, max_depth=6, seed=1).fit(X, y)
    assert (a.predict(X) == y).mean() > 0.9
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))
    assert np.argmax(a.importances_) in (0, 1)


def test_stratified_kfold_properties():
    from ml.cv import stratified_kfold
    y = np.array([0] * 70 + [1] * 30)
    folds = stratified_kfold(y, k=5, seed=0)
    all_val = np.concatenate([va for _, va in folds])
    assert sorted(all_val.tolist()) == list(range(100))          # each sample validated exactly once
    for tr, va in folds:
        assert not set(tr) & set(va)
        assert abs(y[va].sum() - 6) <= 1                          # 30% positives in every fold


def test_cv_preprocessing_is_per_fold():
    from ml.cv import make_folds
    rng = np.random.default_rng(4)
    X = rng.uniform(1, 100, size=(60, 8))
    y = np.array([0, 1] * 30)
    for Xt, yt, Xv, yv in make_folds(X, y, k=3, seed=0):
        assert np.allclose(Xt.mean(axis=0), 0, atol=1e-9)        # scaled with TRAIN stats only
        assert not np.allclose(Xv.mean(axis=0), 0, atol=1e-9)    # validation is not re-centred on itself


if __name__ == "__main__":
    tests =[v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
