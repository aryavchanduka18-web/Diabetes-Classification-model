"""Stratified k-fold cross-validation, with preprocessing fitted inside each fold (no leakage)."""
import numpy as np

from . import preprocess as pp
from .metrics import evaluate

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def stratified_kfold(y, k=5, seed=42):
    """List of (train_idx, val_idx); every fold keeps the class ratio of y."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    offset = 0
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        fold_of[idx] = (np.arange(len(idx)) + offset) % k   # deal samples round-robin
        offset += len(idx)
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(k)]


def preprocess_split(X_train, X_other):
    """Impute + scale using statistics from X_train only; apply to both."""
    medians = pp.fit_medians(X_train)
    X_train, X_other = pp.impute(X_train, medians), pp.impute(X_other, medians)
    mean, std = pp.fit_scaler(X_train)
    return pp.scale(X_train, mean, std), pp.scale(X_other, mean, std)


def make_folds(X, y, k=5, seed=42):
    """Pre-processed folds, built once so every model sees the identical splits."""
    folds = []
    for tr, va in stratified_kfold(y, k, seed):
        Xt, Xv = preprocess_split(X[tr], X[va])
        folds.append((Xt, y[tr], Xv, y[va]))
    return folds


def cross_validate(factory, folds, threshold=0.5):
    """factory() -> fresh model with fit / predict_proba. Returns mean, sample std, per-fold values."""
    rows = []
    for Xt, yt, Xv, yv in folds:
        model = factory().fit(Xt, yt)
        m = evaluate(yv, model.predict_proba(Xv), threshold)
        rows.append([m[name] for name in METRICS])
    arr = np.array(rows)
    return {
        name: {"mean": float(arr[:, i].mean()),
               "std": float(arr[:, i].std(ddof=1)),
               "folds": arr[:, i].tolist()}
        for i, name in enumerate(METRICS)
    }
