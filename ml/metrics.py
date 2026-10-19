"""Classification metrics implemented from scratch (no scikit-learn)."""
import numpy as np


def confusion_matrix(y_true, y_pred):
    """Returns [[TN, FP], [FN, TP]]."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def _safe_div(a, b):
    return a / b if b else 0.0


def accuracy(cm):
    return _safe_div(cm[0, 0] + cm[1, 1], cm.sum())


def precision(cm):
    return _safe_div(cm[1, 1], cm[1, 1] + cm[0, 1])


def recall(cm):
    return _safe_div(cm[1, 1], cm[1, 1] + cm[1, 0])


def f1_score(cm):
    p, r = precision(cm), recall(cm)
    return _safe_div(2 * p * r, p + r)


def roc_curve(y_true, scores):
    """FPR/TPR at every distinct score threshold (descending)."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    s_sorted = scores[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    # keep last index of each distinct score so ties are handled correctly
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, len(y_sorted) - 1]
    tpr = np.r_[0.0, tps[idx] / tps[-1]]
    fpr = np.r_[0.0, fps[idx] / fps[-1]]
    return fpr, tpr


def auc(fpr, tpr):
    """Area under curve by trapezoid rule."""
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def evaluate(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr = roc_curve(y_true, proba)
    return {
        "confusion_matrix": cm,
        "accuracy": accuracy(cm),
        "precision": precision(cm),
        "recall": recall(cm),
        "f1": f1_score(cm),
        "roc_auc": auc(fpr, tpr),
        "fpr": fpr,
        "tpr": tpr,
    }
