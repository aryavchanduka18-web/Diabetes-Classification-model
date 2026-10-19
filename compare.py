"""Compare all models with the same stratified 5-fold CV, then score once on a held-out test set.

Protocol (no information leaks from validation/test data into training):
  1. Stratified 80/20 split (same seed as train.py). The 20% test set is untouched until step 4.
  2. Stratified 5-fold CV on the 80% training part; imputation + scaling are refit inside every fold.
  3. KNN k, tree depth and forest depth are chosen by mean CV F1 over those folds.
  4. Each model is refit on the full training part and scored once on the test set.

Usage:  python compare.py [--csv diabetes.csv] [--out results.json]
"""
import argparse
import json
import time

import numpy as np

from ml import preprocess as pp
from ml.cv import METRICS, cross_validate, make_folds, preprocess_split
from ml.forest import RandomForestScratch
from ml.knn import KNNScratch
from ml.logreg import LogisticRegressionScratch
from ml.metrics import evaluate
from ml.tree import DecisionTreeScratch

K_GRID = [1, 3, 5, 7, 9, 11, 15, 21, 31]
TREE_GRID = [1, 2, 3, 4, 5, 6, 8, 10]
FOREST_GRID = [3, 5, 8]
TUNING_METRIC = "f1"


def tune(make, grid, folds):
    rows = []
    for v in grid:
        res = cross_validate(lambda: make(v), folds)
        rows.append({"param": v, **{m: res[m]["mean"] for m in METRICS}})
    best = max(rows, key=lambda r: r[TUNING_METRIC])
    return best["param"], rows


def test_scores(factory, Xtr, ytr, Xte, yte):
    m = evaluate(yte, factory().fit(Xtr, ytr).predict_proba(Xte))
    return {
        **{name: float(m[name]) for name in METRICS},
        "cm": m["confusion_matrix"].tolist(),
        "fpr": m["fpr"].tolist(),
        "tpr": m["tpr"].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="diabetes.csv")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()

    X, y = pp.load_csv(args.csv)
    X_tr, X_te, y_tr, y_te = pp.stratified_split(X, y, test_size=0.2, seed=args.seed)
    folds = make_folds(X_tr, y_tr, args.folds, args.seed)
    Xtr_s, Xte_s = preprocess_split(X_tr, X_te)
    print(f"train={len(y_tr)}  test={len(y_te)}  folds={args.folds}")

    print("Tuning on CV folds ...")
    best_k, k_rows = tune(lambda k: KNNScratch(k=k), K_GRID, folds)
    best_d, tree_rows = tune(
        lambda d: DecisionTreeScratch(max_depth=d, min_samples_leaf=5), TREE_GRID, folds)
    best_fd, forest_rows = tune(
        lambda d: RandomForestScratch(n_trees=100, max_depth=d, min_samples_leaf=3, seed=0),
        FOREST_GRID, folds)
    print(f"  best k={best_k}  tree depth={best_d}  forest depth={best_fd}  (by CV {TUNING_METRIC})")

    models = {
        "Logistic Regression": ("scratch", "lr=0.1, 3000 iters",
                                lambda: LogisticRegressionScratch(lr=0.1, n_iter=3000)),
        "Logistic Regression (balanced)": ("scratch", "lr=0.1, 3000 iters, class-weighted",
                                           lambda: LogisticRegressionScratch(
                                               lr=0.1, n_iter=3000, class_weight="balanced")),
        "KNN": ("scratch", f"k={best_k}", lambda: KNNScratch(k=best_k)),
        "Decision Tree": ("scratch", f"max_depth={best_d}, min_leaf=5",
                          lambda: DecisionTreeScratch(max_depth=best_d, min_samples_leaf=5)),
        "Random Forest": ("scratch", f"100 trees, max_depth={best_fd}, min_leaf=3",
                          lambda: RandomForestScratch(n_trees=100, max_depth=best_fd,
                                                      min_samples_leaf=3, seed=0)),
    }
    library_ok = True
    try:
        from ml.bench import library_models
        for name, factory in library_models().items():
            models[name] = ("library", "default hyper-parameters, untuned", factory)
    except ImportError:
        library_ok = False
        print("scikit-learn not installed: skipping library benchmark")

    results = {}
    for name, (kind, params, factory) in models.items():
        print(f"CV + test: {name}")
        results[name] = {
            "kind": kind, "params": params,
            "cv": cross_validate(factory, folds),
            "test": test_scores(factory, Xtr_s, y_tr, Xte_s, y_te),
        }

    best_f1 = max(results, key=lambda n: results[n]["cv"]["f1"]["mean"])
    b = results[best_f1]["cv"]["f1"]
    within = [n for n in results if results[n]["cv"]["f1"]["mean"] >= b["mean"] - b["std"]]

    out = {
        "config": {"seed": args.seed, "folds": args.folds, "n_train": int(len(y_tr)),
                   "n_test": int(len(y_te)), "tuning_metric": TUNING_METRIC,
                   "library_available": library_ok, "threshold": 0.5},
        "grids": {"KNN (k)": k_rows, "Decision Tree (max_depth)": tree_rows,
                  "Random Forest (max_depth)": forest_rows},
        "best_cv_f1": best_f1,
        "within_one_std_of_best_f1": within,
        "models": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)

    print(f"\n{'model':34s}" + "".join(f"{m:>16s}" for m in METRICS))
    for name, r in results.items():
        print(f"{name:34s}" + "".join(
            f"{r['cv'][m]['mean']:>9.3f}±{r['cv'][m]['std']:.3f}" for m in METRICS))
    print(f"\nBest by CV F1: {best_f1}. Within 1 std of it: {within}")
    print(f"Saved {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
