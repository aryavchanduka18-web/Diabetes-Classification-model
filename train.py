"""Train the from-scratch logistic regression on the Pima dataset and save model.npz.

Usage:  python train.py [--csv diabetes.csv] [--lr 0.1] [--iters 3000]
"""
import argparse

import numpy as np

from ml import preprocess as pp
from ml.logreg import LogisticRegressionScratch
from ml.metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="diabetes.csv")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--balanced", action="store_true", help="class-weighted loss (boosts recall)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="model.npz")
    args = ap.parse_args()

    X, y = pp.load_csv(args.csv)
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features; positives = {y.sum()} ({y.mean():.1%})")

    X_tr, X_te, y_tr, y_te = pp.stratified_split(X, y, test_size=0.2, seed=args.seed)

    # every statistic below is learned from the TRAIN split only
    medians = pp.fit_medians(X_tr)
    X_tr, X_te = pp.impute(X_tr, medians), pp.impute(X_te, medians)
    mean, std = pp.fit_scaler(X_tr)
    X_tr, X_te = pp.scale(X_tr, mean, std), pp.scale(X_te, mean, std)

    model = LogisticRegressionScratch(
        lr=args.lr, n_iter=args.iters, l2=args.l2,
        class_weight="balanced" if args.balanced else None,
    ).fit(X_tr, y_tr)

    train_m = evaluate(y_tr, model.predict_proba(X_tr), args.threshold)
    test_m = evaluate(y_te, model.predict_proba(X_te), args.threshold)

    print(f"Final training loss: {model.loss_history[-1]:.4f}")
    for name, m in (("train", train_m), ("test ", test_m)):
        print(f"[{name}] acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
              f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  auc={m['roc_auc']:.3f}")
    print("Test confusion matrix [[TN FP] [FN TP]]:\n", test_m["confusion_matrix"])

    np.savez(
        args.out,
        w=model.w, b=model.b, mean=mean, std=std,
        median_cols=np.array(list(medians.keys())),
        median_vals=np.array(list(medians.values())),
        loss_history=np.array(model.loss_history),
        fpr=test_m["fpr"], tpr=test_m["tpr"], cm=test_m["confusion_matrix"],
        metrics=np.array([test_m["accuracy"], test_m["precision"], test_m["recall"],
                          test_m["f1"], test_m["roc_auc"]]),
        train_metrics=np.array([train_m["accuracy"], train_m["precision"], train_m["recall"],
                                train_m["f1"], train_m["roc_auc"]]),
        threshold=args.threshold, n_train=len(y_tr), n_test=len(y_te),
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
