"""All the machine learning for the project, using scikit-learn.

Run `python model.py` to train the models and print the results.
The Streamlit app (app.py) imports train_all() from here.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
# In these columns a value of 0 is impossible, so it really means "not measured".
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
SEED = 42
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# model name -> (scikit-learn estimator, hyper-parameter values to try)
MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"model__class_weight": [None, "balanced"]}),
    "KNN": (
        KNeighborsClassifier(),
        {"model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31]}),
    "Decision Tree": (
        DecisionTreeClassifier(min_samples_leaf=5, random_state=SEED),
        {"model__max_depth": [1, 2, 3, 4, 5, 6, 8, 10]}),
}


def load_data(path="diabetes.csv"):
    df = pd.read_csv(path)
    return df[FEATURES], df["Outcome"]


def make_pipeline(estimator):
    """Fill the hidden zeros with the median, scale every feature, then fit the model.

    A Pipeline re-learns the medians and the scaling from the training rows only, every
    time it is fitted (also inside each cross-validation fold), so no test data leaks in.
    """
    fill_zeros = ColumnTransformer(
        [("median_fill", SimpleImputer(missing_values=0, strategy="median"), ZERO_AS_MISSING)],
        remainder="passthrough")          # Pregnancies, pedigree and age pass through unchanged
    return Pipeline([("fill_zeros", fill_zeros),
                     ("scale", StandardScaler()),
                     ("model", estimator)])


def logistic_coefficients(pipeline):
    """Weights of a fitted logistic regression pipeline, labelled by feature."""
    names = ZERO_AS_MISSING + [f for f in FEATURES if f not in ZERO_AS_MISSING]
    return pd.Series(pipeline.named_steps["model"].coef_[0], index=names)


def train_all(path="diabetes.csv"):
    X, y = load_data(path)
    # 80/20 split; stratify keeps the same share of diabetes cases in both parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = {}
    for name, (estimator, grid) in MODELS.items():
        # try every setting in the grid with 5-fold CV, keep the best mean F1, refit on all training rows
        search = GridSearchCV(make_pipeline(estimator), grid, cv=cv, scoring=METRICS, refit="f1")
        search.fit(X_train, y_train)
        scores, best = search.cv_results_, search.best_index_

        proba = search.predict_proba(X_test)[:, 1]
        pred = search.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, proba)
        results[name] = {
            "model": search.best_estimator_,
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "cv": {m: (float(scores[f"mean_test_{m}"][best]), float(scores[f"std_test_{m}"][best]))
                   for m in METRICS},
            "test": {"accuracy": accuracy_score(y_test, pred),
                     "precision": precision_score(y_test, pred),
                     "recall": recall_score(y_test, pred),
                     "f1": f1_score(y_test, pred),
                     "roc_auc": roc_auc_score(y_test, proba)},
            "confusion_matrix": confusion_matrix(y_test, pred),
            "roc": (fpr, tpr),
            "grid": pd.DataFrame(scores["params"])
                      .rename(columns=lambda c: c.replace("model__", ""))
                      .assign(cv_f1=scores["mean_test_f1"]),
        }
    return results, {"n_train": len(y_train), "n_test": len(y_test)}


if __name__ == "__main__":
    results, info = train_all()
    print(f"training rows: {info['n_train']}   test rows: {info['n_test']}")
    for name, r in results.items():
        print(f"\n{name}   best settings: {r['best_params']}")
        print("  5-fold CV :", "  ".join(f"{m}={r['cv'][m][0]:.3f}+/-{r['cv'][m][1]:.3f}" for m in METRICS))
        print("  test set  :", "  ".join(f"{m}={r['test'][m]:.3f}" for m in METRICS))
        print("  confusion matrix [[TN FP] [FN TP]]:", r["confusion_matrix"].tolist())
