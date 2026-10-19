"""Data loading and preprocessing, all fitted on the training split only (no leakage)."""
import numpy as np

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
# In the Pima dataset a value of 0 here is physiologically impossible => missing.
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ZERO_COLS = [FEATURES.index(c) for c in ZERO_IS_MISSING]


def load_csv(path):
    """Load the Pima CSV (header row + 8 features + Outcome) into X, y."""
    with open(path, newline="") as f:
        header = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    missing = [c for c in FEATURES + ["Outcome"] if c not in header]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    X = data[:, [header.index(c) for c in FEATURES]]
    y = data[:, header.index("Outcome")].astype(int)
    return X, y


def stratified_split(X, y, test_size=0.2, seed=42):
    """Split keeping the class ratio the same in train and test."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = rng.permutation(train_idx)
    test_idx = rng.permutation(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def fit_medians(X_train):
    """Median of each zero-as-missing column, ignoring the zeros."""
    medians = {}
    for j in ZERO_COLS:
        col = X_train[:, j]
        medians[j] = float(np.median(col[col != 0]))
    return medians


def impute(X, medians):
    X = X.copy()
    for j, m in medians.items():
        X[X[:, j] == 0, j] = m
    return X


def fit_scaler(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def scale(X, mean, std):
    return (X - mean) / std
