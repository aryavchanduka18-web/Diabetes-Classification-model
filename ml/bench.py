"""LIBRARY BENCHMARK ONLY. These are scikit-learn models, NOT our implementation.

scikit-learn is imported nowhere else in the project. They exist so the lab file can
compare our from-scratch models against well-known library baselines (optional, ungraded).
"""


class SkLearnModel:
    """Adapts a scikit-learn estimator to the fit / predict_proba interface used here."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)[:, 1]


def library_models():
    """name -> factory. Raises ImportError if scikit-learn is not installed."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    return {
        "Random Forest (scikit-learn)": lambda: SkLearnModel(
            RandomForestClassifier(n_estimators=100, random_state=42)),
        "SVM RBF (scikit-learn)": lambda: SkLearnModel(
            CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False)),   # Platt-scaled probabilities
    }
