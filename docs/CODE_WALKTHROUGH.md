# Code Walkthrough (scikit-learn version)

The whole project is two Python files plus the data. Read `STUDY_GUIDE.md` first for the theory.

```
model.py   all the machine learning (about 100 lines)
app.py     the Streamlit web app, calls model.py
diabetes.csv
```
`app.py` does `from model import train_all, ...`. `python model.py` runs the same training and prints a report.

---

## 1. `model.py`, top to bottom

### Imports
```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
```
Every scikit-learn piece, grouped by job: preprocessing (`SimpleImputer`, `StandardScaler`, `ColumnTransformer`),
models (`LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`), glue (`Pipeline`),
validation (`train_test_split`, `StratifiedKFold`, `GridSearchCV`), scoring (`sklearn.metrics`).

### Constants
```python
FEATURES = ["Pregnancies", "Glucose", ..., "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
SEED = 42
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
```
`ZERO_AS_MISSING`: the columns where 0 is impossible. `SEED` makes every random choice repeatable.

### `MODELS`: the three algorithms and what to tune
```python
MODELS = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {"model__class_weight": [None, "balanced"]}),
    "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31]}),
    "Decision Tree": (DecisionTreeClassifier(min_samples_leaf=5, random_state=SEED),
                      {"model__max_depth": [1, 2, 3, 4, 5, 6, 8, 10]}),
}
```
Each entry is `name: (model, grid)`. The grid lists the settings to try. `model__class_weight` means
"the `class_weight` of the step named `model` inside the pipeline" (double underscore reaches inside a step).
`max_iter=1000` lets the logistic regression optimiser run long enough to converge.

### `load_data`
Reads the CSV with pandas and returns `X` (the 8 feature columns) and `y` (the Outcome column).

### `make_pipeline(estimator)`
```python
fill_zeros = ColumnTransformer(
    [("median_fill", SimpleImputer(missing_values=0, strategy="median"), ZERO_AS_MISSING)],
    remainder="passthrough")
return Pipeline([("fill_zeros", fill_zeros), ("scale", StandardScaler()), ("model", estimator)])
```
- `SimpleImputer(missing_values=0, strategy="median")` treats 0 as "missing" and fills it with the median of
  the non-zero values it saw while fitting.
- `ColumnTransformer` applies that imputer **only to the five columns**; `remainder="passthrough"` leaves
  Pregnancies, DiabetesPedigreeFunction and Age untouched (a 0 there is real).
- `StandardScaler` standardises every column: `(x - mean) / std`.
- `Pipeline` chains the steps: `fit` learns each step from the training rows in order; `predict` applies the same
  steps to new rows automatically. **This is what prevents data leakage**, including inside cross-validation, because
  `GridSearchCV` re-fits the whole pipeline on each fold's training part.

### `logistic_coefficients(pipeline)`
Returns the fitted logistic regression weights as a labelled pandas Series. `ColumnTransformer` puts the five transformed
columns first and the three passthrough columns after, so the names are listed in that order.

### `train_all`: the main function
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
```
80/20 split that keeps the diabetes share equal in both parts (`stratify=y`), then a 5-fold splitter that does the same per fold.

```python
for name, (estimator, grid) in MODELS.items():
    search = GridSearchCV(make_pipeline(estimator), grid, cv=cv, scoring=METRICS, refit="f1")
    search.fit(X_train, y_train)
```
For each model, `GridSearchCV` tries every setting in its grid. For each setting it runs 5-fold CV and records all five metrics
(`scoring=METRICS`). `refit="f1"` means: pick the setting with the best mean F1, then re-train that one on all 614 training rows.
`search` then behaves like the final fitted model (`search.predict`, `search.best_estimator_`).

```python
scores, best = search.cv_results_, search.best_index_
"cv": {m: (scores[f"mean_test_{m}"][best], scores[f"std_test_{m}"][best]) for m in METRICS}
```
`cv_results_` is a table with one row per setting; `best_index_` is the winning row. We read its mean and std for each metric
(std is the population std over the 5 folds).

```python
proba = search.predict_proba(X_test)[:, 1]     # probability of diabetes
pred  = search.predict(X_test)                 # class 0/1 (threshold 0.5)
fpr, tpr, _ = roc_curve(y_test, proba)
```
Final, one-time scoring on the untouched test rows, then the metric functions (`accuracy_score`, `precision_score`,
`recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`). The result dictionary keeps the fitted model, best setting,
CV scores, test scores, confusion matrix, ROC points and the full grid table for the app.

### The `if __name__ == "__main__":` block
Runs `train_all()` and prints, per model: best setting, CV mean +/- std, test scores, confusion matrix.
This block runs only when you execute `python model.py`, not when `app.py` imports the file.

---

## 2. `app.py`

Streamlit re-runs the whole file top to bottom on every click; widgets return their current value.

1. **Imports and constants:** labels for the 8 inputs and your sample values (2, 120, 80, 20, 80, 28.0, 0.5, 45).
2. **`get_results()`** with `@st.cache_resource`: calls `train_all` once and keeps the result in memory, so clicking around does
   not retrain. (Training takes a few seconds.) **Why not save a model file?** A file saved with one scikit-learn version may not load
   under another; training at start-up avoids that, and the dataset is tiny.
3. **`get_data()`** with `@st.cache_data`: loads the CSV for the exploration page.
4. **Sidebar:** `st.sidebar.radio` picks the page; the `if / elif` chain draws that page.
5. **Home:** row counts and test ROC-AUC per model.
6. **Data exploration:** duplicates (`df.duplicated().sum()`), class balance (`value_counts`), hidden zeros
   (`(df[cols] == 0).sum()`), a histogram per class, class means (`groupby("Outcome").mean()`), correlations (`df.corr()`),
   after replacing the five columns' zeros with NaN so they are ignored.
7. **Predict:** eight `st.number_input` boxes; on **Predict**, builds a one-row DataFrame and calls
   `model.predict_proba(row)[0, 1]` for each of the three models. The pipeline inside each model fills zeros and scales the row by itself.
8. **Model comparison:** tables of `mean ± std`, a bar chart with error bars (`yerr`), the rule-of-thumb note (gap to the best F1 versus the
   fold-to-fold spread), test table, ROC curves, logistic regression weights, confusion matrices, and the search tables.
9. **About the models:** short theory, and which scikit-learn class does what.

There is no separate backend: Streamlit runs this Python on the server, and `model.py` is the "logic layer".

---

## 3. What happens when you run it

### `python model.py`
load CSV -> `train_test_split` (614/154) -> for each of 3 models: `GridSearchCV` runs 5-fold CV for every setting (each fold: the
pipeline fills zeros and scales using only that fold's training rows, trains, scores the held-out fold) -> best F1 setting re-fitted
on all 614 rows -> scored once on the 154 test rows -> print.

### Clicking Predict
the 8 numbers -> one-row DataFrame -> for each model `predict_proba`: [fill zeros with stored medians] -> [scale with stored mean/std] ->
[model computes probability] -> table with the probability and the class.

---

## 4. Exercises

1. In a Python shell: `from model import train_all; r, info = train_all()`, then `r["KNN"]["best_params"]`.
2. Print the logistic regression weights: `from model import logistic_coefficients; logistic_coefficients(r["Logistic Regression"]["model"])`.
3. Look inside the pipeline: `r["KNN"]["model"].named_steps` and `r["KNN"]["model"].named_steps["scale"].mean_`.
4. Change `[1, 3, 5, ...]` in the KNN grid to `[1]` and see how CV F1 changes (overfitting).
5. Remove `"scale"` from `make_pipeline` and compare KNN's score: scaling matters for distances.
6. Explain aloud without notes why a `Pipeline` prevents data leakage.
