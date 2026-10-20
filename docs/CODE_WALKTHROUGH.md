# Code Walkthrough: every file, line by line

Read `STUDY_GUIDE.md` first for the theory. Here we connect each idea to the exact code.
Run the commands yourself as you read: understanding comes from seeing it work.

---

## 0. Repo map and data flow

```
ML LAB/
  diabetes.csv         the data (768 rows)
  ml/                  the "backend logic" package (all NumPy, written by us)
    preprocess.py      load csv, impute zeros, scale, stratified split
    logreg.py          sigmoid, loss, logistic regression with gradient descent
    metrics.py         confusion matrix, precision, recall, F1, ROC, AUC
    knn.py             k-nearest neighbours
    tree.py            decision tree (CART, Gini)
    forest.py          random forest built from tree.py
    cv.py              stratified k-fold + cross_validate
    bench.py           scikit-learn RF + SVM (benchmark ONLY)
  train.py             trains logistic regression -> model.npz
  compare.py           runs CV + test for all models -> results.json
  app.py               Streamlit web app (the front end, also loads the model)
  tests.py             13 sanity tests
  model.npz            saved logistic regression (used by Predict page)
  model_plain.npz      plain (unweighted) logistic regression, kept for comparison
  results.json         saved comparison results (used by Model comparison page)
  requirements.txt     libraries needed (Streamlit Cloud reads this)
  README.md            run instructions
```

Who calls whom:
```
train.py  -> ml.preprocess, ml.logreg, ml.metrics          -> writes model.npz
compare.py-> ml.preprocess, ml.cv, ml.knn, ml.tree, ml.forest, ml.logreg, ml.metrics, ml.bench -> writes results.json
app.py    -> ml.preprocess, ml.logreg.sigmoid ; reads model.npz, results.json, diabetes.csv
tests.py  -> everything in ml/
```
`ml/cv.py` uses `preprocess` and `metrics`. `ml/forest.py` uses `tree.py`.

---

## 1. The Python and NumPy toolkit used everywhere

NumPy works on whole arrays at once ("vectorised"), which is far faster than Python loops.

| Idea | Example | Result |
|---|---|---|
| Array and shape | `X = np.array([[1,2],[3,4]])`, `X.shape` | `(2, 2)`: 2 rows, 2 columns |
| Column / row | `X[:, 0]`, `X[1]` | first column; second row |
| Arithmetic on all items | `X * 2`, `X + 1` | element-wise |
| Matrix product | `X @ w` | for X shape (n, d) and w shape (d,) gives n scores |
| Transpose | `X.T` | rows become columns |
| Boolean mask | `x[x > 0]` | only items where the test is True |
| `np.where(cond)` | `np.where(y == 1)[0]` | indices where the condition holds |
| Sum / mean / median / std | `x.mean()`, `X.mean(axis=0)` | `axis=0` = down each column |
| `np.unique(y)` | `[0, 1]` | distinct values |
| `np.argsort(x)` | `[2, 0, 1]` | indices that would sort x |
| `np.cumsum(x)` | `[1,2,3] -> [1,3,6]` | running total |
| Broadcasting | `X - mean` | subtracts a length-d vector from every row |
| Random generator | `rng = np.random.default_rng(42)` | reproducible randomness (same seed, same numbers) |
| `rng.permutation(a)` | shuffled copy | |
| `np.clip(p, eps, 1-eps)` | keeps values in range | avoids `log(0)` |
| `np.savez` / `np.load` | save/load several arrays in one file | `.npz` |

Python features used: functions and classes, `__init__` (constructor), `self`, f-strings, list
comprehension `[f(x) for x in xs]`, `lambda` (small unnamed function), dictionaries, `if __name__ ==
"__main__":` (run this block only when the file is executed directly, not imported), and
`argparse` (command-line options).

A **class** bundles data and functions. Our models all follow the same pattern:
```python
model = SomeModel(settings)   # create
model.fit(X, y)               # learn parameters from data
model.predict_proba(X)        # probabilities between 0 and 1
model.predict(X)              # classes 0 or 1
```
Because every model has `fit` and `predict_proba`, the cross-validation code can treat them identically.

---

## 2. `ml/preprocess.py`

```python
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ZERO_COLS = [FEATURES.index(c) for c in ZERO_IS_MISSING]
```
The fixed column order, the five columns where 0 means missing, and their positions (1, 2, 3, 4, 5).
Pregnancies (position 0) is deliberately excluded.

### `load_csv(path)`
```python
with open(path, newline="") as f:
    header = f.readline().strip().split(",")
data = np.genfromtxt(path, delimiter=",", skip_header=1)
...
X = data[:, [header.index(c) for c in FEATURES]]
y = data[:, header.index("Outcome")].astype(int)
```
Reads the header line to learn column names, loads the numbers (skipping the header), checks all
required columns exist (raises `ValueError` otherwise), then picks the feature columns **in our fixed
order** (so a shuffled CSV still works) and the Outcome column as integers. Returns `X` (768 x 8) and `y` (768).

### `stratified_split(X, y, test_size=0.2, seed=42)`
For each class separately: shuffle that class's row indices, send the first 20% to test and the rest to
train. Then shuffle train and test so classes are not clumped. Because it works per class, ratios are
preserved: class 0 gives 100 test / 400 train, class 1 gives 54 test / 214 train (`round(268 * 0.2) = 54`).
Same seed means the same split every run.

### `fit_medians(X_train)` / `impute(X, medians)`
```python
col = X_train[:, j]
medians[j] = float(np.median(col[col != 0]))
```
For each zero-as-missing column, the median of the non-zero training values. `impute` copies X and
replaces zeros in those columns with the stored medians. **Fit on training, apply to anything**: this is
how leakage is prevented.

### `fit_scaler(X_train)` / `scale(X, mean, std)`
`mean` and `std` per column from training data; `std[std == 0] = 1.0` avoids dividing by zero for a
constant column; `scale` returns `(X - mean) / std` (broadcasting across rows).

---

## 3. `ml/logreg.py`

### `sigmoid(z)` (numerically stable)
```python
out = np.empty_like(z)
pos = z >= 0
out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
ez = np.exp(z[~pos])
out[~pos] = ez / (1.0 + ez)
```
Mathematically the same as `1/(1+e^-z)`. A naive version computes `e^(-z)`, which **overflows** for very
negative z (e.g. `e^1000`). We use the normal formula for z at least 0, and the equivalent
`e^z / (1 + e^z)` for negative z, where `e^z` is tiny and safe.

### `binary_cross_entropy(y, p, w=None, l2=0.0, eps=1e-12)`
```python
p = np.clip(p, eps, 1 - eps)
loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
if w is not None and l2 > 0:
    loss += (l2 / (2 * len(y))) * np.sum(w ** 2)
```
Clip keeps `log` away from 0. Then the average log loss from the guide, plus the optional L2 penalty.

### `class LogisticRegressionScratch`
**Constructor** stores settings: `lr` (learning rate, 0.1), `n_iter` (3000), `l2` (0), `class_weight`
(None or "balanced"); `w`, `b`, `loss_history` start empty.

**`_gradients(X, y, sample_w)`** (the heart of the model):
```python
n = X.shape[0]
p = sigmoid(X @ self.w + self.b)          # predictions for all rows at once
err = (p - y) * sample_w                  # error per row, times its weight
dw = (X.T @ err) / n + (self.l2 / n) * self.w
db = np.sum(err) / n
return dw, db, p
```
This is exactly `dJ/dw = X^T (p - y) / n` and `dJ/db = mean(p - y)` from the derivation.
`X @ self.w` gives all n scores in one matrix product.

**`fit(X, y)`**:
```python
self.w = np.zeros(d); self.b = 0.0
sample_w = np.ones(n)
if self.class_weight == "balanced":
    n_pos = y.sum(); n_neg = n - n_pos
    sample_w = np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))
for _ in range(self.n_iter):
    dw, db, p = self._gradients(X, y, sample_w)
    self.w -= self.lr * dw
    self.b -= self.lr * db
    self.loss_history.append(binary_cross_entropy(y, p, self.w, self.l2))
```
Start at zero. Balanced weights give each positive weight `n / (2 * n_pos)` (bigger) and each negative
`n / (2 * n_neg)` (smaller). Then 3000 times: compute gradients, step downhill, record the loss.
`loss_history` is the curve plotted on the Visualizations page.

**`predict_proba(X)`** = `sigmoid(X @ w + b)`. **`predict(X, threshold=0.5)`** = probabilities at or above the threshold become 1.

**Trace the worked example:** with X = `[[-1],[0],[2]]`, y = `[0,0,1]`, w = 0: p = `[.5,.5,.5]`,
err = `[.5,.5,-.5]`, `X.T @ err` = -1.5, divided by 3 = -0.5 (dw), `sum(err)/3` = 0.1667 (db). After
the update w = 0.05, b = -0.0167, loss falls 0.6931 to 0.6659.

---

## 4. `ml/metrics.py`

### `confusion_matrix(y_true, y_pred)`
Four counts using boolean masks, e.g. `tp = np.sum((y_true == 1) & (y_pred == 1))`. Returns
`[[tn, fp], [fn, tp]]`.

### `accuracy / precision / recall / f1_score(cm)`
They take the matrix and apply the formulas. `_safe_div(a, b)` returns 0 instead of crashing when b is 0
(for example, no positive predictions at all).

### `roc_curve(y_true, scores)`
```python
order = np.argsort(-scores, kind="stable")      # highest score first
y_sorted = y_true[order]; s_sorted = scores[order]
tps = np.cumsum(y_sorted)                        # positives found so far
fps = np.cumsum(1 - y_sorted)                    # negatives found so far
distinct = np.where(np.diff(s_sorted))[0]        # where the score changes
idx = np.r_[distinct, len(y_sorted) - 1]         # last index of each distinct score
tpr = np.r_[0.0, tps[idx] / tps[-1]]
fpr = np.r_[0.0, fps[idx] / fps[-1]]
```
Imagine lowering the threshold one sample at a time. `cumsum` counts positives and negatives above the
threshold. Dividing by the totals (`tps[-1]`, `fps[-1]`) gives TPR and FPR. Only points where the score
changes are kept so **tied scores** are handled correctly (they must enter together). A (0, 0) start is added.

### `auc(fpr, tpr)`
`np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2)`: each slice is a trapezoid: width times average height.

### `evaluate(y_true, proba, threshold=0.5)`
Turns probabilities into classes, builds the matrix, and returns a dictionary with all five metrics plus the ROC points.
This single function is used by `train.py`, `cv.py` and `compare.py`.

---

## 5. `ml/knn.py`

```python
class KNNScratch:
    def fit(self, X, y):          # just store the data
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y).astype(float)

    def predict_proba(self, X):
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ self.X_.T + (self.X_ ** 2).sum(1)[None, :]
        d2 = np.maximum(d2, 0.0)
        idx = np.argpartition(d2, self.k - 1, axis=1)[:, : self.k]
        votes = self.y_[idx]
        ...
        return votes.mean(axis=1)
```
- **Distance trick:** `|a - b|^2 = |a|^2 - 2 a.b + |b|^2`. `(X**2).sum(1)[:, None]` is a column of
  squared lengths of the queries, `X @ self.X_.T` is every query dotted with every training point, and the last
  term is a row of squared lengths of the training points. Broadcasting adds them into a full
  (queries x training) distance table without Python loops. `np.maximum(d2, 0)` fixes tiny negative
  values from rounding.
- **`argpartition(d2, k-1, axis=1)[:, :k]`** finds the indices of the k smallest distances per row
  without fully sorting (faster than sort).
- `votes = self.y_[idx]` looks up their labels; the mean is the positive fraction = probability.
- The optional `weighted=True` mode weights votes by `1 / distance` (not used in the final comparison).

Trace: training `[0],[1],[10],[11]` labels `0,0,1,1`; query `0.4`, k = 3 gives distances `0.16, 0.36, 92.16, ...`,
nearest three are 0, 1, 10, labels `0,0,1`, so p = 1/3, class 0.

---

## 6. `ml/tree.py`

### Helper
`gini(pos, n)` returns `2p(1-p)` with `p = pos / n`.

### Constructor and `fit`
Stores `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` (used by the forest) and `seed`.
`fit` creates a random generator, an importances array, then calls `_build(X, y, 0)` to grow the tree
recursively; `root_` is the top node. At the end importances are normalised to sum to 1.

### `_best_split(X, y, pos)`: the clever part
```python
feats = self._rng.choice(d, m, replace=False) if m < d else np.arange(d)
parent = gini(pos, n)
for j in feats:
    order = np.argsort(X[:, j], kind="stable")
    xs, ys = X[order, j], y[order]
    left_n = np.arange(1, n)                  # split after sample i -> i+1 on the left
    left_pos = np.cumsum(ys)[:-1]
    right_n = n - left_n
    right_pos = pos - left_pos
    valid = (xs[1:] != xs[:-1]) & (left_n >= msl) & (right_n >= msl)
    pl, pr = left_pos / left_n, right_pos / right_n
    weighted = (left_n * 2 * pl * (1 - pl) + right_n * 2 * pr * (1 - pr)) / n
    weighted = np.where(valid, weighted, np.inf)
    i = int(np.argmin(weighted))
    gain = parent - weighted[i]
```
For one feature: sort the rows; there are n - 1 possible cut points (between neighbours). `left_n` = how
many rows go left for each cut, and `left_pos` (running positives count) gives the positives on the left;
the right side is "total minus left". All n - 1 candidate splits are evaluated **at once** with array math,
so there is no loop over thresholds. `valid` rejects cuts between identical values (no real threshold
exists) and cuts that would leave a leaf below `min_samples_leaf`; invalid cuts get infinite impurity so
`argmin` never selects them. The best cut has the lowest weighted child impurity, i.e. the highest gain.
The threshold is the midpoint `(xs[i] + xs[i+1]) / 2`. For a forest, `m < d` means only a random subset of features is tried.

### `_build(X, y, depth)`
```python
node = {"p": pos / n, "n": n}                 # every node knows its positive fraction
if depth >= self.max_depth or n < self.min_samples_split or pos in (0, n):
    return node                               # leaf
gain, j, t = self._best_split(X, y, pos)
if j is None: return node                     # no useful split: leaf
mask = X[:, j] <= t
self.importances_[j] += n * gain
node.update(j=j, t=t, left=self._build(X[mask], y[mask], depth+1),
                      right=self._build(X[~mask], y[~mask], depth+1))
```
A node is a dictionary. A leaf has just `p` and `n`. An internal node also has `j` (feature), `t`
(threshold), `left`, `right`. `pos in (0, n)` means the node is pure. The function calls itself on the left
and right subsets (**recursion**), stopping at the rules from the guide.

### `predict_proba` / `_route`
```python
def _route(self, node, X, idx, out):
    if "j" not in node:                       # leaf
        out[idx] = node["p"]; return
    go_left = X[idx, node["j"]] <= node["t"]
    self._route(node["left"], X, idx[go_left], out)
    self._route(node["right"], X, idx[~go_left], out)
```
Sends the whole group of rows down the tree together, splitting the index list at each node. At a leaf,
every row in that group receives the leaf's probability.

### Extras
`depth()` computes the tree height. `export_text(feature_names)` prints the rules like
`if Glucose <= 127.500: ...`, great for explaining in a viva.

**Trace:** `X=[1,2,3,4]`, `y=[0,0,1,1]`, depth 1: candidate cuts at 1.5, 2.5, 3.5 have weighted impurities
0.333, 0.0, 0.333, so the cut at 2.5 wins with gain 0.5.

---

## 7. `ml/forest.py`

```python
rng = np.random.default_rng(self.seed)
for _ in range(self.n_trees):
    boot = rng.integers(0, n, n)                          # bootstrap: n draws with replacement
    tree = DecisionTreeScratch(max_depth=..., min_samples_leaf=...,
                               max_features=self.max_features, seed=int(rng.integers(2**31))
                               ).fit(X[boot], y[boot])
    self.trees_.append(tree)
self.importances_ = np.mean([t.importances_ for t in self.trees_], axis=0)

def predict_proba(self, X):
    return np.mean([t.predict_proba(X) for t in self.trees_], axis=0)
```
`rng.integers(0, n, n)` returns n random row numbers, repeats allowed: that is a bootstrap sample.
`X[boot]` picks those rows. Each tree gets its own seed so its feature choices differ.
`max_features="sqrt"` makes each split consider only 2 random features out of 8. Prediction averages the
trees' probabilities. A fixed seed means the same forest every run.

---

## 8. `ml/cv.py`

### `stratified_kfold(y, k=5, seed=42)`
```python
fold_of = np.empty(len(y), dtype=int)
offset = 0
for cls in np.unique(y):
    idx = rng.permutation(np.where(y == cls)[0])
    fold_of[idx] = (np.arange(len(idx)) + offset) % k
    offset += len(idx)
return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(k)]
```
For each class, shuffle its rows and deal them like cards into folds 0, 1, 2, 3, 4, 0, 1 ... (`% k`).
`offset` continues the dealing so fold sizes stay balanced across classes. Returns k pairs of
(training indices = every row not in fold f, validation indices = rows in fold f). Every row is validated
exactly once, and each fold has about 35% positives.

### `preprocess_split(X_train, X_other)`
Fit medians and scaler on `X_train` only, apply to both. This is the leakage-safe step.

### `make_folds(X, y, k, seed)`
For each fold: preprocess using that fold's training part, and store `(X_train, y_train, X_val, y_val)`.
Built **once** and reused for every model, so all models see identical data.

### `cross_validate(factory, folds, threshold=0.5)`
```python
for Xt, yt, Xv, yv in folds:
    model = factory().fit(Xt, yt)                 # fresh model each fold
    m = evaluate(yv, model.predict_proba(Xv), threshold)
    rows.append([m[name] for name in METRICS])
arr = np.array(rows)                              # shape (5 folds, 5 metrics)
mean = arr[:, i].mean(); std = arr[:, i].std(ddof=1)
```
`factory` is a function that builds a new untrained model (a `lambda`), so each fold starts clean.
Returns for every metric the mean, the sample std (`ddof=1` means divide by n - 1) and the five fold values.

---

## 9. `ml/bench.py` (benchmark only)

`SkLearnModel` wraps a scikit-learn estimator so it has our `fit` and `predict_proba(X)` interface
(keeping only the positive-class column `[:, 1]`). `library_models()` imports scikit-learn **inside the
function** (so the rest of the project works without it) and returns two factories: a 100-tree
`RandomForestClassifier` and an RBF `SVC` wrapped in `CalibratedClassifierCV` to produce probabilities.
This is the **only** file that imports scikit-learn.

---

## 10. `train.py`

Command line options (`argparse`): `--csv`, `--lr`, `--iters`, `--l2`, `--balanced`, `--threshold`, `--seed`, `--out`.
Flow:
1. `X, y = pp.load_csv(...)`
2. `stratified_split` (80/20)
3. Medians **from train**, impute both; scaler **from train**, scale both
4. Build `LogisticRegressionScratch(...)` and `.fit`
5. `evaluate` on train and test, print them
6. `np.savez(args.out, w=..., b=..., mean=..., std=..., median_cols=..., median_vals=..., loss_history=..., fpr=..., tpr=..., cm=..., metrics=..., train_metrics=..., threshold=..., n_train=..., n_test=...)`

`model.npz` is the app's "brain": the weights plus every number the app needs to preprocess new
input the same way (medians, mean, std) plus the stats it displays. `python train.py --balanced` produced the
current `model.npz` (balanced); the unweighted version is saved as `model_plain.npz`.

---

## 11. `compare.py`

Flow (read `main()` top to bottom):
1. Load data, split 80/20 (same seed 42 as `train.py`, so the same test rows).
2. `folds = make_folds(X_tr, y_tr, 5, seed)`; `Xtr_s, Xte_s = preprocess_split(X_tr, X_te)` for the final test.
3. `tune(make, grid, folds)`: for each candidate value run `cross_validate`, keep the one with the best
   mean F1. Used for k (`K_GRID`), tree depth (`TREE_GRID`), forest depth (`FOREST_GRID`). It returns
   the best value and the table of all scores (shown in the app's expander).
4. A dictionary `models` maps a name to `(kind, params-text, factory)`. `kind` is `"scratch"` or `"library"`.
   The scikit-learn models are added inside `try/except ImportError`, so missing scikit-learn just skips them.
5. For each model: `cross_validate(factory, folds)` (CV scores) and `test_scores(...)` (refit on all 614
   rows, score once on the 154 test rows, with confusion matrix and ROC points).
6. `best_f1` = highest mean CV F1; `within` = every model whose mean F1 is within one std of it
   (this is why the app says "comparable").
7. Everything goes into one dictionary and `json.dump(out, f)` writes `results.json`.

---

## 12. `tests.py` (13 tests)

Run `python tests.py`. Each test is a function starting with `test_` using `assert`.
| Test | What it proves |
|---|---|
| sigmoid | `sigmoid(0)=0.5`; no overflow at +/-1000 |
| gradient_matches_numerical | our analytic gradient equals a finite-difference estimate (error under 1e-6) |
| learns_separable_data | accuracy over 95% on easy data; loss decreases |
| metrics_by_hand | matrix `[[3,1],[2,2]]`, precision 2/3, recall 1/2, F1 4/7 |
| auc_known_values | AUC 0.75, 1.0, and 0.5 for all-tied scores |
| preprocessing_no_leakage | median ignores zeros; stratified split keeps class ratio |
| evaluate_runs | `evaluate` returns perfect scores on a perfect example |
| tree_best_split_by_hand | threshold 2.5 on the toy data |
| tree_depth_limit_and_fit | `max_depth=2` is respected; an unlimited tree memorises training data |
| knn_by_hand | nearest-neighbour votes and p = 1/3 |
| forest_learns_and_is_reproducible | learns a pattern a straight line cannot; same seed gives identical output |
| stratified_kfold_properties | folds are disjoint, cover every row once, keep 30% positives |
| cv_preprocessing_is_per_fold | training folds have mean 0 after scaling while validation folds do not |

**Finite-difference gradient check** (a famous trick): nudge one weight by +/-0.000001, see how the loss
changes, compute `(loss+ - loss-) / (2 * 0.000001)`. If it equals our formula, the formula is right.

---

## 13. `app.py` (front end plus model use)

### Streamlit execution model
Streamlit runs the **entire file top to bottom** each time the page loads or a widget is used.
Widgets such as `st.radio`, `st.number_input`, `st.button` return their current value.

### Structure
1. **Imports and constants:** paths (`model.npz`, `results.json`, `diabetes.csv`), labels, defaults
   (your sample values: 2, 120, 80, 20, 80, 28.0, 0.5, 45), disclaimer text.
2. **`st.set_page_config`** sets the title and wide layout.
3. **Loading the model:**
   ```python
   @st.cache_resource
   def load_model(mtime): return dict(np.load(MODEL_PATH))
   M = load_model(os.path.getmtime(MODEL_PATH))
   ```
   `cache_resource` keeps the loaded file in memory between reruns. Passing the file's modification time
   makes the cache refresh when `model.npz` changes. If the file is missing, `st.error` plus `st.stop()` halts the page.
4. **Unpacking:** `medians` dictionary rebuilt from `median_cols` and `median_vals`; `thr` from `threshold`.
5. **Sidebar:** title, caption, `st.radio` to choose a page, disclaimer box. The chosen string decides which `if/elif` block runs.
6. **`prepare(raw)`**: the same pipeline as training: `impute` with saved medians, then `scale` with saved mean and std. **This is why `model.npz` stores them.**

### The pages
- **Home:** metrics from `M` (training rows, test rows, AUC).
- **Data exploration:** reads `diabetes.csv` (cached), computes duplicate count with `np.unique` on stacked rows,
  class balance bars, zero counts per column, histogram by class (zeros excluded for the five columns),
  class means table, correlation bars and heatmap with `np.corrcoef`, summary stats in an expander.
- **Predict:** eight `st.number_input` widgets; when **Predict** is clicked:
  ```python
  xs = prepare(vals)                                  # impute + scale the 8 numbers
  p = float(sigmoid(xs @ M["w"] + M["b"])[0])         # the logistic regression formula
  positive = p >= thr
  contrib = xs[0] * M["w"]                            # each feature's push on the log-odds
  ```
  Shows the class, probability %, a progress bar and a bar chart of contributions (red = pushes toward diabetes).
- **Model performance:** test metrics from `M["metrics"]`, with train values as captions.
- **Model comparison:** loads `results.json`; splits models into "ours" and "library"; builds tables with
  `mean +/- std`; bar chart with error bars (`yerr=std`); test table; ROC overlay; hyper-parameter tables; honesty notes.
- **Visualizations:** confusion matrix (`imshow`), ROC curve, loss curve (`M["loss_history"]`), learned weights bars.
- **About the model:** formulas rendered with `st.latex` and short descriptions of KNN, tree, forest.

Matplotlib makes the charts (`plt.subplots`, then `st.pyplot(fig)`). pandas is used only for tidy tables.

---

## 14. Data and artifact files

**`diabetes.csv`**: header `Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome`, then 768 rows like `6,148,72,35,0,33.6,0.627,50,1`.

**`model.npz`** (a zip of arrays): `w` (8 weights), `b`, `mean`, `std` (8 each), `median_cols`/`median_vals`
(five columns and their medians), `loss_history` (3000 values), `fpr`/`tpr` (ROC points), `cm` (2x2),
`metrics`/`train_metrics` (5 numbers each), `threshold`, `n_train`, `n_test`.

**`results.json`**:
```
{ "config": {seed, folds, n_train, n_test, tuning_metric, library_available, threshold},
  "grids":  {"KNN (k)": [rows], "Decision Tree (max_depth)": [...], "Random Forest (max_depth)": [...]},
  "best_cv_f1": "<model name>",
  "within_one_std_of_best_f1": ["<names>"],
  "models": { "<name>": { "kind": "scratch|library", "params": "...",
              "cv":   { "accuracy": {mean, std, folds[5]}, "precision": ..., "recall": ..., "f1": ..., "roc_auc": ... },
              "test": { accuracy, precision, recall, f1, roc_auc, cm, fpr, tpr } } } }
```

**`requirements.txt`**: numpy, pandas, matplotlib, `streamlit==1.66.0` (the tested version), scikit-learn.
Streamlit Cloud installs these. The app itself never imports scikit-learn; only `compare.py` does, via `ml/bench.py`.

---

## 15. Three execution traces

### (a) `python train.py --balanced`
load CSV -> stratified 80/20 -> medians from the 614 train rows -> impute train and test -> mean/std from
train -> scale both -> 3000 gradient steps with class weights (loss 0.693 down to about 0.494) -> evaluate
(test accuracy 0.760, recall 0.685, AUC 0.838) -> write `model.npz`.

### (b) `python compare.py` (about 15 seconds)
split -> 5 folds with per-fold preprocessing -> tune k, tree depth, forest depth by mean CV F1 -> for each
of 7 models run 5-fold CV and one test evaluation -> compute "within one std" -> write `results.json`.

### (c) Clicking Predict in the app, with the sample values
`vals = [2, 120, 80, 20, 80, 28.0, 0.5, 45]` -> `impute` (no zeros, nothing to replace) ->
`scale` using saved mean/std -> `z = xs @ w + b` -> `p = sigmoid(z)` = **0.325** -> 0.325 < 0.5 -> class 0,
"Not indicated" -> the bar chart shows glucose and BMI pushing the score and others near zero.
You can reproduce the 0.3251 in a Python shell by running the same lines.

---

## 16. Exercises (do these before the viva)

1. Run `python tests.py`, then break something on purpose (change `/ n` to `/ (n+1)` in `_gradients`) and watch which tests fail. Undo it.
2. In a Python shell: `from ml.logreg import sigmoid`; compute `sigmoid(np.array([-2., 0., 2.]))`.
3. By hand, compute one gradient step for X = `[[-1],[0],[2]]`, y = `[0,0,1]` and check against section 3.
4. Fit `DecisionTreeScratch(max_depth=3)` on the data and print `export_text(FEATURES)`. Explain the first split.
5. Change `k` in `KNNScratch` to 1 and 31 and compare train versus validation accuracy: see bias and variance.
6. Raise the threshold in `train.py` (`--threshold 0.7`) and watch precision up, recall down.
7. Explain aloud, without notes: "Why are medians computed inside each fold?"
8. Draw the pipeline diagram from memory and label where leakage could occur.
