# Diabetes Classification: model comparison (ML lab mini project)

Binary classification on the Pima Indians Diabetes dataset. Five models are written
from scratch with NumPy: logistic regression, KNN, decision tree, random forest, plus
the metrics, preprocessing and cross-validation engine. scikit-learn appears in exactly
one file, `ml/bench.py`, as a clearly labelled "library benchmark" (RF and RBF-SVM).
It is never presented as our implementation. Educational demo, not a medical diagnosis.

## Run

1. `diabetes.csv` in this folder. Columns: Pregnancies, Glucose, BloodPressure,
   SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome.
2. `python -m pip install -r requirements.txt`
3. `python tests.py`     13 sanity tests (gradient check, hand-checked tree/KNN/metrics, CV leakage)
4. `python train.py --balanced`   trains the logistic regression the Predict page uses -> model.npz
5. `python compare.py`   5-fold CV + held-out test for all models -> results.json (~15 s)
6. `streamlit run app.py`

## Layout

| File | What |
|---|---|
| `ml/logreg.py` | sigmoid, log-loss, gradient descent |
| `ml/knn.py` | k-nearest neighbours |
| `ml/tree.py` | CART decision tree (Gini), feature importances, text export |
| `ml/forest.py` | random forest: bootstrap + random feature subsets |
| `ml/metrics.py` | confusion matrix, precision, recall, F1, ROC, AUC |
| `ml/preprocess.py` | CSV load, zero-as-missing imputation, scaler, stratified split |
| `ml/cv.py` | stratified k-fold, preprocessing refit inside each fold |
| `ml/bench.py` | scikit-learn RF + SVM (benchmark only) |
| `train.py`, `compare.py`, `app.py`, `tests.py` | scripts, UI, tests |

## Evaluation protocol (say this in the viva)

1. Stratified 80/20 split; the 20% test set is not touched until the end.
2. Stratified 5-fold CV on the 80%. Median imputation and scaling are fitted on the training
   part of each fold only (no leakage).
3. KNN's k and the tree/forest depth are chosen by mean CV F1.
4. Refit on the full 80%, score once on the test set.

Honest reading of the results: all models land at roughly 74-76% CV accuracy and every model's
mean F1 is within one std of the best. The gaps are smaller than fold-to-fold noise, so the
right claim is "comparable", not "X is best". Hyper-parameters were tuned on the same folds
that are reported, so CV numbers are slightly optimistic; the test set is the unbiased check.

## Suggested split for a group of 3

- Member A: logistic regression, metrics, preprocessing, Streamlit app
- Member B: KNN, cross-validation engine, comparison script
- Member C: decision tree, random forest, tests

Everyone should still be able to explain the whole pipeline: viva questions can go anywhere.

## Viva cheat sheet

- Why standardise? KNN is distance-based and gradient descent converges faster.
- Why impute inside each fold? Otherwise validation data influences the training statistics.
- Gini impurity? 2p(1-p). The tree picks the split with the largest impurity decrease.
- Why does a forest beat a single tree? Bagging averages out the variance of deep trees.
- Why recall? A missed diabetic case (false negative) costs more than a false alarm.
- Gradient: dJ/dw = X^T (p - y) / n.
