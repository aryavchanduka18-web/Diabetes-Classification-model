# Diabetes Classification (ML lab mini project)

Binary classification on the Pima Indians Diabetes dataset (768 rows, 8 numeric features).
Three scikit-learn algorithms are compared: **Logistic Regression, KNN and Decision Tree**.
Educational demo, not a medical diagnosis.

## Run

```
python -m pip install -r requirements.txt
python model.py          # trains the 3 models and prints the results
streamlit run app.py     # opens the web app at http://localhost:8501
```

## Files

| File | What it does |
|---|---|
| `model.py` | All the machine learning (scikit-learn): data loading, Pipeline, tuning, evaluation |
| `app.py` | Streamlit web app: Home, Data exploration, Predict, Model comparison, About the models |
| `diabetes.csv` | The dataset |
| `requirements.txt` | Libraries (Streamlit Cloud reads this) |
| `docs/` | Study guide (theory) and code walkthrough, for viva preparation |

## Method

1. Stratified 80/20 split: 614 training rows, 154 test rows (`train_test_split`, `stratify=y`).
2. One `Pipeline` for every model: `SimpleImputer` (median, for the hidden zeros) then `StandardScaler` then the model.
   A pipeline re-learns the medians and scaling from the training rows only, so nothing leaks.
3. `GridSearchCV` with `StratifiedKFold(5)` tries a few settings per model (class weights, k, tree depth)
   and keeps the best mean F1.
4. The test set is scored once at the end.

## Results (from `python model.py`)

| Model | Best setting | CV F1 | CV recall | CV ROC-AUC | Test accuracy |
|---|---|---|---|---|---|
| Logistic Regression | class_weight=balanced | 0.672 +/- 0.013 | 0.701 | 0.844 | 0.734 |
| KNN | k = 15 | 0.632 +/- 0.048 | 0.571 | 0.832 | 0.740 |
| Decision Tree | max_depth = 5 | 0.573 +/- 0.066 | 0.537 | 0.765 | 0.773 |

Reading it honestly: Logistic Regression has the best cross-validated F1 and recall; KNN is comparable;
the tree is behind in CV even though it happens to score best on the small 154-row test set (luck).
Always predicting "no diabetes" would give 65.1% accuracy, so accuracy alone is not enough.

## Viva cheat sheet

- Why scale? KNN uses distances; logistic regression trains better.
- Why a Pipeline? It prevents data leakage (the median and scaling never see validation or test rows).
- Why recall? A missed diabetic case costs more than a false alarm.
- Why cross-validation? One split depends on luck; 5 folds give a mean and a spread.
- Why is the tree's test score higher but we do not call it best? The test set has only 154 rows (about +/-3.5 points).
