# Study Guide: Diabetes Classification (theory, scikit-learn version)

Read this first, then `CODE_WALKTHROUGH.md`. All numbers come from running `python model.py`.

---

## 0. The big picture

```
diabetes.csv (768 rows: 8 numbers per person + the answer)
   |
   v
split: 80% train (614 rows)  |  20% test (154 rows, locked away)
   |
   v   one scikit-learn Pipeline, used for every model
fill hidden zeros (median)  ->  standardise  ->  model
   |
   v   GridSearchCV + 5-fold cross-validation, on the 614 training rows only
try a few settings per model, keep the best mean F1
   |
   v
score each chosen model ONCE on the 154 test rows
   |
   v
Streamlit app: explore data, compare models, predict for new input
```
Goal: predict whether a person has diabetes (1) or not (0) from 8 measurements, and compare three
algorithms fairly.

---

## 1. Machine learning basics

- **Row / sample:** one person. **Features (X):** the known numbers (8 columns). **Label (y):** the answer, `Outcome` (0 or 1).
- **Supervised learning:** we learn from examples that have correct answers.
- **Classification** predicts a category; **regression** predicts a number. Ours is **binary classification**.
- **Training (fit):** the model adjusts internal numbers (parameters) to match known answers. **Prediction:** using the trained model on new rows.
- **Hyper-parameters:** settings you choose before training (k in KNN, tree depth, class weights). We choose them with cross-validation.
- **Overfitting:** the model memorises noise: great on training data, worse on new data (a deep tree does this).
  **Underfitting:** too simple to learn the pattern.
- **Bias and variance:** bias = error from too simple assumptions; variance = sensitivity to the particular training sample.
  Simple models have high bias and low variance; flexible models the reverse.
- **Train / validation / test:** train fits the model; validation (cross-validation folds) picks settings; the test set is used once at the
  end for an honest score. Tuning on the test set ruins it.

---

## 2. The dataset

Pima Indians Diabetes dataset (US National Institute of Diabetes and Digestive and Kidney Diseases): women at least 21 years old
of Pima Indian heritage. 768 rows.

| Column | Meaning |
|---|---|
| Pregnancies | number of pregnancies |
| Glucose | plasma glucose (2 h oral glucose tolerance test) |
| BloodPressure | diastolic blood pressure (mm Hg) |
| SkinThickness | triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | body mass index |
| DiabetesPedigreeFunction | family-history score |
| Age | years |
| Outcome | 1 = diabetes, 0 = no diabetes |

**Exploration findings**
- No duplicate rows.
- **Class imbalance:** 500 negative (65.1%), 268 positive (34.9%). Always predicting "no diabetes" scores 65.1% accuracy and finds
  nobody. That is why we also use recall, F1 and ROC-AUC.
- **Hidden missing values:** no empty cells, but zeros are impossible in five columns: Glucose 5, BloodPressure 35,
  SkinThickness 227 (29.6%), Insulin 374 (48.7%), BMI 11. Pregnancies = 0 is genuine.
- **Correlation with Outcome** (zeros ignored): Glucose +0.49, BMI +0.31, Age +0.24, Pregnancies +0.22. Correlation is from -1 to +1 and is not causation.

---

## 3. Preprocessing and the Pipeline

### 3.1 Median imputation (`SimpleImputer`)
Replace each impossible zero with the **median** of the real values in that column. The median is the middle value when sorted, so
extreme values (Insulin reaches 846) do not drag it. Example: `[0, 120, 100, 0, 140]`: the real values `[100, 120, 140]` have
median 120, so the column becomes `[120, 120, 100, 120, 140]`.
`SimpleImputer(missing_values=0, strategy="median")` ignores the zeros when computing the median, then fills them.

### 3.2 Standardisation (`StandardScaler`)
`z = (x - mean) / std` gives each column mean 0 and std 1. Example: `[2, 4, 6]` (mean 4, std 1.633) becomes `[-1.2247, 0, 1.2247]`.
Why: KNN uses distances, so large-scale columns (Insulin) would dominate; logistic regression trains better and its weights become comparable.

### 3.3 Stratified split (`train_test_split(stratify=y)`)
Keeps the share of diabetes cases (about 35%) the same in the train and test parts.

### 3.4 Data leakage and why we use a Pipeline
**Leakage** = information from validation or test rows influences training, which makes scores look better than they really are.
If we computed medians and scaling on the whole dataset first, the test rows would influence them. A scikit-learn `Pipeline` chains
"fill zeros, scale, model" so that every time it is fitted it learns the medians and the scaling from **its training rows only**, then just
applies them to other rows. Inside cross-validation, `GridSearchCV` re-fits the whole pipeline on each fold's training part, so there is no leak.

---

## 4. Logistic Regression (`LogisticRegression`)

**Idea:** turn a weighted sum of the features into a probability.

1. Score: `z = w1*x1 + ... + w8*x8 + b`.
2. Sigmoid: `p = 1 / (1 + e^(-z))`. It squeezes any number into (0, 1). sigmoid(0) = 0.5, sigmoid(2) = 0.8808, sigmoid(-2) = 0.1192.
3. Predict diabetes when `p >= 0.5` (the threshold).

**Odds and log-odds:** odds = p / (1 - p); the model says `log-odds = z`. A feature's weight is how much one unit of it adds to the
log-odds; `e^weight` is the odds ratio. Because we standardise, "one unit" = one standard deviation, so weights are comparable.
For this data Glucose has the largest weight, then BMI.

**Decision boundary:** `z = 0` is a flat surface, so logistic regression can only separate the classes with a flat boundary.

**Loss (what training minimises): log-loss** `-[y ln p + (1 - y) ln(1 - p)]` averaged over rows. Confident and wrong is punished heavily.
It comes from maximum likelihood and is convex, so there is one best solution.

**How it is trained.** The idea is gradient descent: nudge the weights in the direction that lowers the loss. For one row the gradient of the
loss with respect to the score is simply `p - y`, so `dJ/dw = (1/n) * X^T (p - y)`.
Tiny example (X = `[-1, 0, 2]`, y = `[0, 0, 1]`, start w = 0, b = 0, step 0.1): p = `[.5, .5, .5]`, loss 0.6931,
dJ/dw = -0.5, dJ/db = 0.1667 so w = 0.05, b = -0.0167; new loss 0.6659 (lower). scikit-learn does this with a faster optimiser (**L-BFGS**),
same loss, fewer steps. `max_iter=1000` sets the step limit.

**Regularisation:** scikit-learn adds an **L2 penalty** by default (`C=1.0`) that discourages huge weights and reduces overfitting.

**`class_weight="balanced"`:** gives each diabetes case a bigger weight (`n / (2 * n_class)`) so the rarer class is not ignored.
Effect: higher recall, lower precision. Our grid tries `None` and `"balanced"`; `"balanced"` won on cross-validated F1.

Strengths: fast, interpretable, probabilities. Weakness: only a flat boundary.

---

## 5. K-Nearest Neighbours (`KNeighborsClassifier`)

**Idea:** similar people have similar outcomes. To predict for a new person, find the **k** most similar training people and take a vote.
- Distance: Euclidean, `sqrt(sum (a_j - b_j)^2)` on the scaled features (scikit-learn default).
- Probability of diabetes = share of the k neighbours who have it.
- Example: training points 0 and 1 (class 0), 10 and 11 (class 1); query 0.4 with k = 3 has neighbours 0, 1, 10, so p = 1/3, class 0.
- **No training step** ("lazy"): it stores the data and does the work at prediction time.
- **Choosing k:** small k follows noise (low bias, high variance); large k over-smooths toward the majority class (high bias, low variance).
  We tried k = 1, 3, 5, 7, 9, 11, 15, 21, 31 and cross-validation picked **k = 15**.
- Needs scaling; probabilities are coarse (multiples of 1/k).

---

## 6. Decision Tree (`DecisionTreeClassifier`)

**Idea:** a flowchart of yes/no questions such as "Glucose <= 127.5?", learned from the data. A leaf gives the answer (and the probability = share of
training rows in that leaf that have diabetes).

- **Gini impurity** of a node with positive share p: `2p(1 - p)`. A pure node scores 0; a 50/50 node scores 0.5, the maximum.
- **How it picks a question:** try every feature and threshold; the best is the one that most reduces the size-weighted Gini of the two children
  (the "gain" = parent Gini minus weighted child Gini). It is greedy: it picks the best split now without planning ahead.
- **Worked example:** values `[1, 2, 3, 4]`, labels `[0, 0, 1, 1]`: parent Gini 0.5. Splitting at 2.5 gives two pure children (Gini 0), so the gain is 0.5, the best possible.
- **Controlling overfitting:** an unlimited tree memorises the training data. We limit `max_depth` (tried 1 to 10; cross-validation picked **5**) and set
  `min_samples_leaf=5` so no leaf is tiny.
- `random_state` makes ties between equally good splits repeatable.
- Strengths: easy to read, handles non-linear patterns, no scaling needed. Weakness: unstable and prone to overfitting; here it has the lowest ROC-AUC.

---

## 7. Evaluation metrics

**Confusion matrix** (`confusion_matrix`): `[[TN, FP], [FN, TP]]`.

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actually 0** | TN | FP (false alarm) |
| **Actually 1** | FN (missed case) | TP |

- **Accuracy** = (TP + TN) / all. **Precision** = TP / (TP + FP): of those flagged, how many are right.
  **Recall** = TP / (TP + FN): of the real cases, how many we catch. **F1** = 2PR / (P + R), the harmonic mean (low if either is low).
- Worked example: truth `[1,1,1,0,0,0,0,1]`, predictions `[1,1,0,0,0,1,0,0]`: matrix `[[3,1],[2,2]]`; accuracy 0.625; precision 0.667; recall 0.5; F1 0.571.
- **Which matters here?** For screening, a missed case (FN) is worse than a false alarm, so **recall** matters. Accuracy alone fools you on imbalanced data.
- **Threshold trade-off:** the 0.5 cutoff is a choice. Lower it: higher recall, lower precision. `class_weight="balanced"` has a similar effect.
- **ROC curve (`roc_curve`) and AUC (`roc_auc_score`):** slide the threshold; at each step plot TPR (= recall) against FPR (= FP / (FP + TN)).
  AUC is the area under the curve = the probability that a random positive gets a higher score than a random negative. 0.5 = chance, 1.0 = perfect;
  it does not depend on the 0.5 threshold. Example: truth `[0,0,1,1]`, scores `[.1,.4,.35,.8]`: AUC 0.75 (3 of the 4 positive/negative pairs are ranked correctly).

---

## 8. Validation

- **Why not one split?** One split gives one number that depends on luck.
- **K-fold cross-validation (`StratifiedKFold(5)`):** split the 614 training rows into 5 folds with the same class ratio; train on 4, validate on 1; repeat so every row is validated once.
  Report the **mean and std** of the 5 scores. The std shows how stable a model is.
- **`GridSearchCV`:** automates the tuning. For every setting in the grid it runs the 5-fold CV and records all five metrics. With `refit="f1"` it keeps the setting with the best mean F1 and
  re-trains it on all 614 training rows. (Its std is the population std over the 5 folds.)
- **Caveat:** we choose the setting using the same folds we report, so CV scores are slightly optimistic. The untouched test set is the unbiased check.
- **Test-set uncertainty:** with only 154 rows, the standard error of an accuracy near 0.75 is `sqrt(0.75 * 0.25 / 154)` = 0.035, about plus or minus 3.5 points. Gaps of 1 to 3 points mean nothing.

---

## 9. Results and what to say

| Model | Best setting | CV accuracy | CV recall | CV F1 | CV ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | class_weight=balanced | 0.762 +/- 0.011 | 0.701 | 0.672 +/- 0.013 | 0.844 |
| KNN | k = 15 | 0.769 +/- 0.033 | 0.571 | 0.632 +/- 0.048 | 0.832 |
| Decision Tree | max_depth = 5 | 0.725 +/- 0.031 | 0.537 | 0.573 +/- 0.066 | 0.765 |

Test set (154 rows, scored once): accuracy LR 0.734, KNN 0.740, Tree 0.773; recall LR 0.704, KNN 0.556, Tree 0.759; ROC-AUC LR 0.813, KNN 0.807, Tree 0.799.

**How to interpret:**
- **Logistic Regression** is best in cross-validation (F1, recall, AUC), and its std is small, so it is also the most stable.
- **KNN** is comparable: its F1 gap (0.04) is smaller than its fold-to-fold spread (0.05).
- **The tree** is clearly behind in CV (gap 0.10 is larger than its spread 0.07), **yet it scores best on the test set.** That is luck on a small test set, a good example of why we do
  not trust one small test split. The honest conclusion: logistic regression is the safest choice and the models are in the same range (accuracy 73% to 77%).
- Educational demo, not a medical diagnosis.

---

## 10. Architecture, the "backend" and deployment

- **No separate server, database or API.** Streamlit runs your Python on the server and draws the web page.
- `model.py` is the logic layer; `app.py` is the interface and imports it.
- **The app trains the models when it starts** (`st.cache_resource` keeps them in memory) instead of loading a saved model file. A saved scikit-learn model may not load under a
  different scikit-learn version, and the dataset is tiny, so training takes only a few seconds.
- **Streamlit re-runs the whole script on every click**; widgets return their current value.
- **Deployment:** code on GitHub; Streamlit Community Cloud installs `requirements.txt` and runs `app.py`.

---

## 11. Formula and function cheat sheet

```
sigmoid(z)   = 1 / (1 + e^-z)          log-odds = ln(p / (1 - p)) = w . x + b
log-loss     = -mean[ y ln p + (1-y) ln(1-p) ]
z-score      = (x - mean) / std        Euclidean = sqrt( sum (a_j - b_j)^2 )
Gini         = 2 p (1 - p)             gain = Gini_parent - (nL/n) Gini_L - (nR/n) Gini_R
accuracy = (TP+TN)/all   precision = TP/(TP+FP)   recall = TP/(TP+FN)   F1 = 2PR/(P+R)
test std error = sqrt( acc (1 - acc) / n_test )
```

| scikit-learn piece | Job |
|---|---|
| `train_test_split(stratify=y)` | 80/20 split keeping class ratio |
| `SimpleImputer(missing_values=0, strategy="median")` | fill the hidden zeros |
| `ColumnTransformer` | apply the imputer to selected columns only |
| `StandardScaler` | z-score scaling |
| `Pipeline` | chain steps, prevents leakage |
| `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier` | the three models |
| `StratifiedKFold(5)` | stratified cross-validation folds |
| `GridSearchCV` | try settings with CV, keep the best, refit |
| `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `roc_curve` | evaluation |
