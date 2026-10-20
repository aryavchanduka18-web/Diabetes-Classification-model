# Study Guide: Diabetes Classification Mini Project (theory from zero)

Read this first. Then read `CODE_WALKTHROUGH.md`, which explains every file line by line.
Every number below was computed with this project's own code, so you can reproduce it.

---

## 0. The big picture in one page

```
diabetes.csv  (768 rows, 8 numbers per person + the answer)
     |
     v
split 80% train (614 rows)  /  20% test (154 rows, locked away)
     |
     v  inside every fold of 5-fold cross-validation
fix hidden zeros (median)  ->  standardise  ->  train a model  ->  score on the held-out fold
     |
     v
5 models x same folds: Logistic Regression, KNN, Decision Tree, Random Forest (+ scikit-learn benchmark)
     |
     v
pick settings by CV, refit on all 614 rows, score ONCE on the 154 test rows
     |
     v
results.json + model.npz  ->  Streamlit app shows them and predicts for new input
```

The goal of the whole project: given 8 measurements of a person, predict whether the dataset says
they have diabetes (1) or not (0), and **prove honestly how good and how comparable the models are**.

---

## 1. Machine learning basics

### 1.1 What "learning" means
Normal programming: you write the rules. Machine learning: you give examples (inputs with the right
answers) and the computer finds the rules itself.

- **Example / row / sample:** one person.
- **Features (X):** the numbers we know, e.g. Glucose, BMI. X is a table with n rows and d columns
  (here n = 768, d = 8).
- **Label / target (y):** the answer we want to predict. Here `Outcome`, 0 or 1.
- **Model:** a formula with adjustable numbers inside, called **parameters** (for logistic
  regression: the weights w and the bias b).
- **Training / fitting:** adjusting the parameters so predictions match the known answers.
- **Inference / prediction:** using the trained model on new data.

### 1.2 Supervised learning, classification, binary
- **Supervised:** training data has correct answers.
- **Classification:** the answer is a category. **Regression:** the answer is a number.
- **Binary classification:** exactly two categories (0 and 1). That is our problem.

### 1.3 Parameters vs hyper-parameters
- **Parameters** are learned from data (weights in logistic regression, split thresholds in a tree).
- **Hyper-parameters** are chosen by you before training (learning rate, k in KNN, tree depth,
  number of trees). We choose them with cross-validation.

### 1.4 Generalisation, overfitting, underfitting
We do not care how well the model remembers the training data. We care how well it does on **new**
people. This is **generalisation**.

- **Overfitting:** the model memorises noise. Training score is great, new-data score is worse.
  Example: a decision tree with unlimited depth classifies all training rows perfectly.
- **Underfitting:** the model is too simple to capture the pattern. Both scores are poor.

### 1.5 Bias and variance
- **Bias:** error from wrong assumptions (a straight line cannot draw a curvy boundary).
- **Variance:** error from being too sensitive to the particular training sample.
- Simple models: high bias, low variance. Flexible models (deep tree, KNN with k = 1): low bias,
  high variance. Good modelling balances the two. Random forests reduce variance by averaging.

### 1.6 Train / validation / test
- **Train:** used to fit parameters.
- **Validation (CV folds):** used to choose hyper-parameters and compare models.
- **Test:** touched once at the very end to give an honest estimate. If you tune on the test set it
  stops being honest.

---

## 2. The dataset

**Pima Indians Diabetes dataset** (originally from the US National Institute of Diabetes and
Digestive and Kidney Diseases): females, at least 21 years old (Age ranges 21 to 81), of Pima
Indian heritage. 768 rows.

| Column | Meaning |
|---|---|
| Pregnancies | number of times pregnant |
| Glucose | plasma glucose concentration (2 hours into an oral glucose tolerance test) |
| BloodPressure | diastolic blood pressure (mm Hg) |
| SkinThickness | triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | body mass index (weight in kg / height in m squared) |
| DiabetesPedigreeFunction | a score for diabetes family history |
| Age | years |
| Outcome | 1 = diabetes, 0 = no diabetes (the target) |

### 2.1 What the exploration page found
- **No duplicate rows** (checked in code).
- **Class imbalance:** 500 negative (65.1%), 268 positive (34.9%).
- **Accuracy trap:** always predicting "no diabetes" scores 65.1% accuracy and finds nobody.
  This is why we also use recall, F1 and ROC-AUC.
- **Hidden missing values:** the file has no empty cells, but zeros are impossible in five columns.
  Glucose 5, BloodPressure 35, SkinThickness 227 (29.6%), Insulin 374 (48.7%), BMI 11.
  These zeros mean "not measured". Pregnancies = 0 (111 rows) is legitimate.
- **Correlation with Outcome (after fixing zeros):** Glucose +0.49, BMI +0.31, Age +0.24,
  Pregnancies +0.22, SkinThickness +0.22, Insulin +0.20, Pedigree +0.17, BloodPressure +0.17.
  Correlation measures straight-line association, from -1 to +1. It is not causation.
- **Class means:** glucose 110.7 (no diabetes) vs 142.1 (diabetes); BMI 30.9 vs 35.4.

---

## 3. Preprocessing theory

### 3.1 Handling the hidden zeros: median imputation
Replace each impossible zero with the **median** of that column's real (non-zero) values in the
**training data**. The median is the middle value when sorted, so a few extreme values (Insulin
reaches 846) do not drag it like they drag the mean.

Example: column `[0, 120, 100, 0, 140]`. The non-zero values are `[100, 120, 140]`, median 120, so the
column becomes `[120, 120, 100, 120, 140]`.

Alternatives and why we did not use them: dropping rows would delete almost half the data (Insulin);
mean imputation is pulled by outliers. A limitation: filling hides that the value was missing.

### 3.2 Standardisation (z-score scaling)
For each column: `z = (x - mean) / std`. Result: mean 0, std 1.

Example: `[2, 4, 6]` has mean 4 and std 1.633, so z = `[-1.2247, 0, 1.2247]`.

Why: features live on different scales (Insulin up to 846, pedigree under 2.5).
- KNN uses distances, so big-scale features would dominate.
- Gradient descent converges faster and weights become comparable.
- Trees do not need it; we scale anyway so every model sees identical inputs.

The mean and std are computed from **training data only** and then applied to validation or test data.

### 3.3 Stratified splitting
A random split could by chance put too few positives in a fold. **Stratified** means each part keeps
the same class ratio (about 35% positive). Our code splits each class separately: class 0 has 500
rows, so 100 go to test; class 1 has 268 rows, so 54 go to test. Total: 614 train, 154 test.

### 3.4 Data leakage (the single most important idea in validation)
**Leakage** means information from validation or test data influences training, making scores look
better than reality. Leaks we avoid:
1. Computing medians or means on the whole dataset before splitting. (We compute them on the
   training part, inside every CV fold.)
2. Choosing hyper-parameters using the test set. (We use CV on training rows.)
3. Scoring the test set more than once and keeping the best. (We score it once.)

---

## 4. Logistic regression (the main model)

### 4.1 The problem with a straight line
Linear regression predicts any real number, including -3 or 42. A probability must be between 0 and 1.
Logistic regression fixes this by passing a linear score through a squashing function.

### 4.2 Step by step
1. **Linear score:** `z = w1*x1 + w2*x2 + ... + w8*x8 + b = w . x + b`.
   Each weight says how much its feature pushes the answer. b is the baseline (bias/intercept).
2. **Sigmoid:** `p = sigmoid(z) = 1 / (1 + e^(-z))`. This squeezes any z into (0, 1).
   - sigmoid(0) = 0.5; sigmoid(2) = 0.8808; sigmoid(-2) = 0.1192.
   - Big positive z gives p near 1; big negative z gives p near 0.
   - Handy fact: its slope is `p * (1 - p)`.
3. **Probability:** p = P(diabetes = 1 | x).
4. **Decision:** predict class 1 if `p >= 0.5` (the threshold, which can be changed).

### 4.3 Odds and log-odds (how to interpret weights)
- Odds = p / (1 - p). If p = 0.75, odds = 3 ("3 to 1").
- Log-odds = ln(odds). The model says: **log-odds = w . x + b**. So the linear score z *is* the log-odds.
- Increasing a feature by 1 unit adds its weight to the log-odds. `e^weight` is the **odds ratio**.
- Our features are standardised, so "1 unit" means **one standard deviation**, and weights are comparable.
  Balanced model weights: Glucose 1.22, BMI 0.64, Pregnancies 0.33, Pedigree 0.30, Age 0.17,
  BloodPressure -0.06, SkinThickness -0.01, Insulin -0.03. Glucose matters most.
  Example: one std more glucose multiplies the odds by `e^1.22` = about 3.4.

### 4.4 Decision boundary
`w . x + b = 0` is where p = 0.5. In 8 dimensions that is a flat surface (a hyperplane), so logistic
regression can only separate classes with a flat boundary. That is its main limitation.

### 4.5 Loss function: how wrong are we?
We need one number measuring how bad the current weights are. For one sample with true label y
and predicted probability p, the **log loss** is:

```
loss = -[ y * ln(p) + (1 - y) * ln(1 - p) ]
```
- If y = 1: loss = -ln(p). p = 0.99 gives 0.01 (good); p = 0.01 gives 4.6 (very bad).
- If y = 0: loss = -ln(1 - p). Same idea mirrored.
- Confident and wrong is punished heavily; unsure costs a moderate amount.
The overall loss J is the **average over all samples** (binary cross-entropy).

**Where it comes from (maximum likelihood):** the probability the model assigns to the true label is
`p^y * (1-p)^(1-y)`. Training picks weights that make the whole dataset as likely as possible. Taking
the negative log turns the product into the average of the loss above.

**Why not squared error?** With a sigmoid, squared error makes a bumpy (non-convex) surface and learns
slowly when confidently wrong. Log loss is convex for logistic regression: one global minimum, so
gradient descent cannot get stuck.

At the start (w = 0, b = 0) every p = 0.5 and the loss is ln 2 = 0.6931.

### 4.6 The gradient (derivation you can reproduce on paper)
We want the direction that reduces J fastest. Take one sample. With `p = sigmoid(z)`:

- `d loss / d p = -(y/p) + (1 - y)/(1 - p) = (p - y) / (p * (1 - p))`
- `d p / d z = p * (1 - p)`
- Chain rule: `d loss / d z = (p - y) / (p(1-p)) * p(1-p) = p - y`   <-- everything cancels
- `z = w . x + b`, so `d z / d w = x` and `d z / d b = 1`

Therefore, averaged over n samples:

```
dJ/dw = (1/n) * sum_i (p_i - y_i) * x_i     =  (1/n) * X^T (p - y)      (matrix form)
dJ/db = (1/n) * sum_i (p_i - y_i)
```
Meaning: error (p - y) times the feature. If a feature is high when the model over-predicts, its
weight is pushed down.

### 4.7 Gradient descent
Repeat many times: `w = w - alpha * dJ/dw`, `b = b - alpha * dJ/db`.
- **alpha = learning rate.** Too small: slow. Too big: overshoots and may diverge. We use 0.1.
- **Batch gradient descent:** each step uses all training rows.
- **Iterations:** 3000. The loss curve flattens near 0.49 (class-weighted model), meaning converged.

**Worked example (one step).** Data (one feature, already scaled): X = `[-1, 0, 2]`, y = `[0, 0, 1]`.
Start w = 0, b = 0.
- p = `[0.5, 0.5, 0.5]`, loss = 0.6931
- errors p - y = `[0.5, 0.5, -0.5]`
- dJ/dw = (`-1*0.5 + 0*0.5 + 2*(-0.5)`) / 3 = **-0.5**; dJ/db = (0.5 + 0.5 - 0.5)/3 = **0.1667**
- With alpha = 0.1: w = 0 - 0.1*(-0.5) = **0.05**; b = 0 - 0.1*0.1667 = **-0.0167**
- New p = `[0.4833, 0.4958, 0.5208]`: the third sample (true 1) moved up, the first two moved down.
  New loss = **0.6659**, lower than 0.6931. One step made the model better.

### 4.8 L2 regularisation (in the code, off by default)
Adds a penalty `(lambda / 2n) * sum(w^2)` to the loss to discourage huge weights and reduce
overfitting. Gradient gets an extra `(lambda / n) * w`. The bias is not penalised. We run with
lambda = 0.

### 4.9 Class weights ("balanced")
Rare positives are outnumbered 65 to 35, so a plain model leans toward "no diabetes". Class
weighting multiplies each sample's error by `n / (2 * n_class)`, making a positive count more.
Effect on the test set: recall rose from 0.537 to 0.685, precision fell from 0.744 to 0.649.
There is no free lunch: you trade false negatives for false positives.

### 4.10 Strengths and weaknesses
Strengths: fast, interpretable, gives probabilities, convex training. Weaknesses: only linear
boundaries, needs scaled inputs for gradient descent, cannot find feature interactions on its own.

---

## 5. K-Nearest Neighbours (KNN)

**Idea:** similar people have similar outcomes. To predict for a new person, find the k most similar
training people and take a vote.

1. Compute the distance to every training point. We use **Euclidean distance**:
   `d(a, b) = sqrt( sum_j (a_j - b_j)^2 )`.
2. Take the k nearest.
3. Probability of class 1 = fraction of those k that are positive. Predict 1 if that is at least 0.5.

**Example:** training points at 0, 1 (class 0) and 10, 11 (class 1); query 0.4 with k = 3. The three
nearest are 0, 1, 10 (labels 0, 0, 1), so p = 1/3 and the prediction is class 0. Query 9 with k = 1
has nearest neighbour 10 (class 1), so p = 1.

- **No training step** ("lazy learner"): it just stores the data. All the work happens at prediction time.
- **Choosing k:** small k (k = 1) follows noise: low bias, high variance. Large k over-smooths toward
  the majority class: high bias, low variance. We use odd k to avoid tied votes and pick k = 9 by CV.
- **Scaling is essential** because distances add up differences across features.
- **Curse of dimensionality:** in many dimensions all points look equally far apart. With 8 features it is fine.
- **Probabilities are coarse:** only multiples of 1/k, which lowers its ROC-AUC (0.797).
- **Cost:** each prediction compares against all training rows.

---

## 6. Decision tree (CART)

**Idea:** a flowchart of yes/no questions learned from data, e.g. "Is Glucose at most 127.5? If
yes go left, else right ... until a leaf gives the answer."

### 6.1 Parts
- **Root:** first question. **Internal node:** a question `feature <= threshold`.
  **Leaf:** final answer. The leaf stores the fraction of training positives that fell into it, which is the probability.
- **Depth:** how many questions deep the longest path is.

### 6.2 Gini impurity: how mixed is a node?
For a node whose positive fraction is p: `Gini = 2 * p * (1 - p)` (the two-class form of
`1 - p_pos^2 - p_neg^2`).
- Pure node (all one class): p = 0 or 1, Gini = 0.
- Perfectly mixed: p = 0.5, Gini = 0.5 (the maximum).
- A 30% positive node: 2 * 0.3 * 0.7 = 0.42.
(Entropy is another impurity measure; it behaves similarly.)

### 6.3 Choosing a split (greedy)
For every feature and every possible threshold, split the data in two and compute the **weighted
impurity** of the children: `(n_left/n) * Gini_left + (n_right/n) * Gini_right`.
The **gain** = parent Gini - weighted child Gini. Take the split with the biggest gain.
It is "greedy" because it picks the best split now without planning ahead.

**Efficient scan:** sort the feature once, then walk through it keeping a running count of positives
on the left. Thresholds are the midpoints between consecutive **distinct** values.

**Worked example:** values `[1, 2, 3, 4]`, labels `[0, 0, 1, 1]`. Parent: p = 0.5, Gini = 0.5.
Split at 2.5: left = `[0, 0]` (pure, Gini 0), right = `[1, 1]` (pure, Gini 0). Weighted child
Gini = 0. Gain = 0.5 - 0 = **0.5**, the best possible. The code finds exactly threshold 2.5, with
left leaf p = 0.0 and right leaf p = 1.0.

### 6.4 When to stop
A tree that never stops memorises the data. We stop when: the node is pure; depth reaches
`max_depth`; there are too few samples (`min_samples_split`); a split would leave a leaf smaller than
`min_samples_leaf`; or no split improves impurity. In our run: depth 8, minimum 5 samples per leaf.

### 6.5 Feature importance
Each split adds `(samples in node) * gain` to its feature's score; scores are normalised to sum to 1.
Features used often and early in big nodes get high scores.

### 6.6 Strengths and weaknesses
Strengths: easy to read, handles non-linear patterns and interactions, no scaling needed.
Weaknesses: high variance (small data changes give a different tree), overfits easily, blocky
boundaries. Our single tree has the weakest ROC-AUC (0.760).

### 6.7 A real tree from our data (for illustration)
A depth-2 tree (minimum 5 per leaf) fitted on all 768 rows after fixing the zeros, printed by
`export_text`:

```
if Glucose <= 127.500:
|   if Age <= 28.500:
|   |   -> p(diabetes)=0.08  (n=271)
|   else:
|   |   -> p(diabetes)=0.33  (n=214)
else:
|   if BMI <= 29.950:
|   |   -> p(diabetes)=0.32  (n=75)
|   else:
|   |   -> p(diabetes)=0.72  (n=208)
```
How to read it: low glucose and young age is the safest group (8% diabetic, 271 people); high glucose
and high BMI is the riskiest (72%, 208 people). Glucose does the first split (the strongest single
feature, matching logistic regression and the correlations). Importances for this tree: Glucose 0.66,
BMI 0.19, Age 0.15, all others 0. This is a teaching picture; the deployed tree is deeper (depth 8).

---

## 7. Random forest

**Idea:** one tree is unstable, but the average of many different trees is stable ("wisdom of crowds").

### 7.1 Two sources of randomness
1. **Bagging (bootstrap aggregating):** each tree trains on a **bootstrap sample**: n rows drawn
   *with replacement*. A given row is left out of a sample with probability `(1 - 1/n)^n`, about
   **0.368**, so each tree sees roughly 63% of the distinct rows.
2. **Random feature subsets:** at every split a tree may only consider `sqrt(8)` = 2 randomly chosen
   features. This stops all trees from copying the same strongest feature (glucose), so the trees
   are less correlated.

### 7.2 Prediction
Average the probabilities from all trees (100 of them, each depth 8, at least 3 samples per leaf).

### 7.3 Why it works (variance reduction)
If you average B predictions, each with variance s^2 and pairwise correlation r, the variance of the
average is `r*s^2 + (1 - r) * s^2 / B`. More trees shrink the second term, and random feature
subsets shrink r. Bias stays about the same. That is why our forest (ROC-AUC 0.828) beats a single
tree (0.760).

### 7.4 Not implemented (be honest in the viva)
Out-of-bag error estimates, pruning, categorical splits.

---

## 8. Evaluation metrics

### 8.1 Confusion matrix
Rows = truth, columns = prediction. Layout in our code: `[[TN, FP], [FN, TP]]`.

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actually 0** | TN (true negative) | FP (false positive, false alarm) |
| **Actually 1** | FN (false negative, missed case) | TP (true positive) |

### 8.2 Formulas
- **Accuracy** = (TP + TN) / all
- **Precision** = TP / (TP + FP): of those we flagged, how many were right?
- **Recall (sensitivity)** = TP / (TP + FN): of the real positives, how many did we catch?
- **F1** = 2 * precision * recall / (precision + recall): the harmonic mean. It is low if either is low,
  unlike a plain average.
- Specificity = TN / (TN + FP) (not reported, but good to know).

**Worked example:** truth `[1,1,1,0,0,0,0,1]`, predictions `[1,1,0,0,0,1,0,0]`.
TP = 2, TN = 3, FP = 1, FN = 2, so the matrix is `[[3,1],[2,2]]`.
Accuracy = 5/8 = 0.625; precision = 2/3 = 0.667; recall = 2/4 = 0.5; F1 = 0.571.

### 8.3 Which metric matters here?
For a screening-style problem a missed case (FN) is worse than a false alarm (FP), so **recall**
matters. Precision, F1 and accuracy complete the picture. Accuracy alone fools you on imbalanced data.

### 8.4 The threshold trade-off
The 0.5 cutoff is a choice. Lower it: catch more positives (recall up) but more false alarms
(precision down). Raise it: the reverse.

### 8.5 ROC curve and AUC
- Slide the threshold from high to low. At each threshold record **TPR** (= recall) and
  **FPR** = FP / (FP + TN). Plot TPR against FPR: that is the ROC curve.
- The diagonal is random guessing. A curve hugging the top-left is excellent.
- **AUC** = area under the curve. Meaning: the probability that a random positive gets a higher
  score than a random negative. 0.5 = chance, 1.0 = perfect. It does **not** depend on the threshold.
- Computation: sort samples by score; walk down the list, counting positives (TPR up) and negatives
  (FPR right). Area by the trapezoid rule.

**Worked example:** truth `[0,0,1,1]`, scores `[0.1,0.4,0.35,0.8]`. Sorted by score descending:
0.8 (pos), 0.4 (neg), 0.35 (pos), 0.1 (neg). The ROC points are
FPR `[0, 0, 0.5, 0.5, 1]`, TPR `[0, 0.5, 0.5, 1, 1]`. Area = 0.75. Check by pairs: of 4
(positive, negative) pairs, 3 rank the positive higher, and 3/4 = 0.75.

---

## 9. Validation theory

### 9.1 Why a single split is not enough
One train/validation split gives one number that depends on luck. We want a stable estimate and a
measure of how much it wobbles.

### 9.2 K-fold cross-validation
Split the training rows into k = 5 folds. Train on 4, validate on 1, repeat 5 times so each row is
validated exactly once. Report the **mean** and **standard deviation** across the 5 scores.
**Stratified** k-fold keeps the class ratio in every fold. Our code deals the rows of each class
round-robin into the 5 folds. Std uses n - 1 (sample standard deviation).

### 9.3 Preprocessing inside the fold
Medians and scaling are fit on the 4 training folds only, then applied to the validation fold.

### 9.4 Choosing hyper-parameters
We try k = 1,3,5,7,9,11,15,21,31 for KNN, depths 1, 2, 3, 4, 5, 6, 8, 10 for the tree, depths 3, 5, 8 for the forest,
and keep the best mean CV F1. Chosen: **k = 9, tree depth 8, forest depth 8.**
Caveats: (a) because we chose on the same folds we report, CV numbers are slightly optimistic
(**nested CV** would fix that); (b) the forest's best depth was the largest we tried, so a deeper one
might do slightly better.

### 9.5 The test set and its uncertainty
Scored once. It has only 154 rows, so the **standard error** of an accuracy near 0.76 is
`sqrt(0.76 * 0.24 / 154)` = **0.034**, about plus or minus 3.4 points. Gaps of 1 to 2 points mean nothing.

### 9.6 Are the models actually different?
Mean CV F1 ranges from 0.594 to 0.662 while the fold-to-fold std is 0.05 to 0.10. Every model is
within one std of the best. So the honest statement is: **comparable performance**, not "X wins".
Stronger tests would be repeated CV or a paired test on fold scores.

---

## 10. The scikit-learn benchmark (not our implementation)

Two library models are included only as an outside reference, run through the same folds:
- **Random Forest (scikit-learn):** same idea as ours. Ours scored 0.762 CV accuracy vs 0.744, which is
  a good sanity check that our code is right.
- **SVM with RBF kernel:** finds the boundary that maximises the **margin** between classes. The
  **kernel trick** (RBF) lets it draw curved boundaries by measuring similarity with a Gaussian bump
  `exp(-gamma * distance^2)`. `C` controls how much margin violation is tolerated. We use default
  settings and calibrate its scores into probabilities (Platt scaling) so we can compute AUC.

---

## 11. Architecture and the "backend"

There is **no separate server, database or API**. The structure is:

1. **Offline stage (scripts):** `train.py` and `compare.py` do all the computing and save files:
   `model.npz` (learned weights and scaling numbers) and `results.json` (all comparison results).
2. **Online stage (the app):** `app.py` is a Streamlit program. Streamlit runs Python on the server
   and draws the web page. It **re-runs the whole script from top to bottom every time you click
   something**; widgets just return their current values. Heavy loading is cached
   (`st.cache_resource` for the model, `st.cache_data` for the dataset).
3. **Files are the storage.** No data from users is saved.
4. **Deployment:** the code lives on GitHub; Streamlit Community Cloud installs `requirements.txt`
   and runs `app.py`.

The "backend logic" is the `ml/` package: preprocessing, models, metrics, cross-validation.

---

## 12. Results summary (what to say)

5-fold CV on 614 training rows (mean +/- std):

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.759 +/- 0.061 | 0.547 | 0.613 | 0.831 |
| Logistic Regression (balanced) | 0.749 +/- 0.064 | 0.701 | 0.662 | 0.830 |
| KNN (k = 9) | 0.756 +/- 0.043 | 0.607 | 0.634 | 0.797 |
| Decision Tree (depth 8) | 0.749 +/- 0.037 | 0.621 | 0.633 | 0.760 |
| Random Forest (ours) | 0.762 +/- 0.045 | 0.594 | 0.634 | 0.828 |
| Random Forest (scikit-learn) | 0.744 +/- 0.052 | 0.599 | 0.620 | 0.816 |
| SVM RBF (scikit-learn) | 0.751 +/- 0.044 | 0.533 | 0.600 | 0.820 |

Held-out test accuracy: LR 0.773, LR balanced 0.760, KNN 0.779, Tree 0.734, Forest 0.766.

**One-sentence conclusion:** all models reach about 75% accuracy and 0.80 to 0.83 AUC; differences are within noise;
class-weighted logistic regression gives the best recall; this is an educational demo, not a diagnosis.

---

## 13. Glossary and formula sheet

| Term | Meaning |
|---|---|
| Feature | an input column |
| Label | the correct answer (Outcome) |
| Epoch / iteration | one gradient descent update |
| Learning rate (alpha) | step size of gradient descent |
| Loss | a number measuring how wrong the model is |
| Convex | bowl-shaped loss with one global minimum |
| Threshold | cutoff turning probability into a class (0.5) |
| Stratified | keeping class ratios the same |
| Leakage | validation/test information sneaking into training |
| Bootstrap | sample n rows with replacement |
| Ensemble | combining many models |
| Hyper-parameter | a setting you choose (k, depth) |
| Calibration | making predicted probabilities match real frequencies |

```
sigmoid(z)     = 1 / (1 + e^-z)
log-odds       = ln(p / (1 - p)) = w . x + b
log loss       = -mean[ y ln p + (1-y) ln(1-p) ]
gradient       : dJ/dw = X^T (p - y) / n ,  dJ/db = mean(p - y)
update         : w = w - alpha * dJ/dw
z-score        = (x - mean) / std
Euclidean dist = sqrt( sum (a_j - b_j)^2 )
Gini           = 2 p (1 - p)
gain           = Gini_parent - (nL/n) Gini_L - (nR/n) Gini_R
accuracy       = (TP+TN)/all     precision = TP/(TP+FP)
recall         = TP/(TP+FN)      F1 = 2PR/(P+R)
test std error = sqrt( acc (1-acc) / n_test )
```
