"""Streamlit front end for the from-scratch diabetes classifier.  Run: streamlit run app.py"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ml import preprocess as pp
from ml.logreg import sigmoid

MODEL_PATH = "model.npz"
RESULTS_PATH = "results.json"
DATA_PATH = "diabetes.csv"
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1-score", "roc_auc": "ROC-AUC"}
STUDENT = "Aryav_2427010214"
LABELS = ["Pregnancies", "Glucose (mg/dL)", "Blood pressure (mmHg)", "Skin thickness (mm)",
          "Insulin (mu U/ml)", "BMI", "Diabetes pedigree function", "Age (years)"]
DEFAULTS = [2, 120, 80, 20, 80, 28.0, 0.5, 45]  # sample values only
DISCLAIMER = "Educational ML demo. Not a medical diagnosis."

st.set_page_config(page_title="GlucoPredict", page_icon=":material/glucose:", layout="wide")


@st.cache_resource
def load_model(mtime):
    return dict(np.load(MODEL_PATH))


if not os.path.exists(MODEL_PATH):
    st.error("model.npz not found. Put diabetes.csv here and run `python train.py` first.")
    st.stop()

M = load_model(os.path.getmtime(MODEL_PATH))
medians = dict(zip(M["median_cols"].astype(int).tolist(), M["median_vals"].tolist()))
thr = float(M["threshold"])

st.sidebar.title("GlucoPredict")
st.sidebar.caption(f"ML mini project · {STUDENT}")
page = st.sidebar.radio("Go to", ["Home", "Data exploration", "Predict", "Model performance",
                                  "Model comparison", "Visualizations", "About the model"])
st.sidebar.info(DISCLAIMER)


def prepare(raw):
    x = pp.impute(np.array([raw], dtype=float), medians)
    return pp.scale(x, M["mean"], M["std"])


if page == "Home":
    st.title("Diabetes classification with Logistic Regression")
    st.write("Binary classification on the Pima Indians Diabetes dataset (8 numeric features). "
             "Logistic regression, preprocessing and all metrics are implemented from scratch with NumPy.")
    c = st.columns(3)
    c[0].metric("Training rows", int(M["n_train"]))
    c[1].metric("Test rows", int(M["n_test"]))
    c[2].metric("Test ROC-AUC", f"{M['metrics'][4]:.3f}")
    st.caption(DISCLAIMER)

elif page == "Data exploration":
    st.header("Data exploration")
    if not os.path.exists(DATA_PATH):
        st.info("diabetes.csv not found next to app.py.")
        st.stop()

    @st.cache_data
    def load_data(mtime):
        return pp.load_csv(DATA_PATH)

    X, y = load_data(os.path.getmtime(DATA_PATH))
    n = len(y)
    n_dup = n - len(np.unique(np.c_[X, y], axis=0))
    c = st.columns(4)
    c[0].metric("Rows", n)
    c[1].metric("Features", X.shape[1])
    c[2].metric("Diabetes-positive", f"{y.mean():.1%}")
    c[3].metric("Duplicate rows", n_dup)

    st.subheader("1. Class balance")
    col1, col2 = st.columns([1, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(3.2, 3))
        counts = [int((y == 0).sum()), int((y == 1).sum())]
        ax.bar(["No diabetes (0)", "Diabetes (1)"], counts, color=["#2471a3", "#c0392b"])
        for i, v in enumerate(counts):
            ax.text(i, v + 5, str(v), ha="center")
        ax.set_ylabel("Rows")
        fig.tight_layout()
        st.pyplot(fig)
    with col2:
        st.markdown(
            f"The classes are imbalanced ({counts[0]} vs {counts[1]}). A model that **always predicts "
            f"'no diabetes' scores {counts[0] / n:.1%} accuracy** while catching zero patients. That is why "
            "we report recall, F1 and ROC-AUC next to accuracy, split with stratification, and offer a "
            "class-weighted logistic regression.")

    st.subheader("2. Hidden missing values: zeros that are not real")
    st.markdown("There are no empty cells and no duplicate rows, but in five columns a **0 is physiologically "
                "impossible** (a living person cannot have 0 blood pressure or BMI). These zeros are missing "
                "values in disguise.")
    miss = pd.DataFrame({
        "Column": pp.ZERO_IS_MISSING,
        "Zeros": [int((X[:, pp.FEATURES.index(c)] == 0).sum()) for c in pp.ZERO_IS_MISSING],
    })
    miss["% of rows"] = (miss["Zeros"] / n * 100).round(1)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(miss.set_index("Column"), use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 2.8))
        ax.barh(miss["Column"], miss["% of rows"], color="#d68910")
        ax.set_xlabel("% of rows that are 0")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
    st.caption("Fix used: replace them with the median of the training data (fitted inside each CV fold "
               "so no information leaks). Pregnancies = 0 is legitimate and left alone.")

    st.subheader("3. How each feature differs between the classes")
    feat = st.selectbox("Feature", pp.FEATURES, index=1)
    j = pp.FEATURES.index(feat)
    col = X[:, j]
    keep = col != 0 if feat in pp.ZERO_IS_MISSING else np.ones(n, dtype=bool)
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.histogram_bin_edges(col[keep], bins=25)
    ax.hist(col[keep & (y == 0)], bins=bins, alpha=0.6, color="#2471a3", label="No diabetes")
    ax.hist(col[keep & (y == 1)], bins=bins, alpha=0.6, color="#c0392b", label="Diabetes")
    ax.set_xlabel(feat)
    ax.set_ylabel("Rows")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    if feat in pp.ZERO_IS_MISSING:
        st.caption(f"{int((~keep).sum())} disguised-missing zeros excluded from this histogram.")

    Xi = pp.impute(X, pp.fit_medians(X))
    st.markdown("**Class means** (after median imputation):")
    means = pd.DataFrame({"No diabetes": Xi[y == 0].mean(axis=0), "Diabetes": Xi[y == 1].mean(axis=0)},
                         index=pp.FEATURES).round(2)
    means["Difference"] = (means["Diabetes"] - means["No diabetes"]).round(2)
    st.dataframe(means, use_container_width=True)

    st.subheader("4. Which features relate to the outcome?")
    corr_y = np.array([np.corrcoef(Xi[:, k], y)[0, 1] for k in range(8)])
    order = np.argsort(corr_y)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.barh(np.array(pp.FEATURES)[order], corr_y[order], color="#2471a3")
        ax.set_xlabel("Pearson correlation with Outcome")
        fig.tight_layout()
        st.pyplot(fig)
    with col2:
        M = np.corrcoef(np.c_[Xi, y].T)
        labels = [f[:8] for f in pp.FEATURES] + ["Outcome"]
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(9), labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(9), labels, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Feature correlation matrix", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
    top = [pp.FEATURES[k] for k in np.argsort(-np.abs(corr_y))[:2]]
    st.caption(f"{top[0]} and {top[1]} correlate most with the outcome, matching the weights the logistic "
               "regression learned. Correlation is not causation, and all correlations here are modest.")

    with st.expander("Summary statistics (raw data)"):
        st.dataframe(pd.DataFrame(X, columns=pp.FEATURES).describe().T.round(2), use_container_width=True)

elif page == "Predict":
    st.header("Enter measurements")
    st.caption("A value of 0 in Glucose, Blood pressure, Skin thickness, Insulin or BMI is treated as "
               "missing and replaced by the training median, as in training.")
    cols = st.columns(4)
    vals = []
    for i, (lab, d) in enumerate(zip(LABELS, DEFAULTS)):
        with cols[i % 4]:
            if isinstance(d, float):
                vals.append(st.number_input(lab, min_value=0.0, value=float(d), step=0.1, key=lab))
            else:
                vals.append(st.number_input(lab, min_value=0, value=int(d), step=1, key=lab))
    if st.button("Predict", type="primary"):
        xs = prepare(vals)
        p = float(sigmoid(xs @ M["w"] + M["b"])[0])
        positive = p >= thr
        c1, c2 = st.columns(2)
        c1.metric("Predicted class", "1 · Diabetes indicated" if positive else "0 · Not indicated")
        c2.metric("Estimated probability of class 1", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        contrib = (xs[0] * M["w"])
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(pp.FEATURES, contrib, color=["#c0392b" if v > 0 else "#2471a3" for v in contrib])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("contribution to log-odds (w × scaled x)")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(DISCLAIMER)

elif page == "Model performance":
    st.header("Test-set performance")
    names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    cols = st.columns(5)
    for c, n, v, t in zip(cols, names, M["metrics"], M["train_metrics"]):
        c.metric(n, f"{v:.3f}")
        c.caption(f"train: {t:.3f}")
    st.caption(f"Decision threshold = {thr}. Recall matters most for screening: a false negative is a missed case.")

elif page == "Model comparison":
    st.header("Model comparison")
    if not os.path.exists(RESULTS_PATH):
        st.info("results.json not found. Run `python compare.py` first.")
        st.stop()
    with open(RESULTS_PATH) as f:
        R = json.load(f)
    cfg, models = R["config"], R["models"]
    scratch = [n for n, r in models.items() if r["kind"] == "scratch"]
    library = [n for n, r in models.items() if r["kind"] == "library"]

    st.markdown(
        f"All models use the **same stratified {cfg['folds']}-fold cross-validation** on the "
        f"{cfg['n_train']} training rows (imputation and scaling are refit inside every fold). "
        f"The {cfg['n_test']}-row test set is untouched until the final scoring. "
        f"KNN's k, the tree depth and the forest depth are chosen by mean CV {cfg['tuning_metric'].upper()}.")
    st.success("**Our implementation (from scratch, NumPy only):** " + ", ".join(scratch))
    if library:
        st.warning("**Library benchmark: scikit-learn, NOT our implementation:** " + ", ".join(library)
                   + ". Shown only as an outside reference, with default untuned settings.")

    def cv_table(names):
        rows = {n: {METRIC_LABELS[m]: f"{models[n]['cv'][m]['mean']:.3f} ± {models[n]['cv'][m]['std']:.3f}"
                    for m in METRIC_KEYS} for n in names}
        return pd.DataFrame(rows).T

    st.subheader("Cross-validation: mean ± std over folds")
    st.caption("Our implementation")
    st.dataframe(cv_table(scratch), use_container_width=True)
    if library:
        st.caption("Library benchmark (scikit-learn)")
        st.dataframe(cv_table(library), use_container_width=True)

    metric = st.selectbox("Chart metric", METRIC_KEYS, index=3, format_func=METRIC_LABELS.get)
    names = scratch + library
    means = [models[n]["cv"][metric]["mean"] for n in names]
    stds = [models[n]["cv"][metric]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = ["#2471a3" if n in scratch else "#aab7b8" for n in names]
    ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(range(len(names)), names, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"CV {METRIC_LABELS[metric]} (blue = ours, grey = scikit-learn benchmark; bars = ±1 std)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    st.pyplot(fig)

    best = R["best_cv_f1"]
    within = R["within_one_std_of_best_f1"]
    st.info(
        f"Highest mean CV F1: **{best}**. Models within one std of it: {', '.join(within)}. "
        "When the gaps are smaller than the fold-to-fold spread, the ranking is not statistically "
        "meaningful: say \"comparable\", not \"best\".")

    st.subheader("Held-out test set (scored once)")
    trows = {n: {METRIC_LABELS[m]: round(models[n]["test"][m], 3) for m in METRIC_KEYS} for n in names}
    st.dataframe(pd.DataFrame(trows).T, use_container_width=True)
    st.caption(f"Only {cfg['n_test']} test rows, so one accuracy standard error is about ±3.5 points.")

    fig, ax = plt.subplots(figsize=(5, 4))
    for n in names:
        t = models[n]["test"]
        ax.plot(t["fpr"], t["tpr"], "--" if n in library else "-", label=f"{n} ({t['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC curves (dashed = library benchmark)")
    ax.legend(fontsize=6)
    st.pyplot(fig)

    with st.expander("Hyper-parameter search (mean CV scores)"):
        for title, rows in R["grids"].items():
            df = pd.DataFrame(rows).rename(columns=METRIC_LABELS).set_index("param").round(3)
            st.caption(title)
            st.dataframe(df, use_container_width=True)
    with st.expander("How to read this honestly"):
        st.markdown(
            "- Hyper-parameters were picked on the same CV folds that are reported, so CV scores are "
            "slightly optimistic; the held-out test row is the unbiased check.\n"
            "- Std is the sample std across folds (ddof=1).\n"
            "- Threshold is 0.5 for every model. Recall can be raised by lowering it or by class weighting.\n"
            "- Educational demo, not a medical diagnosis.")

elif page == "Visualizations":
    st.header("Visualizations")
    c1, c2 = st.columns(2)
    cm = M["cm"]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title("Confusion matrix")
    c1.pyplot(fig)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(M["fpr"], M["tpr"], label=f"AUC = {M['metrics'][4]:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    c2.pyplot(fig)
    c3, c4 = st.columns(2)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(M["loss_history"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-loss")
    ax.set_title("Gradient descent convergence")
    c3.pyplot(fig)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.barh(pp.FEATURES, M["w"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Learned weights (standardised features)")
    fig.tight_layout()
    c4.pyplot(fig)

else:
    st.header("About the model")
    st.markdown("**Model:** z = w·x + b,  p = σ(z)")
    st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}")
    st.latex(r"J(w,b)=-\frac1n\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]")
    st.latex(r"\frac{\partial J}{\partial w}=\frac1n X^\top(p-y),\qquad \frac{\partial J}{\partial b}=\frac1n\sum_i(p_i-y_i)")
    st.latex(r"w \leftarrow w-\alpha\,\partial J/\partial w")
    st.subheader("Other from-scratch models (see Model comparison)")
    st.markdown(
        "**KNN** (`ml/knn.py`): distance to every training point, vote among the k nearest.")
    st.latex(r"d(a,b)^2=\lVert a\rVert^2-2\,a\cdot b+\lVert b\rVert^2,\qquad "
             r"\hat p=\tfrac1k\sum_{i\in N_k(x)}y_i")
    st.markdown("**Decision tree** (`ml/tree.py`): CART. At each node pick the feature and threshold "
                "that most reduces Gini impurity, then recurse until a depth or size limit.")
    st.latex(r"G=2p(1-p),\qquad \text{split score}=G_{parent}-\tfrac{n_L}{n}G_L-\tfrac{n_R}{n}G_R")
    st.markdown("**Random forest** (`ml/forest.py`): many trees, each on a bootstrap sample and a random "
                "subset of features at every split; average their probabilities (bagging).")
    st.subheader("Pipeline")
    st.markdown("Pipeline: stratified 80/20 split → zero-as-missing → train-median imputation → "
                "standardisation (train stats) → gradient descent → threshold at "
                f"{thr}. Source: `ml/logreg.py`, `ml/metrics.py`, `ml/preprocess.py`.")
