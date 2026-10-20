"""Streamlit app: diabetes classification with scikit-learn.  Run: streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model import FEATURES, METRICS, ZERO_AS_MISSING, logistic_coefficients, train_all

DATA_PATH = "diabetes.csv"
STUDENT = "Aryav_2427010214"
LABELS = ["Pregnancies", "Glucose (mg/dL)", "Blood pressure (mmHg)", "Skin thickness (mm)",
          "Insulin (mu U/ml)", "BMI", "Diabetes pedigree function", "Age (years)"]
DEFAULTS = [2, 120, 80, 20, 80, 28.0, 0.5, 45]  # sample values only
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1-score", "roc_auc": "ROC-AUC"}
DISCLAIMER = "Educational ML demo. Not a medical diagnosis."

st.set_page_config(page_title="GlucoPredict", page_icon=":material/glucose:", layout="wide")


@st.cache_resource(show_spinner="Training the models (a few seconds)...")
def get_results():
    return train_all(DATA_PATH)       # trains the 3 models once, then reuses them


@st.cache_data
def get_data():
    return pd.read_csv(DATA_PATH)


results, info = get_results()
df = get_data()

st.sidebar.title("GlucoPredict")
st.sidebar.caption(f"ML mini project · {STUDENT}")
page = st.sidebar.radio("Go to", ["Home", "Data exploration", "Predict",
                                  "Model comparison", "About the models"])
st.sidebar.info(DISCLAIMER)


# ---------------------------------------------------------------- Home
if page == "Home":
    st.title("Diabetes classification")
    st.write("Binary classification on the Pima Indians Diabetes dataset (8 numeric features). "
             "Three scikit-learn algorithms are compared: Logistic Regression, KNN and Decision Tree.")
    c = st.columns(3)
    c[0].metric("Rows in dataset", len(df))
    c[1].metric("Training rows", info["n_train"])
    c[2].metric("Test rows", info["n_test"])
    st.subheader("Test-set ROC-AUC")
    st.dataframe(pd.Series({n: round(r["test"]["roc_auc"], 3) for n, r in results.items()},
                           name="ROC-AUC"))
    st.caption(DISCLAIMER)

# ---------------------------------------------------------------- Data exploration
elif page == "Data exploration":
    st.header("Data exploration")
    n = len(df)
    c = st.columns(4)
    c[0].metric("Rows", n)
    c[1].metric("Features", len(FEATURES))
    c[2].metric("Diabetes-positive", f"{df['Outcome'].mean():.1%}")
    c[3].metric("Duplicate rows", int(df.duplicated().sum()))

    st.subheader("1. Class balance")
    counts = df["Outcome"].value_counts().sort_index()
    st.bar_chart(counts.rename({0: "No diabetes (0)", 1: "Diabetes (1)"}))
    st.markdown(f"The classes are imbalanced ({counts[0]} vs {counts[1]}). A model that **always predicts "
                f"'no diabetes' gets {counts[0] / n:.1%} accuracy** and finds nobody, so we also look at "
                "recall, F1 and ROC-AUC, and split with stratification.")

    st.subheader("2. Hidden missing values: zeros that are not real")
    st.markdown("No empty cells and no duplicates, but in five columns **0 is impossible** (nobody has "
                "0 blood pressure or 0 BMI). Those zeros are missing values in disguise.")
    miss = pd.DataFrame({"Zeros": (df[ZERO_AS_MISSING] == 0).sum()})
    miss["% of rows"] = (miss["Zeros"] / n * 100).round(1)
    col1, col2 = st.columns(2)
    col1.dataframe(miss)
    col2.bar_chart(miss["% of rows"])
    st.caption("Fix: replace them with the column median, learned from training rows only "
               "(inside the scikit-learn Pipeline). Pregnancies = 0 is genuine and left alone.")

    clean = df.copy()
    clean[ZERO_AS_MISSING] = clean[ZERO_AS_MISSING].replace(0, np.nan)

    st.subheader("3. How each feature differs between the classes")
    feat = st.selectbox("Feature", FEATURES, index=1)
    values = clean[feat]
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.histogram_bin_edges(values.dropna(), bins=25)
    ax.hist(values[df["Outcome"] == 0].dropna(), bins=bins, alpha=0.6, color="#2471a3", label="No diabetes")
    ax.hist(values[df["Outcome"] == 1].dropna(), bins=bins, alpha=0.6, color="#c0392b", label="Diabetes")
    ax.set_xlabel(feat)
    ax.set_ylabel("Rows")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("**Class means** (hidden zeros ignored):")
    st.dataframe(clean.groupby("Outcome").mean().T.rename(columns={0: "No diabetes", 1: "Diabetes"}).round(2))

    st.subheader("4. Which features relate to the outcome?")
    corr = clean.corr()
    col1, col2 = st.columns(2)
    col1.bar_chart(corr["Outcome"].drop("Outcome").sort_values())
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), [c[:8] for c in corr.columns], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr)), [c[:8] for c in corr.columns], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    col2.pyplot(fig)
    top = corr["Outcome"].drop("Outcome").abs().sort_values(ascending=False).index[:2]
    st.caption(f"{top[0]} and {top[1]} correlate most with the outcome. Correlation is not causation.")

# ---------------------------------------------------------------- Predict
elif page == "Predict":
    st.header("Enter measurements")
    st.caption("A 0 in Glucose, Blood pressure, Skin thickness, Insulin or BMI is treated as missing "
               "and replaced by the training median, exactly as in training.")
    cols = st.columns(4)
    vals = []
    for i, (label, default) in enumerate(zip(LABELS, DEFAULTS)):
        with cols[i % 4]:
            if isinstance(default, float):
                vals.append(st.number_input(label, min_value=0.0, value=float(default), step=0.1, key=label))
            else:
                vals.append(st.number_input(label, min_value=0, value=int(default), step=1, key=label))
    if st.button("Predict", type="primary"):
        row = pd.DataFrame([dict(zip(FEATURES, vals))])
        out = []
        for name, r in results.items():
            p = r["model"].predict_proba(row)[0, 1]
            out.append({"Model": name, "Probability of diabetes": f"{p:.1%}",
                        "Predicted class": "1 · Diabetes indicated" if p >= 0.5 else "0 · Not indicated"})
        st.dataframe(pd.DataFrame(out).set_index("Model"))
        st.caption("The three models can disagree: that is normal. " + DISCLAIMER)

# ---------------------------------------------------------------- Model comparison
elif page == "Model comparison":
    st.header("Model comparison")
    st.markdown(
        f"All three models are scikit-learn classifiers inside the same Pipeline (fill hidden zeros, scale, "
        f"model). Each model's setting was chosen by **GridSearchCV with stratified 5-fold cross-validation** "
        f"on the {info['n_train']} training rows (best mean F1). The {info['n_test']}-row test set is used "
        f"once, at the end.")

    st.subheader("Cross-validation: mean ± std over the 5 folds")
    cv_table = pd.DataFrame(
        {n: {METRIC_LABELS[m]: f"{r['cv'][m][0]:.3f} ± {r['cv'][m][1]:.3f}" for m in METRICS}
         for n, r in results.items()}).T
    st.dataframe(cv_table)

    metric = st.selectbox("Chart metric", METRICS, index=3, format_func=METRIC_LABELS.get)
    names = list(results)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(names, [results[n]["cv"][metric][0] for n in names],
           yerr=[results[n]["cv"][metric][1] for n in names], capsize=5, color="#2471a3")
    ax.set_ylim(0, 1)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"CV {METRIC_LABELS[metric]} (bars = ±1 std)")
    fig.tight_layout()
    st.pyplot(fig)

    best = max(results, key=lambda n: results[n]["cv"]["f1"][0])
    best_mean, best_std = results[best]["cv"]["f1"]
    notes = []
    for n, r in results.items():
        if n != best:
            mean, std = r["cv"]["f1"]
            gap = best_mean - mean
            verdict = "comparable (gap is smaller than the fold-to-fold spread)" if gap <= max(best_std, std) \
                else "behind (gap is larger than the fold-to-fold spread)"
            notes.append(f"- **{n}**: F1 is {gap:.3f} below the best, so {verdict}")
    st.info(f"Highest mean CV F1: **{best}**.\n\n" + "\n".join(notes) +
            "\n\nThis is a rough rule of thumb, not a statistical test.")

    st.subheader("Held-out test set (scored once)")
    st.dataframe(pd.DataFrame({n: {METRIC_LABELS[m]: round(r["test"][m], 3) for m in METRICS}
                               for n, r in results.items()}).T)
    st.caption(f"Only {info['n_test']} test rows, so accuracy is uncertain by about ±3.5 points. "
               "A model can win the test set by luck and still lose in cross-validation.")

    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    for n, r in results.items():
        ax.plot(*r["roc"], label=f"{n} ({r['test']['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC curves")
    ax.legend(fontsize=7)
    c1.pyplot(fig)
    coef = logistic_coefficients(results["Logistic Regression"]["model"]).sort_values()
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.barh(coef.index, coef.values, color="#2471a3")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Logistic regression weights (scaled features)")
    fig.tight_layout()
    c2.pyplot(fig)

    st.subheader("Confusion matrices (test set)")
    for col, (n, r) in zip(st.columns(3), results.items()):
        fig, ax = plt.subplots(figsize=(3, 2.8))
        ax.imshow(r["confusion_matrix"], cmap="Blues")
        for (i, j), v in np.ndenumerate(r["confusion_matrix"]):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(n, fontsize=9)
        fig.tight_layout()
        col.pyplot(fig)

    with st.expander("Chosen settings and the full search"):
        st.dataframe(pd.Series({n: str(r["best_params"]) for n, r in results.items()}, name="Best setting"))
        for n, r in results.items():
            st.caption(f"{n}: mean CV F1 for every setting tried")
            st.dataframe(r["grid"].round(3))
    with st.expander("How to read this honestly"):
        st.markdown(
            "- Settings were picked on the same folds that are reported, so CV scores are slightly optimistic; "
            "the test set is the unbiased check.\n"
            "- std is taken over only 5 folds, so it is itself rough.\n"
            "- Threshold is 0.5 for every model. Recall can be raised by lowering it or by class weighting.\n"
            f"- {DISCLAIMER}")

# ---------------------------------------------------------------- About
else:
    st.header("About the models")
    st.subheader("The Pipeline (same for every model)")
    st.markdown("1. **SimpleImputer(missing_values=0, strategy='median')** on the five columns where 0 is impossible.\n"
                "2. **StandardScaler**: subtract the mean, divide by the std.\n"
                "3. The **model**.\n\n"
                "A scikit-learn `Pipeline` learns the medians and scaling from the training rows only, every "
                "time it is fitted, so no information leaks from validation or test data.")
    st.subheader("Logistic Regression  (`LogisticRegression`)")
    st.latex(r"z = w\cdot x + b,\qquad p=\sigma(z)=\frac{1}{1+e^{-z}}")
    st.markdown("Predicts diabetes when p is at least 0.5. Training minimises the log-loss with gradient descent. "
                "`class_weight='balanced'` makes the rarer diabetes cases count more, which raises recall.")
    st.subheader("K-Nearest Neighbours  (`KNeighborsClassifier`)")
    st.markdown("Finds the k most similar training people (Euclidean distance on scaled features) and takes a vote. "
                "k is tuned by cross-validation.")
    st.subheader("Decision Tree  (`DecisionTreeClassifier`)")
    st.latex(r"\text{Gini}=2p(1-p)")
    st.markdown("Repeatedly asks the yes/no question about one feature that most reduces Gini impurity. "
                "`max_depth` limits how far it can grow, which controls overfitting.")
    st.subheader("Evaluation")
    st.markdown("`train_test_split(stratify=y)` for the 80/20 split, `StratifiedKFold(5)` with `GridSearchCV` to choose "
                "settings, then `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `confusion_matrix` and "
                "`roc_curve` from `sklearn.metrics`.")
