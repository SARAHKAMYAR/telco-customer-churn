"""Train, select and evaluate churn classifiers without using test labels for selection."""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, fbeta_score, precision_score, recall_score,
    precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
FEATURES = ["Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months",
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges"]
NUMERIC = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]
COLORS = {"Majority baseline": "#96A2B3", "Logistic regression": "#147D92", "Random forest": "#D87545"}
ROOT = Path(__file__).resolve().parents[1]


def load_data(path):
    data = pd.read_excel(path)
    required = set(FEATURES + ["CustomerID", "Churn Value", "Churn Label"])
    if missing := required - set(data.columns):
        raise ValueError(f"Wrong dataset variant. Missing columns: {sorted(missing)}")
    if data["CustomerID"].isna().any() or data["CustomerID"].duplicated().any():
        raise ValueError("Customer IDs must be present and unique.")
    if not data["Churn Value"].isin([0, 1]).all():
        raise ValueError("Churn Value must contain only 0 and 1.")
    if not data["Churn Label"].map({"Yes": 1, "No": 0}).eq(data["Churn Value"]).all():
        raise ValueError("Churn Label and Churn Value disagree.")
    data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")
    return data


def split_data(data):
    X, y = data[FEATURES].copy(), data["Churn Value"].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=SEED)
    assert not (set(X_train.index) & set(X_val.index) or set(X_train.index) & set(X_test.index) or set(X_val.index) & set(X_test.index))
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])


def train_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    candidates = {
        "Majority baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=1),
    }
    models, rows, parameters = {}, [], {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor()), ("model", estimator)])
        if name == "Random forest":
            search = GridSearchCV(pipe, {"model__max_depth": [6, 12, None], "model__min_samples_leaf": [2, 8]},
                cv=cv, scoring="average_precision", n_jobs=1, error_score="raise")
            search.fit(X_train, y_train)
            models[name] = search.best_estimator_
            scores = np.array([search.cv_results_[f"split{i}_test_score"][search.best_index_] for i in range(3)])
            parameters = search.best_params_
        else:
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1, error_score="raise")
            models[name] = pipe.fit(X_train, y_train)
        rows.append({"model": name, "cv_average_precision": scores.mean(), "fold_std": scores.std()})
    comparison = pd.DataFrame(rows).sort_values("cv_average_precision", ascending=False)
    # Cross-validation selects the model family. Validation selects only its threshold.
    selected = comparison.iloc[0]["model"]
    return models, comparison, selected, parameters


def choose_threshold(y_val, probabilities):
    thresholds = np.arange(.05, .951, .01)
    rows = []
    for t in thresholds:
        prediction = (probabilities >= t).astype(int)
        rows.append({"threshold": float(t), "precision": precision_score(y_val, prediction, zero_division=0),
            "recall": recall_score(y_val, prediction), "f2": fbeta_score(y_val, prediction, beta=2),
            "contact_rate": prediction.mean()})
    table = pd.DataFrame(rows)
    # If F2 ties, use the higher threshold to contact fewer customers.
    best = table.sort_values(["f2", "threshold"], ascending=False).iloc[0]
    return float(best["threshold"]), table


def metrics(y, probabilities, threshold):
    prediction = (probabilities >= threshold).astype(int)
    return {"accuracy": accuracy_score(y, prediction), "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction), "f1": f1_score(y, prediction),
        "roc_auc": roc_auc_score(y, probabilities), "average_precision": average_precision_score(y, probabilities),
        "brier_score": brier_score_loss(y, probabilities), "contact_rate": prediction.mean()}


def evaluate(models, selected, threshold, X_test, y_test):
    rows, probabilities = [], {}
    for name, model in models.items():
        p = model.predict_proba(X_test)[:, 1]
        probabilities[name] = p
        rows.append({"model": name, "threshold": .5, **metrics(y_test, p, .5)})
    rows.append({"model": selected + " (validation threshold)", "threshold": threshold,
        **metrics(y_test, probabilities[selected], threshold)})
    return pd.DataFrame(rows), probabilities


def bootstrap_intervals(y, p, repetitions=500):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    values = []
    for _ in range(repetitions):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append([roc_auc_score(y[idx], p[idx]), average_precision_score(y[idx], p[idx])])
    interval = np.quantile(values, [.025, .975], axis=0)
    return {"roc_auc": interval[:, 0].tolist(), "average_precision": interval[:, 1].tolist()}


def save_figure(fig, name, output):
    fig.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_figures(data, splits, models, selected, threshold, threshold_table, probabilities, comparison, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 15,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#334155", "text.color": "#18283B", "axes.edgecolor": "#CBD5E1",
        "figure.facecolor": "#FAFBFD", "axes.facecolor": "#FAFBFD", "svg.fonttype": "none"})
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    train = data.loc[X_train.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    counts = y_train.value_counts().reindex([0, 1])
    bars = axes[0].bar(["Stayed", "Churned"], counts, color=["#147D92", "#D87545"], width=.55)
    axes[0].bar_label(bars, labels=[f"{v:,}\n{v / len(train):.1%}" for v in counts], padding=7)
    axes[0].set_ylim(0, counts.max() * 1.25)
    axes[0].set(title="Churn is the smaller class", ylabel="Customers in the training set")
    segments = train.groupby("Contract")["Churn Value"].agg(["mean", "count"]).reindex(["Month-to-month", "One year", "Two year"])
    bars = axes[1].bar(segments.index, segments["mean"], color="#147D92", width=.55)
    axes[1].bar_label(bars, labels=[f"{r['mean']:.1%}\nn={int(r['count']):,}" for _, r in segments.iterrows()], padding=7)
    axes[1].set_ylim(0, .65)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(title="Churn differs by contract", ylabel="Observed churn rate")
    fig.suptitle("Customer churn overview · training data", fontsize=18, fontweight="bold")
    save_figure(fig, "01_customer_overview", output)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    bins = pd.cut(train["Tenure Months"], [-1, 12, 24, 48, 72], labels=["0–12", "13–24", "25–48", "49–72"])
    grouped = train.groupby(bins, observed=True)["Churn Value"].agg(["mean", "count"])
    bars = ax.bar(grouped.index.astype(str), grouped["mean"], width=.55, color="#147D92")
    ax.bar_label(bars, labels=[f"{r['mean']:.1%} (n={int(r['count']):,})" for _, r in grouped.iterrows()], padding=8)
    ax.set_ylim(0, max(grouped["mean"]) + .15)
    ax.set(title="Shorter-tenure customers have higher observed churn", xlabel="Tenure (months)", ylabel="Churn rate · training data")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save_figure(fig, "02_tenure", output)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for name, p in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        precision, recall, _ = precision_recall_curve(y_test, p)
        axes[0].plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} · {roc_auc_score(y_test, p):.3f}")
        axes[1].plot(recall, precision, color=COLORS[name], lw=2, label=f"{name} · {average_precision_score(y_test, p):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#CBD5E1")
    axes[1].axhline(y_test.mean(), ls="--", color="#CBD5E1")
    axes[0].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision–recall curves", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=9, loc="lower left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.suptitle("Model comparison · held-out test set", fontsize=18, fontweight="bold")
    save_figure(fig, "03_test_curves", output)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for metric, color in [("precision", "#147D92"), ("recall", "#D87545"), ("f2", "#6F62A7")]:
        ax.plot(threshold_table["threshold"], threshold_table[metric], label=metric.upper() if metric == "f2" else metric.title(), color=color, lw=2)
    ax.axvline(threshold, color="#334155", ls="--", label=f"Selected threshold = {threshold:.2f}")
    ax.set(title="Choosing how many customers to flag", xlabel="Churn score threshold", ylabel="Validation score", ylim=(0, 1))
    ax.legend(loc="lower left")
    save_figure(fig, "04_validation_threshold", output)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, t in zip(axes, [.5, threshold]):
        cm = confusion_matrix(y_test, probabilities[selected] >= t, labels=[0, 1])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ratio = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]:,}\n{ratio:.1%} of actual class", ha="center", va="center", fontsize=12, color="white" if ratio > .5 else "#18283B")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"],
            xlabel="Predicted", ylabel="Actual", title=f"Threshold {t:.2f}")
    fig.suptitle(f"{selected} · test-set errors", fontsize=18, fontweight="bold")
    save_figure(fig, "05_confusion_matrices", output)

    importance = permutation_importance(models[selected], X_test, y_test, n_repeats=10,
        random_state=SEED, scoring="average_precision", n_jobs=1)
    importance_table = pd.DataFrame({"feature": FEATURES, "mean_ap_drop": importance.importances_mean, "std_ap_drop": importance.importances_std}).sort_values("mean_ap_drop", ascending=False)
    top = importance_table.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.barh(top["feature"], top["mean_ap_drop"], xerr=top["std_ap_drop"], color="#147D92", capsize=3)
    ax.axvline(0, color="#CBD5E1")
    ax.set(title="Which inputs matter to the selected model?", xlabel="Decrease in test average precision after shuffling")
    fig.text(.5, -.02, "Error bars: standard deviation across 10 shuffles. Correlated inputs can share importance.", ha="center", fontsize=10)
    save_figure(fig, "06_permutation_importance", output)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for name in ["Logistic regression", "Random forest"]:
        actual, predicted = calibration_curve(y_test, probabilities[name], n_bins=8, strategy="quantile")
        axes[0].plot(predicted, actual, "o-", color=COLORS[name], label=f"{name} · Brier {brier_score_loss(y_test, probabilities[name]):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#CBD5E1")
    axes[0].set(title="Do risk scores match observed rates?", xlabel="Mean predicted score", ylabel="Observed churn rate", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=9)
    p = probabilities[selected]
    order = np.argsort(-p, kind="stable")
    fraction = np.arange(1, len(y_test) + 1) / len(y_test)
    capture = np.cumsum(np.asarray(y_test)[order]) / y_test.sum()
    axes[1].plot(fraction, capture, color="#147D92", lw=2, label=selected)
    axes[1].plot([0, 1], [0, 1], "--", color="#96A2B3", label="Random ranking")
    axes[1].set(title="How many churners does a shortlist reach?", xlabel="Fraction of customers contacted", ylabel="Fraction of churners captured")
    axes[1].legend(fontsize=9)
    save_figure(fig, "07_calibration_and_gains", output)
    return importance_table


def run(data_path, output=ROOT):
    data_path, output = Path(data_path), Path(output)
    data = load_data(data_path)
    splits = split_data(data)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    models, cv, selected, params = train_models(X_train, y_train)
    threshold, threshold_table = choose_threshold(y_val, models[selected].predict_proba(X_val)[:, 1])
    results, probabilities = evaluate(models, selected, threshold, X_test, y_test)
    importance = create_figures(data, splits, models, selected, threshold, threshold_table, probabilities, cv, output / "figures")
    report_dir = output / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "test_metrics.csv", index=False)
    cv.to_csv(report_dir / "cross_validation.csv", index=False)
    threshold_table.to_csv(report_dir / "validation_thresholds.csv", index=False)
    importance.to_csv(report_dir / "permutation_importance.csv", index=False)
    intervals = bootstrap_intervals(y_test, probabilities[selected])
    k = int(np.ceil(.2 * len(y_test)))
    capture = float(np.asarray(y_test)[np.argsort(-probabilities[selected], kind="stable")[:k]].sum() / y_test.sum())
    metadata = {"seed": SEED, "rows": len(data), "features": FEATURES, "selected_model": selected,
        "validation_threshold": threshold, "rf_parameters": params, "bootstrap_95_intervals": intervals,
        "top_20_percent_capture": capture, "total_charges_missing": int(data["Total Charges"].isna().sum()),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "splits": {name: {"rows": len(y), "churn_rate": float(y.mean())} for name, y in [("train", y_train), ("validation", y_val), ("test", y_test)]}}
    (report_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(cv.to_string(index=False))
    print(f"\nSelected model: {selected}; validation F2 threshold: {threshold:.2f}\n")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return metadata, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/Telco_customer_churn.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.data, args.output)
