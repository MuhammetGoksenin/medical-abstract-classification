"""
Calibration / confidence analysis on the validation set.

Question answered: when a model says "I am X% sure", is it actually right X% of the time?

Models re-trained here with the exact configs from 04_baseline_benchmark.ipynb:
  - Logistic Regression : TF-IDF unigram, max_features=30000, C=2, class_weight="balanced"
  - Naive Bayes (default): TF-IDF unigram+bigram, max_features=30000, MultinomialNB()  (alpha=1.0)
  - Naive Bayes (tuned)  : BoW unigram, max_features=30000, MultinomialNB(alpha=0.5)

Extra models (e.g. BiomedBERT) can be added from a probabilities CSV with columns
abstract_id, prob_1 ... prob_5 (validation set):
  python calibration_analysis.py DATA_DIR --probs "BiomedBERT=biomedbert_validation_probabilities.csv"

The test set is never read.

Outputs (in --out, default "calibration_outputs"):
  calibration_summary_validation.csv   one row per model
  reliability_diagram.png              confidence vs actual accuracy
  <model>_validation_confidence.csv    per-abstract prediction and confidence
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

LABELS = np.array([1, 2, 3, 4, 5])
BINS = np.linspace(0, 1, 11)          # 10 equal-width confidence bins
THRESHOLDS = [0.5, 0.7]               # "send to human review" thresholds


def fit_predict_proba(vectorizer, model, train_df, val_df):
    """Fit on train only, return class probabilities for validation."""
    X_train = vectorizer.fit_transform(train_df["medical_abstract"].astype(str))
    model.fit(X_train, train_df["condition_label"])
    return model.predict_proba(vectorizer.transform(val_df["medical_abstract"].astype(str)))


def expected_calibration_error(conf, correct):
    """Weighted average gap between confidence and accuracy over the bins."""
    ece = 0.0
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += abs(correct[in_bin].mean() - conf[in_bin].mean()) * in_bin.mean()
    return ece


def analyse(name, probs, y_true):
    conf = probs.max(axis=1)
    pred = LABELS[probs.argmax(axis=1)]
    correct = pred == y_true
    row = {
        "Model": name,
        "Macro F1": f1_score(y_true, pred, average="macro"),
        "Accuracy": correct.mean(),
        "Mean confidence": conf.mean(),
        "ECE": expected_calibration_error(conf, correct),
        "Errors with confidence > 0.9": f"{int(((conf > 0.9) & ~correct).sum())} / {int((~correct).sum())}",
    }
    for t in THRESHOLDS:
        kept = conf >= t
        row[f"Auto-classified at >= {t}"] = kept.mean()
        row[f"Accuracy at >= {t}"] = correct[kept].mean() if kept.any() else np.nan
    curve = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.sum() >= 5:          # skip nearly empty bins (too noisy to plot)
            curve.append((conf[in_bin].mean(), correct[in_bin].mean()))
    return row, curve, pred, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="folder with train.csv and validation.csv")
    parser.add_argument("--probs", action="append", default=[],
                        help='extra model as "Name=path.csv" (abstract_id, prob_1..prob_5)')
    parser.add_argument("--out", default="calibration_outputs")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "validation.csv"))
    y_val = val_df["condition_label"].values

    models = {
        "Logistic Regression": fit_predict_proba(
            TfidfVectorizer(lowercase=True, max_features=30000, ngram_range=(1, 1)),
            LogisticRegression(C=2, class_weight="balanced", max_iter=1000, random_state=42),
            train_df, val_df),
        "Naive Bayes (default)": fit_predict_proba(
            TfidfVectorizer(lowercase=True, max_features=30000, ngram_range=(1, 2)),
            MultinomialNB(), train_df, val_df),
        "Naive Bayes (tuned)": fit_predict_proba(
            CountVectorizer(lowercase=True, max_features=30000, ngram_range=(1, 1)),
            MultinomialNB(alpha=0.5), train_df, val_df),
    }
    for spec in args.probs:
        name, path = spec.split("=", 1)
        extra = pd.read_csv(path).set_index("abstract_id").loc[val_df["abstract_id"]]
        models[name] = extra[[f"prob_{i}" for i in LABELS]].values

    rows, curves = [], {}
    for name, probs in models.items():
        row, curve, pred, conf = analyse(name, probs, y_val)
        rows.append(row)
        curves[name] = curve
        pd.DataFrame({"abstract_id": val_df["abstract_id"], "true_label": y_val,
                      "predicted_label": pred, "confidence": conf.round(4)}).to_csv(
            os.path.join(args.out, name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                         + "_validation_confidence.csv"), index=False)

    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(os.path.join(args.out, "calibration_summary_validation.csv"), index=False)
    print(summary.to_string(index=False))

    # Reliability diagram (tuned NB left out of the plot: almost all its points sit in the last bin)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    plot_names = [n for n in curves if n != "Naive Bayes (tuned)"]
    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=200)
    ax.plot([0, 1], [0, 1], color="#b9b8b3", lw=1.5, ls="--", label="Perfect calibration")
    for name, color in zip(plot_names, colors):
        xs, ys = zip(*curves[name])
        ax.plot(xs, ys, marker="o", ms=8, lw=2, color=color, label=name, mec="white", mew=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model confidence")
    ax.set_ylabel("Actual accuracy")
    ax.text(0.06, 0.86, "Above the line: underconfident", fontsize=10, color="#52514e")
    ax.text(0.55, 0.06, "Below the line: overconfident", fontsize=10, color="#52514e")
    ax.grid(color="#ebeae6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, fontsize=9)
    ax.set_title("Reliability diagram (validation)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "reliability_diagram.png"))
    print(f"\nSaved outputs to {args.out}/")


if __name__ == "__main__":
    main()
