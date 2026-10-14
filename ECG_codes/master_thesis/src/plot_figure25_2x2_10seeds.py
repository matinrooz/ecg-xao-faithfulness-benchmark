"""
  2x2 layout for all od our 10 seeds for  Figure 25.

"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score)
from sklearn.preprocessing import label_binarize

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
RESULT = os.path.join(ROOT, "result")
OUTDIR = os.path.join(RESULT, "figures_5class_10seed")
PRED   = os.path.join(OUTDIR, "predictions")

SEEDS   = [42, 123, 777, 2024, 9999, 47, 48, 49, 50, 51]
CLASSES = list(range(5))
MODELS  = [("cnn", "CNN"), ("cnn_bilstm", "CNN–BiLSTM with Attention")]
PANELS  = [("accuracy", "(a) Accuracy", "Accuracy"),
           ("macro_f1", "(b) Macro F1", "Macro F1 Score"),
           ("macro_auc", "(c) Macro AUC", "Macro AUC"),
           ("macro_ap", "(d) Macro Average Precision", "Macro Average Precision")]
POINT_COLOR = "#4c72b0"


def per_seed_metrics(key):
    out = {m: [] for m, _, _ in PANELS}
    for s in SEEDS:
        p = np.load(os.path.join(PRED, f"{key}_seed_{s}_probs.npy"))
        y = np.load(os.path.join(PRED, f"{key}_seed_{s}_true.npy"))
        yb = label_binarize(y, classes=CLASSES); yp = p.argmax(1)
        out["accuracy"].append(accuracy_score(y, yp))
        out["macro_f1"].append(f1_score(y, yp, average="macro"))
        out["macro_auc"].append(roc_auc_score(yb, p, average="macro", multi_class="ovr"))
        out["macro_ap"].append(average_precision_score(yb, p, average="macro"))
    return out


def main():
    for key, _ in MODELS:
        for s in SEEDS:
            if not os.path.exists(os.path.join(PRED, f"{key}_seed_{s}_probs.npy")):
                raise SystemExit(f"STOP - missing {key}_seed_{s}_probs.npy")

    metrics = {key: per_seed_metrics(key) for key, _ in MODELS}
    assert all(len(metrics[k][m]) == 10 for k, _ in MODELS for m, _, _ in PANELS)

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    rng = np.random.default_rng(0)
    positions = [1, 2]

    for ax, (mkey, title, ylab) in zip(axes.ravel(), PANELS):
        data = [metrics[key][mkey] for key, _ in MODELS]
        ax.boxplot(data, positions=positions, widths=0.55)
        for pos, vals in zip(positions, data):
            xs = pos + rng.uniform(-0.09, 0.09, size=len(vals))
            ax.scatter(xs, vals, s=26, color=POINT_COLOR, edgecolor="black",
                       linewidth=0.4, alpha=0.8, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([lbl for _, lbl in MODELS], rotation=12, ha="right")
        ax.set_xlim(0.5, 2.5)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(pad=1.6, h_pad=2.4, w_pad=2.4)
    os.makedirs(OUTDIR, exist_ok=True)
    png = os.path.join(OUTDIR, "figure25_2x2_10seeds.png")
    pdf = os.path.join(OUTDIR, "figure25_2x2_10seeds.pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved:", pdf)
    print("saved:", png)


if __name__ == "__main__":
    main()
