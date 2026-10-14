"""
 macro-average one-vs-rest ROC  and Precision-Recall 

"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import label_binarize

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
RESULT = os.path.join(ROOT, "result")
OUTDIR = os.path.join(RESULT, "figures_5class_10seed")
PRED   = os.path.join(OUTDIR, "predictions")

SEEDS       = [42, 123, 777, 2024, 9999, 47, 48, 49, 50, 51]
NUM_CLASSES = 5
CLASSES     = list(range(NUM_CLASSES))
GRID        = np.linspace(0.0, 1.0, 200)
LINEWIDTH, BAND_ALPHA = 2.6, 0.15

MODELS = [   # key, display label, colour, line style
    ("cnn",        "CNN",                        "#0072B2", "-"),
    ("cnn_bilstm", "CNN–BiLSTM with Attention",  "#D55E00", "--"),
]


def macro_roc(y, p):
    yb = label_binarize(y, classes=CLASSES)
    tprs = []
    for c in CLASSES:
        fpr, tpr, _ = roc_curve(yb[:, c], p[:, c])
        t = np.interp(GRID, fpr, tpr); t[0] = 0.0
        tprs.append(t)
    return np.mean(tprs, axis=0)


def macro_pr(y, p):
    yb = label_binarize(y, classes=CLASSES)
    precs = []
    for c in CLASSES:
        pr, rc, _ = precision_recall_curve(yb[:, c], p[:, c])
        o = np.argsort(rc)
        precs.append(np.interp(GRID, rc[o], pr[o]))
    return np.mean(precs, axis=0)


def load(key, s):
    p = np.load(os.path.join(PRED, f"{key}_seed_{s}_probs.npy"))
    y = np.load(os.path.join(PRED, f"{key}_seed_{s}_true.npy"))
    return y, p


def seed_curves(key, fn):
    return np.vstack([fn(*load(key, s)) for s in SEEDS])


def seed_scores(key):
    aucs, aps = [], []
    for s in SEEDS:
        y, p = load(key, s)
        yb = label_binarize(y, classes=CLASSES)
        aucs.append(roc_auc_score(yb, p, average="macro", multi_class="ovr"))
        aps.append(average_precision_score(yb, p, average="macro"))
    return np.array(aucs), np.array(aps)


def band(ax, x, M, color, ls, label):
    m, sd = M.mean(0), M.std(0)
    ax.plot(x, m, color=color, lw=LINEWIDTH, ls=ls, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=BAND_ALPHA, linewidth=0)


def main():
    for key, _, _, _ in MODELS:
        for s in SEEDS:
            for suf in ("probs", "true"):
                if not os.path.exists(os.path.join(PRED, f"{key}_seed_{s}_{suf}.npy")):
                    raise SystemExit(f"STOP - missing prediction file: {key}_seed_{s}_{suf}.npy")

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "legend.fontsize": 9, "figure.dpi": 150})
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(13, 5.2))

    #  macro ROC
    for key, disp, color, ls in MODELS:
        M = seed_curves(key, macro_roc)
        auc, _ = seed_scores(key)
        band(axr, GRID, M, color, ls, f"{disp} (AUC={auc.mean():.3f}±{auc.std():.3f})")
    axr.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    axr.set_xlim(0, 1); axr.set_ylim(0, 1)
    axr.set_xlabel("False Positive Rate"); axr.set_ylabel("True Positive Rate")
    axr.set_title("(a) Macro-average ROC (one-vs-rest)")
    axr.legend(loc="lower right"); axr.grid(True, alpha=0.3)

    #  macro PR
    for key, disp, color, ls in MODELS:
        M = seed_curves(key, macro_pr)
        _, ap = seed_scores(key)
        band(axp, GRID, M, color, ls, f"{disp} (AP={ap.mean():.3f}±{ap.std():.3f})")
    axp.set_xlim(0, 1); axp.set_ylim(0, 1)
    axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
    axp.set_title("(b) Macro-average Precision-Recall (one-vs-rest)")
    axp.legend(loc="lower left"); axp.grid(True, alpha=0.3)

    fig.suptitle("Macro-average one-vs-rest ROC and Precision-Recall for the CNN and "
                 "CNN–BiLSTM with Attention across 10 random seeds", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(OUTDIR, exist_ok=True)
    png = os.path.join(OUTDIR, "roc_pr_5class_10seed.png")
    pdf = os.path.join(OUTDIR, "roc_pr_5class_10seed.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("saved:", png)
    print("saved:", pdf)


if __name__ == "__main__":
    main()
