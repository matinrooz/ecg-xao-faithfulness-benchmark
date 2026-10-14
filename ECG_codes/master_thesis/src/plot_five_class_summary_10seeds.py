"""
Five-class performance summary for 10 seeds 


"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve, accuracy_score,
                             f1_score, roc_auc_score, average_precision_score)
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
METRICS = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
           ("macro_auc", "Macro ROC-AUC"), ("macro_ap", "Macro Avg Precision")]


def load(key, s):
    p = np.load(os.path.join(PRED, f"{key}_seed_{s}_probs.npy"))
    y = np.load(os.path.join(PRED, f"{key}_seed_{s}_true.npy"))
    return y, p


def per_seed_metrics(key):
    out = {k: [] for k, _ in METRICS}
    for s in SEEDS:
        y, p = load(key, s)
        yb = label_binarize(y, classes=CLASSES)
        out["accuracy"].append(accuracy_score(y, p.argmax(1)))
        out["macro_f1"].append(f1_score(y, p.argmax(1), average="macro"))
        out["macro_auc"].append(roc_auc_score(yb, p, average="macro", multi_class="ovr"))
        out["macro_ap"].append(average_precision_score(yb, p, average="macro"))
    return out


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


def seed_curves(key, fn):
    return np.vstack([fn(*load(key, s)) for s in SEEDS])


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

    metrics = {key: per_seed_metrics(key) for key, _, _, _ in MODELS}
    for key, _, _, _ in MODELS:
        assert len(metrics[key]["accuracy"]) == 10, "expected exactly 10 seeds"

    # printed mean ± std table
    print("Metric                 CNN                 CNN-BiLSTM with Attention")
    for k, lab in METRICS:
        c, l = np.array(metrics["cnn"][k]), np.array(metrics["cnn_bilstm"][k])
        print(f"{lab:20s}  {c.mean():.3f} ± {c.std():.3f}       {l.mean():.3f} ± {l.std():.3f}")

    plt.rcParams.update({"font.size": 11, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rng = np.random.default_rng(0)

    # (a) boxplots
    ax = axes[0]
    xpos, width = np.arange(len(METRICS)), 0.34
    for j, (key, disp, color, _) in enumerate(MODELS):
        data = [metrics[key][k] for k, _ in METRICS]
        pos = xpos + (j - 0.5) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9,
                        patch_artist=True, medianprops=dict(color="black"))
        for b in bp["boxes"]:
            b.set(facecolor=color, alpha=0.35, edgecolor=color)
        for ki, (k, _) in enumerate(METRICS):
            xs = pos[ki] + (rng.random(len(SEEDS)) - 0.5) * width * 0.5
            ax.scatter(xs, metrics[key][k], s=18, color=color,
                       edgecolor="black", linewidth=0.4, zorder=3,
                       label=disp if ki == 0 else None)
    ax.set_xticks(xpos)
    ax.set_xticklabels([lab for _, lab in METRICS], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("(a) Classification metrics (10 seeds)")
    ax.legend(loc="lower right"); ax.grid(True, axis="y", alpha=0.3)

    # (b) macro ROC
    ax = axes[1]
    for key, disp, color, ls in MODELS:
        M = seed_curves(key, macro_roc)
        auc = np.array(metrics[key]["macro_auc"])
        band(ax, GRID, M, color, ls, f"{disp} (AUC={auc.mean():.3f}±{auc.std():.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.4", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("(b) Macro-average ROC (one-vs-rest)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)

    # (c) macro PR
    ax = axes[2]
    for key, disp, color, ls in MODELS:
        M = seed_curves(key, macro_pr)
        ap = np.array(metrics[key]["macro_ap"])
        band(ax, GRID, M, color, ls, f"{disp} (AP={ap.mean():.3f}±{ap.std():.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("(c) Macro-average Precision-Recall (one-vs-rest)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)

    fig.suptitle("Five-class classification performance: "
                 "CNN vs CNN–BiLSTM with Attention", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(OUTDIR, exist_ok=True)
    png = os.path.join(OUTDIR, "five_class_performance_summary_10seeds.png")
    pdf = os.path.join(OUTDIR, "five_class_performance_summary_10seeds.pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved:", pdf)
    print("saved:", png)


if __name__ == "__main__":
    main()
