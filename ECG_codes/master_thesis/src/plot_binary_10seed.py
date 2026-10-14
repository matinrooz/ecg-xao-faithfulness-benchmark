"""
Binary NORM-vs-MI 10-seed reporting
"""

import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "result", "figures_binary_10seed")
PRED   = os.path.join(OUTDIR, "predictions")

SEEDS = [42, 123, 777, 2024, 9999, 47, 48, 49, 50, 51]
POS   = 1                                     # MI positive
GRID  = np.linspace(0.0, 1.0, 200)
LINEWIDTH, BAND_ALPHA = 2.6, 0.15
POINT_COLOR = "#4c72b0"
MODELS = [("cnn", "CNN", "#0072B2", "-"), ("cnn_bilstm", "CNN–BiLSTM with Attention", "#D55E00", "--")]
PANELS = [("accuracy", "(a) Accuracy", "Accuracy"),
          ("macro_f1", "(b) Macro F1", "Macro F1 Score"),
          ("auc", "(c) ROC-AUC", "ROC-AUC"),
          ("ap", "(d) Average Precision", "Average Precision")]


def load(key, s):
    p = np.load(os.path.join(PRED, f"{key}_seed_{s}_probs.npy"))
    y = np.load(os.path.join(PRED, f"{key}_seed_{s}_true.npy"))
    return y, p


def per_seed(key):
    out = {k: [] for k, _, _ in PANELS}
    for s in SEEDS:
        y, p = load(key, s); yp = p.argmax(1)
        out["accuracy"].append(accuracy_score(y, yp))
        out["macro_f1"].append(f1_score(y, yp, average="macro"))
        out["auc"].append(roc_auc_score(y, p[:, POS]))
        out["ap"].append(average_precision_score(y, p[:, POS]))
    return out


def roc_band(key):
    ts = []
    for s in SEEDS:
        y, p = load(key, s); fpr, tpr, _ = roc_curve(y, p[:, POS])
        t = np.interp(GRID, fpr, tpr); t[0] = 0.0; ts.append(t)
    return np.vstack(ts)


def pr_band(key):
    ps = []
    for s in SEEDS:
        y, p = load(key, s); pr, rc, _ = precision_recall_curve(y, p[:, POS])
        o = np.argsort(rc); ps.append(np.interp(GRID, rc[o], pr[o]))
    return np.vstack(ps)


def band(ax, x, M, color, ls, label):
    m, sd = M.mean(0), M.std(0)
    ax.plot(x, m, color=color, lw=LINEWIDTH, ls=ls, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=BAND_ALPHA, linewidth=0)


def main():
    for key, _, _, _ in MODELS:
        for s in SEEDS:
            for suf in ("probs", "true"):
                if not os.path.exists(os.path.join(PRED, f"{key}_seed_{s}_{suf}.npy")):
                    raise SystemExit(f"STOP missing {key}_seed_{s}_{suf}.npy")
    ref = load("cnn", SEEDS[0])[0]
    for key, _, _, _ in MODELS:
        for s in SEEDS:
            assert np.array_equal(load(key, s)[0], ref), "labels differ across seeds"
    prevalence = float(ref.mean())

    metrics = {key: per_seed(key) for key, _, _, _ in MODELS}
    os.makedirs(OUTDIR, exist_ok=True)

    # --- CSVs ---
    with open(os.path.join(OUTDIR, "binary_per_seed_metrics_10seed.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "seed", "accuracy", "macro_f1", "roc_auc", "average_precision"])
        for key, disp, _, _ in MODELS:
            for i, s in enumerate(SEEDS):
                w.writerow([disp, s, f"{metrics[key]['accuracy'][i]:.6f}", f"{metrics[key]['macro_f1'][i]:.6f}",
                            f"{metrics[key]['auc'][i]:.6f}", f"{metrics[key]['ap'][i]:.6f}"])
    with open(os.path.join(OUTDIR, "binary_summary_10seed.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "metric", "mean", "std"])
        for key, disp, _, _ in MODELS:
            for mk, _, _ in PANELS:
                a = np.array(metrics[key][mk]); w.writerow([disp, mk, f"{a.mean():.6f}", f"{a.std():.6f}"])

    # Table 4 
    def m3(key, mk): return f"{np.mean(metrics[key][mk]):.3f}"
    tex = [
        r"\begin{table}[htbp]", r"\centering", r"\begin{tabular}{lcccc}", r"\toprule",
        r"Model & Accuracy & Macro F1 & ROC-AUC & Average Precision \\", r"\midrule",
        f"CNN                        & {m3('cnn','accuracy')} & {m3('cnn','macro_f1')} & {m3('cnn','auc')} & {m3('cnn','ap')} \\\\",
        f"CNN--BiLSTM with Attention & {m3('cnn_bilstm','accuracy')} & {m3('cnn_bilstm','macro_f1')} & {m3('cnn_bilstm','auc')} & {m3('cnn_bilstm','ap')} \\\\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Mean classification performance on the binary NORM-versus-MI task across ten random seeds.}",
        r"\label{tab:binary_summary}", r"\end{table}"]
    open(os.path.join(OUTDIR, "table4_binary_10seed.tex"), "w").write("\n".join(tex) + "\n")

    # boxplots 
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    rng = np.random.default_rng(0); pos = [1, 2]
    for ax, (mk, title, ylab) in zip(axes.ravel(), PANELS):
        data = [metrics[key][mk] for key, _, _, _ in MODELS]
        ax.boxplot(data, positions=pos, widths=0.55)
        for p_, vals in zip(pos, data):
            xs = p_ + rng.uniform(-0.09, 0.09, size=len(vals))
            ax.scatter(xs, vals, s=26, color=POINT_COLOR, edgecolor="black", linewidth=0.4, alpha=0.8, zorder=3)
        ax.set_xticks(pos); ax.set_xticklabels([d for _, d, _, _ in MODELS], rotation=12, ha="right")
        ax.set_xlim(0.5, 2.5); ax.set_title(title); ax.set_ylabel(ylab); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(pad=1.6, h_pad=2.4, w_pad=2.4)
    fig.savefig(os.path.join(OUTDIR, "binary_figure25_2x2_10seeds.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "binary_figure25_2x2_10seeds.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    #  ROC + PR
    plt.rcParams.update({"font.size": 11, "legend.fontsize": 9})
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(13, 5.2))
    for key, disp, color, ls in MODELS:
        auc = np.array(metrics[key]["auc"])
        band(axr, GRID, roc_band(key), color, ls, f"{disp} (AUC={auc.mean():.3f}±{auc.std():.3f})")
    axr.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
    axr.set_xlim(0, 1); axr.set_ylim(0, 1); axr.set_xlabel("False Positive Rate"); axr.set_ylabel("True Positive Rate")
    axr.set_title("(a) ROC (MI positive)"); axr.legend(loc="lower right"); axr.grid(True, alpha=0.3)
    for key, disp, color, ls in MODELS:
        ap = np.array(metrics[key]["ap"])
        band(axp, GRID, pr_band(key), color, ls, f"{disp} (AP={ap.mean():.3f}±{ap.std():.3f})")
    axp.axhline(prevalence, ls=":", color="0.5", lw=1, label=f"No skill (MI prevalence={prevalence:.3f})")
    axp.set_xlim(0, 1); axp.set_ylim(0, 1); axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
    axp.set_title("(b) Precision-Recall (MI positive)"); axp.legend(loc="lower left"); axp.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "binary_roc_pr_10seed.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "binary_roc_pr_10seed.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    #  console summary 
    print("Binary 10-seed mean ± std:")
    for key, disp, _, _ in MODELS:
        s = {mk: (np.mean(metrics[key][mk]), np.std(metrics[key][mk])) for mk, _, _ in PANELS}
        print(f"  {disp:26s} Acc {s['accuracy'][0]:.3f}±{s['accuracy'][1]:.3f}  F1 {s['macro_f1'][0]:.3f}±{s['macro_f1'][1]:.3f}  "
              f"AUC {s['auc'][0]:.3f}±{s['auc'][1]:.3f}  AP {s['ap'][0]:.3f}±{s['ap'][1]:.3f}")
    print("MI prevalence:", round(prevalence, 4))
    print("outputs ->", OUTDIR)


if __name__ == "__main__":
    main()
