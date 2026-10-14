
#Plot the mean perturbation curves from the saved experiment results.


import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_IG  = "results/ig"
ROOT_SG  = "results/smoothgrad"
ROOT_DTD = "results/dtd_final"

METHOD_ROOTS = {
    "ig":            ROOT_IG,
    "smoothgrad_ig": ROOT_SG,
    "dtd":           ROOT_DTD,
}

MODELS   = ["cnn", "cnn_lstm"]
PERTURBS = ["zero_out", "interpolation"]
GRAN     = "segment"

OUT_DIR    = "results/figures_method_curve_grid"
OUT_PREFIX = "method_curve_grid_segment"

SUBPLOT_TITLES = {
    ("zero_out",      "cnn"):      "CNN · zero-out",
    ("zero_out",      "cnn_lstm"): "CNN-LSTM · zero-out",
    ("interpolation", "cnn"):      "CNN · interpolation",
    ("interpolation", "cnn_lstm"): "CNN-LSTM · interpolation",
}

METHOD_STYLE = {
    "ig":            {"label": "IG",            "color": "#1f77b4", "lw": 2.0, "ls": "-"},
    "smoothgrad_ig": {"label": "SmoothGrad-IG", "color": "#ff7f0e", "lw": 2.0, "ls": "--"},
    "dtd":           {"label": "DTD",            "color": "#2ca02c", "lw": 2.0, "ls": "-."},
    "random":        {"label": "Random",         "color": "#999999", "lw": 1.5, "ls": ":"},
}


def _exp_dir(method_key: str, model: str, perturb: str) -> str:
    return os.path.join(METHOD_ROOTS[method_key], model, method_key, perturb, GRAN)


def load_mean_curves(method_key: str, model: str, perturb: str):
    
    #Load the saved mean perturbation curves for an experiment.
    
    exp_dir   = _exp_dir(method_key, model, perturb)
    main_path = os.path.join(exp_dir, "mean_curve_main.npy")
    rand_path = os.path.join(exp_dir, "mean_curve_random.npy")

    if not os.path.exists(main_path):
        raise FileNotFoundError(
            f"mean_curve_main.npy not found in:\n  {exp_dir}\n"
            "Re-run the benchmark so plotting.py writes it alongside mean_curve.png."
        )

    mean_main = np.load(main_path).astype(np.float64)
    mean_rand = np.load(rand_path).astype(np.float64) if os.path.exists(rand_path) else None
    return mean_main, mean_rand


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data   = {}
    random = {}
    missing = []

    for perturb in PERTURBS:
        for model in MODELS:
            for method_key in ["ig", "smoothgrad_ig", "dtd"]:
                try:
                    mean_main, mean_rand = load_mean_curves(method_key, model, perturb)
                    data[(perturb, model, method_key)] = mean_main
                    print(
                        f"  loaded  {method_key:15s} / {model:8s} / {perturb:15s} "
                        f"→ {len(mean_main)} steps"
                    )
                    # IG's random curve is used as the shared Random baseline
                    if method_key == "ig" and mean_rand is not None:
                        random[(perturb, model)] = mean_rand
                except Exception as exc:
                    tag = f"{method_key}/{model}/{perturb}"
                    missing.append(tag)
                    print(f"  MISSING {tag}: {exc}")

    if not data:
        print("\nERROR: No data loaded. Check the result root paths at the top of this script.")
        sys.exit(1)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8),
        sharex=True,
    )

    csv_rows = []

    for row_idx, perturb in enumerate(PERTURBS):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            ax.set_title(SUBPLOT_TITLES[(perturb, model)], fontsize=12, fontweight="bold")

            for method_key in ["ig", "smoothgrad_ig", "dtd"]:
                key = (perturb, model, method_key)
                if key not in data:
                    continue
                curve = data[key]
                x     = np.linspace(0, 1, len(curve))
                sty   = METHOD_STYLE[method_key]
                ax.plot(x, curve,
                        color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                        label=sty["label"])
                for i, (xi, yi) in enumerate(zip(x, curve)):
                    csv_rows.append(dict(
                        model=model, perturb=perturb, method=method_key,
                        step=i, fraction_perturbed=round(float(xi), 6),
                        mean_ba=round(float(yi), 6),
                    ))

            if (perturb, model) in random:
                curve = random[(perturb, model)]
                x     = np.linspace(0, 1, len(curve))
                sty   = METHOD_STYLE["random"]
                ax.plot(x, curve,
                        color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                        label=sty["label"])
                for i, (xi, yi) in enumerate(zip(x, curve)):
                    csv_rows.append(dict(
                        model=model, perturb=perturb, method="random",
                        step=i, fraction_perturbed=round(float(xi), 6),
                        mean_ba=round(float(yi), 6),
                    ))

            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.25, linewidth=0.7)
            ax.tick_params(labelsize=9)
            if col_idx == 0:
                ax.set_ylabel("Mean balanced accuracy", fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("Fraction of segments perturbed", fontsize=10)

    # # Use one shared legend for all subplots.
    seen, all_handles, all_labels = set(), [], []
    for ax in axes.flat:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in seen:
                all_handles.append(h)
                all_labels.append(lbl)
                seen.add(lbl)
    order_map = {sty["label"]: i for i, sty in enumerate(METHOD_STYLE.values())}
    paired = sorted(zip(all_labels, all_handles), key=lambda t: order_map.get(t[0], 99))
    all_labels, all_handles = zip(*paired) if paired else ([], [])

    fig.legend(
        all_handles, all_labels,
        loc="lower center",
        ncol=len(all_labels),
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "Mean Perturbation Curves by Attribution Method (segment granularity)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    png_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}.png")
    pdf_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}.pdf")
    csv_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_values.csv")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {csv_path}")

    if missing:
        print(f"\nMissing combinations ({len(missing)}):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
