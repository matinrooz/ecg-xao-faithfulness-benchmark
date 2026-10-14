"""
Bland-Altman agreement analysis.

Zero-out vs interpolation comparisons are restricted to MI samples
(sample_class == 1), since interpolation is not applied to NORM ECGs in this
study. Every other comparison (method vs method, architecture vs architecture,
granularity vs granularity) uses all paired samples.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bland_altman_analysis(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    save_dir: str,
    title: str,
) -> Dict:
    #Bland-Altman agreement analysis for two paired AUC arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert len(a) == len(b)

    mean_ab   = (a + b) / 2.0
    diff_ab   = a - b
    mean_diff = float(np.mean(diff_ab))
    std_diff  = float(np.std(diff_ab, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_ab, diff_ab, color="steelblue", alpha=0.75, s=55, zorder=3)
    ax.axhline(mean_diff, color="black",  linewidth=1.8, linestyle="-",
               label=f"Mean diff = {mean_diff:+.4f}")
    ax.axhline(loa_upper, color="crimson", linewidth=1.3, linestyle="--",
               label=f"+1.96 SD = {loa_upper:+.4f}")
    ax.axhline(loa_lower, color="crimson", linewidth=1.3, linestyle="--",
               label=f"-1.96 SD = {loa_lower:+.4f}")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel(f"Mean AUC   ({label_a} + {label_b}) / 2", fontsize=11)
    ax.set_ylabel(f"Difference AUC   {label_a} - {label_b}", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "bland_altman.png"), dpi=200)
    plt.close()

    if mean_diff > 0:
        interp = f"{label_a} has higher AUC on average -> {label_b} is more faithful"
    elif mean_diff < 0:
        interp = f"{label_b} has higher AUC on average -> {label_a} is more faithful"
    else:
        interp = "No mean difference — methods agree on average"

    lines = [
        "=" * 58,
        "Bland-Altman Agreement Analysis",
        f"  A = {label_a}",
        f"  B = {label_b}",
        "=" * 58,
        f"n pairs              : {len(a)}",
        f"Mean diff  (A - B)   : {mean_diff:+.6f}",
        f"SD of diff           : {std_diff:.6f}",
        f"LoA upper (+1.96 SD) : {loa_upper:+.6f}",
        f"LoA lower (-1.96 SD) : {loa_lower:+.6f}",
        f"LoA width            : {loa_upper - loa_lower:.6f}",
        "",
        f"Interpretation: {interp}",
        f"(AUC lower = better faithfulness; diff > 0 -> A worse)",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(save_dir, "bland_altman_report.txt"), "w") as f:
        f.write(text)

    return {
        "mean_diff": mean_diff, "std_diff": std_diff,
        "loa_upper": loa_upper, "loa_lower": loa_lower, "n": len(a),
    }


def run_bland_altman_analyses(
    out_dir: str,
    ig_results_dir: str = None,
    smoothgrad_results_dir: str = None,
    dtd_results_dir: str = None,
) -> None:
    """
    Run all Bland-Altman comparisons. Zero-out vs interpolation is restricted to
    MI samples (sample_class == 1) as we said already the rest use every paired sample
    """
    _ig_dir         = ig_results_dir         or out_dir
    _smoothgrad_dir = smoothgrad_results_dir or out_dir
    _dtd_dir        = dtd_results_dir        or out_dir

    def _load(model: str, method: str, perturb: str, gran: str,
              root: str = None, include_class: bool = False) -> pd.DataFrame:
        path = os.path.join(root or out_dir, model, method, perturb, gran, "summary.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Result not found: {path}")
        df = pd.read_csv(path).sort_values("index").reset_index(drop=True)
        if include_class and "sample_class" not in df.columns:
            raise KeyError(
                f"'sample_class' column not found in {path}. "
                "Re-run the perturbation evaluation with sample_class saved to summary.csv."
            )
        return df

    ba_root = os.path.join(out_dir, "bland_altman_mi_only")
    os.makedirs(ba_root, exist_ok=True)

    print(f"\n{'='*65}")
    print("  BLAND-ALTMAN ANALYSES  (MI-only variant)")
    print("  zero-out vs interpolation: MI samples only (sample_class == 1)")
    print(f"{'='*65}")

    # method vs method
    METHOD_COMPARISONS = [
        {
            "name":    "1_ig_vs_smoothgrad_ig",
            "title":   "Bland-Altman: IG vs SmoothGrad-IG\n"
                       "(CNN-LSTM . interpolation . segment)",
            "label_a": "IG",
            "label_b": "SmoothGrad-IG",
            "path_a":  ("cnn_lstm", "ig",            "interpolation", "segment"),
            "path_b":  ("cnn_lstm", "smoothgrad_ig", "interpolation", "segment"),
            "root_a":  "_ig_dir",
            "root_b":  "_smoothgrad_dir",
        },
        {
            "name":    "4_ig_vs_dtd",
            "title":   "Bland-Altman: IG vs DTD\n"
                       "(CNN-LSTM . interpolation . segment)",
            "label_a": "IG",
            "label_b": "DTD",
            "path_a":  ("cnn_lstm", "ig",  "interpolation", "segment"),
            "path_b":  ("cnn_lstm", "dtd", "interpolation", "segment"),
            "root_a":  "_ig_dir",
            "root_b":  "_dtd_dir",
        },
        {
            "name":    "5_smoothgrad_ig_vs_dtd",
            "title":   "Bland-Altman: SmoothGrad-IG vs DTD\n"
                       "(CNN-LSTM . interpolation . segment)",
            "label_a": "SmoothGrad-IG",
            "label_b": "DTD",
            "path_a":  ("cnn_lstm", "smoothgrad_ig", "interpolation", "segment"),
            "path_b":  ("cnn_lstm", "dtd",           "interpolation", "segment"),
            "root_a":  "_smoothgrad_dir",
            "root_b":  "_dtd_dir",
        },
        {
            "name":    "6_ig_vs_dtd_zero_out",
            "title":   "Bland-Altman: IG vs DTD\n"
                       "(CNN-LSTM . zero-out . segment)",
            "label_a": "IG",
            "label_b": "DTD",
            "path_a":  ("cnn_lstm", "ig",  "zero_out", "segment"),
            "path_b":  ("cnn_lstm", "dtd", "zero_out", "segment"),
            "root_a":  "_ig_dir",
            "root_b":  "_dtd_dir",
        },
    ]

    root_map = {
        "_ig_dir":         _ig_dir,
        "_smoothgrad_dir": _smoothgrad_dir,
        "_dtd_dir":        _dtd_dir,
    }

    for comp in METHOD_COMPARISONS:
        print(f"\n-- {comp['name']} --")
        try:
            df_a = _load(*comp["path_a"], root=root_map[comp["root_a"]])
            df_b = _load(*comp["path_b"], root=root_map[comp["root_b"]])
            merged = (
                df_a[["index", "auc_main"]]
                .merge(df_b[["index", "auc_main"]], on="index", suffixes=("_a", "_b"))
            )
            a = merged["auc_main_a"].to_numpy(dtype=np.float64)
            b = merged["auc_main_b"].to_numpy(dtype=np.float64)
            print(f"  Paired on index: {len(merged)} samples")
            save_dir = os.path.join(ba_root, comp["name"])
            bland_altman_analysis(
                a=a, b=b,
                label_a=comp["label_a"],
                label_b=comp["label_b"],
                save_dir=save_dir,
                title=comp["title"],
            )
            pd.DataFrame({
                "sample_idx": merged["index"].to_numpy(dtype=int),
                "auc_A": a, "auc_B": b,
                "mean_auc": (a + b) / 2.0, "diff_auc": a - b,
            }).to_csv(os.path.join(save_dir, "paired_auc.csv"), index=False)
            print(f"  Saved: paired_auc.csv  ({len(merged)} rows)")
        except Exception as exc:
            print(f"  SKIPPED — {exc}")

    # CNN vs CNN-LSTM
    ARCH_COMPARISONS = [
        {
            "name":    "4_cnn_vs_cnn_lstm",
            "title":   "Bland-Altman: CNN vs CNN-LSTM\n"
                       "(IG . interpolation . segment . identical sample set)",
            "label_a": "CNN", "label_b": "CNN-LSTM",
            "path_cnn":  ("cnn",      "ig", "interpolation", "segment"),
            "path_lstm": ("cnn_lstm", "ig", "interpolation", "segment"),
            "root":      "_ig_dir",
        },
        {
            "name":    "sg_cnn_vs_cnn_lstm",
            "title":   "Bland-Altman: CNN vs CNN-LSTM\n"
                       "(SmoothGrad-IG . interpolation . segment . identical sample set)",
            "label_a": "CNN", "label_b": "CNN-LSTM",
            "path_cnn":  ("cnn",      "smoothgrad_ig", "interpolation", "segment"),
            "path_lstm": ("cnn_lstm", "smoothgrad_ig", "interpolation", "segment"),
            "root":      "_smoothgrad_dir",
        },
        {
            "name":    "dtd_cnn_vs_cnn_lstm",
            "title":   "Bland-Altman: CNN vs CNN-LSTM\n"
                       "(DTD . interpolation . segment . identical sample set)",
            "label_a": "CNN", "label_b": "CNN-LSTM",
            "path_cnn":  ("cnn",      "dtd", "interpolation", "segment"),
            "path_lstm": ("cnn_lstm", "dtd", "interpolation", "segment"),
            "root":      "_dtd_dir",
        },
    ]

    for comp in ARCH_COMPARISONS:
        print(f"\n-- {comp['name']} --")
        try:
            r = root_map[comp["root"]]
            df_cnn  = _load(*comp["path_cnn"],  root=r)
            df_lstm = _load(*comp["path_lstm"], root=r)
            merged = df_cnn[["index", "auc_main"]].merge(
                df_lstm[["index", "auc_main"]], on="index",
                suffixes=("_cnn", "_lstm")
            )
            if len(merged) < len(df_cnn) or len(merged) < len(df_lstm):
                print(f"  WARNING: index lists differ — using intersection "
                      f"({len(merged)} samples).")
            else:
                print(f"  Confirmed: identical sample list ({len(merged)} samples) OK")
            a = merged["auc_main_cnn"].to_numpy(dtype=np.float64)
            b = merged["auc_main_lstm"].to_numpy(dtype=np.float64)
            save_dir = os.path.join(ba_root, comp["name"])
            os.makedirs(save_dir, exist_ok=True)
            bland_altman_analysis(
                a=a, b=b,
                label_a=comp["label_a"], label_b=comp["label_b"],
                save_dir=save_dir, title=comp["title"],
            )
            pd.DataFrame({
                "sample_idx": merged["index"].to_numpy(dtype=int),
                "auc_A": a, "auc_B": b,
                "mean_auc": (a + b) / 2.0, "diff_auc": a - b,
            }).to_csv(os.path.join(save_dir, "paired_auc.csv"), index=False)
            print(f"  Saved: paired_auc.csv  ({len(merged)} rows)")
        except Exception as exc:
            print(f"  SKIPPED — {exc}")

    # zero-out vs interpolation, MI samples only: interpolation is not applied to
    # NORM ECGs, so both measurements only exist for MI samples (sample_class == 1)
    ZEROOUT_VS_INTERP = [
        {
            "name":    "2_zero_out_vs_interpolation",
            "title":   "Bland-Altman: zero-out vs interpolation\n"
                       "(CNN-LSTM . IG . segment . MI samples only)",
            "label_a": "zero-out", "label_b": "interpolation",
            "path_a":  ("cnn_lstm", "ig", "zero_out",      "segment"),
            "path_b":  ("cnn_lstm", "ig", "interpolation", "segment"),
            "root":    "_ig_dir",
        },
        {
            "name":    "sg_zero_out_vs_interpolation",
            "title":   "Bland-Altman: zero-out vs interpolation\n"
                       "(CNN-LSTM . SmoothGrad-IG . segment . MI samples only)",
            "label_a": "zero-out", "label_b": "interpolation",
            "path_a":  ("cnn_lstm", "smoothgrad_ig", "zero_out",      "segment"),
            "path_b":  ("cnn_lstm", "smoothgrad_ig", "interpolation", "segment"),
            "root":    "_smoothgrad_dir",
        },
        {
            "name":    "dtd_zero_out_vs_interpolation",
            "title":   "Bland-Altman: zero-out vs interpolation\n"
                       "(CNN-LSTM . DTD . segment . MI samples only)",
            "label_a": "zero-out", "label_b": "interpolation",
            "path_a":  ("cnn_lstm", "dtd", "zero_out",      "segment"),
            "path_b":  ("cnn_lstm", "dtd", "interpolation", "segment"),
            "root":    "_dtd_dir",
        },
        {
            "name":    "cnn_ig_zero_out_vs_interpolation",
            "title":   "Bland-Altman: zero-out vs interpolation\n"
                       "(CNN . IG . segment . MI samples only)",
            "label_a": "zero-out", "label_b": "interpolation",
            "path_a":  ("cnn", "ig", "zero_out",      "segment"),
            "path_b":  ("cnn", "ig", "interpolation", "segment"),
            "root":    "_ig_dir",
        },
        {
            "name":    "cnn_sg_zero_out_vs_interpolation",
            "title":   "Bland-Altman: zero-out vs interpolation\n"
                       "(CNN . SmoothGrad-IG . segment . MI samples only)",
            "label_a": "zero-out", "label_b": "interpolation",
            "path_a":  ("cnn", "smoothgrad_ig", "zero_out",      "segment"),
            "path_b":  ("cnn", "smoothgrad_ig", "interpolation", "segment"),
            "root":    "_smoothgrad_dir",
        },
        {
            "name":    "cnn_dtd_zero_out_vs_interpolation",
            "title":   "Bland-Altman: zero-out vs interpolation\n"
                       "(CNN . DTD . segment . MI samples only)",
            "label_a": "zero-out", "label_b": "interpolation",
            "path_a":  ("cnn", "dtd", "zero_out",      "segment"),
            "path_b":  ("cnn", "dtd", "interpolation", "segment"),
            "root":    "_dtd_dir",
        },
    ]

    merge_on = ["index", "attribution_method", "eval_granularity"]

    for comp in ZEROOUT_VS_INTERP:
        print(f"\n-- {comp['name']} (MI samples only) --")
        try:
            r = root_map[comp["root"]]
            df_a = _load(*comp["path_a"], root=r, include_class=True)
            df_b = _load(*comp["path_b"], root=r, include_class=False)

            cols_a = merge_on + ["auc_main", "sample_class"]
            merged = (
                df_a[cols_a]
                .merge(df_b[merge_on + ["auc_main"]], on=merge_on,
                       suffixes=("_a", "_b"))
            )

            n_before = len(merged)
            merged = merged[merged["sample_class"] == 1].reset_index(drop=True)
            n_after = len(merged)
            print(f"  Paired on {merge_on}: {n_before} total → "
                  f"{n_after} MI samples retained (sample_class == 1)")

            a = merged["auc_main_a"].to_numpy(dtype=np.float64)
            b = merged["auc_main_b"].to_numpy(dtype=np.float64)

            save_dir = os.path.join(ba_root, comp["name"])
            bland_altman_analysis(
                a=a, b=b,
                label_a=comp["label_a"], label_b=comp["label_b"],
                save_dir=save_dir, title=comp["title"],
            )
            pd.DataFrame({
                "sample_idx":   merged["index"].to_numpy(dtype=int),
                "sample_class": merged["sample_class"].to_numpy(dtype=int),
                "auc_A": a, "auc_B": b,
                "mean_auc": (a + b) / 2.0, "diff_auc": a - b,
            }).to_csv(os.path.join(save_dir, "paired_auc.csv"), index=False)
            print(f"  Saved: paired_auc.csv  ({n_after} rows, MI only)")
        except Exception as exc:
            print(f"  SKIPPED — {exc}")

    # compare segment vs timestep
    GRAN_COMPARISONS = [
        {
            "name":    "3_segment_vs_timestep",
            "title":   "Bland-Altman: segment vs timestep\n"
                       "(CNN-LSTM . IG . interpolation)",
            "label_a": "segment", "label_b": "timestep",
            "path_a":  ("cnn_lstm", "ig", "interpolation", "segment"),
            "path_b":  ("cnn_lstm", "ig", "interpolation", "timestep"),
            "root":    "_ig_dir",
        },
        {
            "name":    "sg_segment_vs_timestep",
            "title":   "Bland-Altman: segment vs timestep\n"
                       "(CNN-LSTM . SmoothGrad-IG . interpolation)",
            "label_a": "segment", "label_b": "timestep",
            "path_a":  ("cnn_lstm", "smoothgrad_ig", "interpolation", "segment"),
            "path_b":  ("cnn_lstm", "smoothgrad_ig", "interpolation", "timestep"),
            "root":    "_smoothgrad_dir",
        },
        {
            "name":    "dtd_segment_vs_timestep",
            "title":   "Bland-Altman: segment vs timestep\n"
                       "(CNN-LSTM . DTD . interpolation)",
            "label_a": "segment", "label_b": "timestep",
            "path_a":  ("cnn_lstm", "dtd", "interpolation", "segment"),
            "path_b":  ("cnn_lstm", "dtd", "interpolation", "timestep"),
            "root":    "_dtd_dir",
        },
    ]

    for comp in GRAN_COMPARISONS:
        print(f"\n-- {comp['name']} --")
        try:
            r = root_map[comp["root"]]
            df_a = _load(*comp["path_a"], root=r)
            df_b = _load(*comp["path_b"], root=r)
            merged = (
                df_a[["index", "auc_main"]]
                .merge(df_b[["index", "auc_main"]], on="index", suffixes=("_a", "_b"))
            )
            a = merged["auc_main_a"].to_numpy(dtype=np.float64)
            b = merged["auc_main_b"].to_numpy(dtype=np.float64)
            print(f"  Paired on index: {len(merged)} samples")
            save_dir = os.path.join(ba_root, comp["name"])
            bland_altman_analysis(
                a=a, b=b,
                label_a=comp["label_a"], label_b=comp["label_b"],
                save_dir=save_dir, title=comp["title"],
            )
            pd.DataFrame({
                "sample_idx": merged["index"].to_numpy(dtype=int),
                "auc_A": a, "auc_B": b,
                "mean_auc": (a + b) / 2.0, "diff_auc": a - b,
            }).to_csv(os.path.join(save_dir, "paired_auc.csv"), index=False)
            print(f"  Saved: paired_auc.csv  ({len(merged)} rows)")
        except Exception as exc:
            print(f"  SKIPPED — {exc}")

    print(f"\n  Bland-Altman plots + reports saved to: {ba_root}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Bland-Altman analyses (MI-only variant). "
            "zero-out vs interpolation comparisons use MI samples only."
        )
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ig_results_dir",         default=None)
    parser.add_argument("--smoothgrad_results_dir", default=None)
    parser.add_argument("--dtd_results_dir",        default=None)
    args = parser.parse_args()

    run_bland_altman_analyses(
        args.out_dir,
        ig_results_dir=args.ig_results_dir,
        smoothgrad_results_dir=args.smoothgrad_results_dir,
        dtd_results_dir=args.dtd_results_dir,
    )


if __name__ == "__main__":
    main()
