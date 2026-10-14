"""
This part is related to Command-line entry point for the ECG XAI benchmark.
"""

import os
import sys
import types
import random
import argparse
import csv
from typing import Dict, List, Tuple, Optional


if __name__ == "__main__" and __package__ is None:
    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
    __package__ = "xai_benchmark"

import numpy as np
import torch
import torch.nn as nn

from .models import ECGCNN1D_Baseline, ECGCNN1D_Deeper, ECGCNNLSTMAttn
from .data_loading import load_data, build_model, load_checkpoint, _detect_cnn_arch_from_sd
from .norm_population import (
    compute_norm_ig_orders, compute_norm_specificity_curve,
    compute_mi_ig_orders,   compute_mi_sensitivity_curve,
)
from .experiment import run_experiment, generate_perturbation_comparison_plots
from .bland_altman import run_bland_altman_analyses
from .attribution.ig import integrated_gradients
from .metrics import completeness_check


def set_seed(seed: int) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _correct_indices(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    target_class: int,
    device,
) -> List[int]:
    # Return correctly classified samples for the target class
    model.eval()
    correct = []
    for idx in np.where(y == target_class)[0]:
        with torch.no_grad():
            x_t  = torch.tensor(X[idx][None], dtype=torch.float32, device=device)
            pred = int(torch.argmax(model(x_t), dim=1).item())
        if pred == target_class:
            correct.append(int(idx))
    return correct


def build_experiment_configs(global_args) -> List[Dict]:
    # Build experiment configurations from the CLI arguments

    MODELS = [
        {
            "name":        "cnn",
            "model_type":  "cnn",
            "cnn_arch":    "baseline",
            "ckpt":        global_args.cnn_ckpt,
            "dropout":     0.3,
            "lstm_hidden": 64,
            "attn_hidden": 64,
        },
        {
            "name":        "cnn_lstm",
            "model_type":  "cnn_lstm_attn",
            "cnn_arch":    "baseline",
            "ckpt":        global_args.lstm_ckpt,
            "dropout":     0.3,
            "lstm_hidden": 64,
            "attn_hidden": 64,
        },
    ]

    METHODS = [
        {
            "name":               "ig",
            "attribution_method": "ig",
            "sg_samples":         20,
            "sg_sigma":           0.05,
        },
        {
            "name":               "smoothgrad_ig",
            "attribution_method": "smoothgrad_ig",
            "sg_samples":         global_args.sg_samples,
            "sg_sigma":           0.05,
        },
        {
            "name":               "dtd",
            "attribution_method": "dtd",
            "sg_samples":         0,
            "sg_sigma":           0.0,
        },
    ]

    PERTURBATIONS = [
        {
            "name":         "zero_out",
            "perturb_mode": "zero_out",
            "baseline":     "zero",
            "interp_alpha": 0.5,
        },
        {
            "name":         "interpolation",
            "perturb_mode": "interpolate",
            "baseline":     "aligned_synth",
            "interp_alpha": 0.5,
        },
    ]

    GRANULARITIES = [
        {
            "name":             "segment",
            "eval_granularity": "segment",
            "n_segments":       50,
            "chunk_size":       10,
        },
        {
            "name":             "timestep",
            "eval_granularity": "timestep",
            "n_segments":       50,
            "chunk_size":       10,
        },
    ]

    allowed_models   = set(global_args.models_to_run)
    allowed_methods  = set(global_args.methods)
    allowed_perturbs = set(global_args.perturbations)
    allowed_grans    = set(global_args.granularities)

    experiments = []
    for model in MODELS:
        if model["name"] not in allowed_models:
            continue
        for method in METHODS:
            if method["name"] not in allowed_methods:
                continue
            for perturb in PERTURBATIONS:
                if perturb["name"] not in allowed_perturbs:
                    continue
                for gran in GRANULARITIES:
                    if gran["name"] not in allowed_grans:
                        continue
                    experiments.append({
                        "folder_model":   model["name"],
                        "folder_method":  method["name"],
                        "folder_perturb": perturb["name"],
                        "folder_gran":    gran["name"],
                        "model_type":     model["model_type"],
                        "cnn_arch":       model["cnn_arch"],
                        "ckpt":           model["ckpt"],
                        "dropout":        model["dropout"],
                        "lstm_hidden":    model["lstm_hidden"],
                        "attn_hidden":    model["attn_hidden"],
                        "attribution_method": method["attribution_method"],
                        "sg_samples":         method["sg_samples"],
                        "sg_sigma":           method["sg_sigma"],
                        "perturb_mode":   perturb["perturb_mode"],
                        "baseline":       perturb["baseline"],
                        "interp_alpha":   perturb["interp_alpha"],
                        "eval_granularity": gran["eval_granularity"],
                        "n_segments":       gran["n_segments"],
                        "chunk_size":       gran["chunk_size"],
                    })
    return experiments


def run_completeness_check_mode(global_args):
    """ Evaluate IG completeness without running the perturbation benchmark."""
    device = torch.device(
        global_args.device
        if (global_args.device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )
    print(f"\nDevice: {device}")

    print("\n[Completeness] Loading test set...")
    X, y, _, _ = load_data(global_args.data_dir, split=global_args.split)
    print(f"  X: {X.shape}  y: {y.shape}")

    model_specs = []
    if "cnn" in global_args.models_to_run:
        model_specs.append({
            "name":        "cnn",
            "model_type":  "cnn",
            "cnn_arch":    "baseline",
            "ckpt":        global_args.cnn_ckpt,
            "dropout":     0.3,
            "lstm_hidden": 64,
            "attn_hidden": 64,
        })
    if "cnn_lstm" in global_args.models_to_run:
        model_specs.append({
            "name":        "cnn_lstm",
            "model_type":  "cnn_lstm_attn",
            "cnn_arch":    "baseline",
            "ckpt":        global_args.lstm_ckpt,
            "dropout":     0.3,
            "lstm_hidden": 64,
            "attn_hidden": 64,
        })

    mi_indices = sorted(np.where(y == global_args.target_class)[0].tolist())
    n_check    = global_args.completeness_samples
    if n_check > 0:
        mi_indices = mi_indices[:n_check]
    print(f"\n[Completeness] Checking {len(mi_indices)} MI samples "
          f"(target_class={global_args.target_class}, zero baseline)")

    for spec in model_specs:
        print(f"\n{'='*60}")
        print(f"  Model: {spec['name']}  ckpt={spec['ckpt']}")
        print(f"{'='*60}")

        if spec["model_type"] == "cnn":
            _sd_raw = torch.load(spec["ckpt"], map_location="cpu")
            if isinstance(_sd_raw, dict) and "model_state_dict" in _sd_raw:
                _sd_raw = _sd_raw["model_state_dict"]
            elif isinstance(_sd_raw, dict) and "state_dict" in _sd_raw:
                _sd_raw = _sd_raw["state_dict"]
            _detected = _detect_cnn_arch_from_sd(_sd_raw)
            if _detected != "unknown":
                spec["cnn_arch"] = _detected

        model_raw = build_model(
            model_type=spec["model_type"],
            num_classes=global_args.num_classes,
            cnn_arch=spec["cnn_arch"],
            dropout=spec["dropout"],
            lstm_hidden=spec["lstm_hidden"],
            attn_hidden=spec["attn_hidden"],
        )
        model = load_checkpoint(
            model_raw, spec["ckpt"], device,
            expected_arch=spec["cnn_arch"] if spec["model_type"] == "cnn" else "",
        )
        model.eval()

        records = []
        for sample_idx in mi_indices:
            x        = X[sample_idx]
            baseline = np.zeros_like(x)
            tc       = global_args.target_class

            ig_attr = integrated_gradients(
                model, x, tc, device,
                baseline=baseline,
                steps=global_args.ig_steps,
            )

            result = completeness_check(
                model, x, ig_attr, tc, device, baseline=baseline
            )

            abs_err = abs(result["ig_sum"] - result["diff"])
            records.append({
                "sample_idx":  sample_idx,
                "target_class": tc,
                "ig_sum":      result["ig_sum"],
                "f_x":         result["f_x"],
                "f_baseline":  result["f_baseline"],
                "output_diff": result["diff"],
                "abs_error":   abs_err,
                "rel_error":   result["rel_error"],
            })
            print(f"  idx={sample_idx:5d}  ig_sum={result['ig_sum']:+.6f}  "
                  f"f(x)-f(b)={result['diff']:+.6f}  "
                  f"rel_err={result['rel_error']:.2e}")

        model_out = os.path.join(global_args.out_dir, spec["name"])
        os.makedirs(model_out, exist_ok=True)
        csv_path = os.path.join(model_out, "completeness_results.csv")
        fieldnames = ["sample_idx", "target_class", "ig_sum", "f_x",
                      "f_baseline", "output_diff", "abs_error", "rel_error"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        rel_errors = np.array([r["rel_error"] for r in records], dtype=np.float64)
        n          = len(rel_errors)
        pct_1e2    = 100.0 * np.mean(rel_errors < 1e-2)
        pct_5e2    = 100.0 * np.mean(rel_errors < 5e-2)

        summary_lines = [
            f"IG Completeness Check — {spec['name']}",
            f"{'='*50}",
            f"n_samples        : {n}",
            f"target_class     : {global_args.target_class}",
            f"baseline         : zeros (zero_out mode)",
            f"ig_steps         : {global_args.ig_steps}",
            f"",
            f"Relative error  (|sum(IG) - (f(x)-f(b))| / (|f(x)-f(b)| + 1e-8)):",
            f"  mean           : {rel_errors.mean():.4e}",
            f"  median         : {float(np.median(rel_errors)):.4e}",
            f"  std            : {rel_errors.std():.4e}",
            f"  min            : {rel_errors.min():.4e}",
            f"  max            : {rel_errors.max():.4e}",
            f"  % < 1e-2       : {pct_1e2:.1f}%",
            f"  % < 5e-2       : {pct_5e2:.1f}%",
            f"",
            f"Interpretation:",
            f"  rel_error < 1e-2 (~1%)  → excellent completeness",
            f"  rel_error < 5e-2 (~5%)  → acceptable for midpoint-rule IG",
            f"  rel_error > 5e-2        → check ig_steps (try --ig_steps 100)",
        ]
        txt_path = os.path.join(model_out, "completeness_summary.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")

        print(f"\n  --- Completeness summary ({spec['name']}) ---")
        for line in summary_lines:
            print(f"  {line}")
        print(f"\n  Results saved to:")
        print(f"    {csv_path}")
        print(f"    {txt_path}")


def run_all_experiments(global_args):
    """Load both models once, precompute the population curves, and run every experiment."""
    device = torch.device(
        global_args.device
        if (global_args.device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )
    print(f"\nDevice: {device}")

    print("\n[Data] Loading test set...")
    X, y, X_means, X_stds = load_data(global_args.data_dir, split=global_args.split)
    print(f"  X: {X.shape}  y: {y.shape}")

    experiments = build_experiment_configs(global_args)
    total       = len(experiments)

    if total == 0:
        print("\nERROR: No experiments to run after applying filters. "
              "Check --methods / --perturbations / --granularities / --models_to_run.")
        return

    print(f"\n[Config] Running {total} experiment(s) with filters:")
    print(f"  models       : {global_args.models_to_run}")
    print(f"  methods      : {global_args.methods}")
    print(f"  perturbations: {global_args.perturbations}")
    print(f"  granularities: {global_args.granularities}")

    print("\n[Models] Loading all models upfront...")
    loaded_models: Dict[str, nn.Module] = {}
    seen_ckpts = {}
    for exp in experiments:
        key = exp["ckpt"]
        if key not in seen_ckpts:
            seen_ckpts[key] = exp

    for key, exp in seen_ckpts.items():
        print(f"  Loading {exp['model_type']}  ckpt={key}")

        if exp["model_type"] == "cnn":
            _sd_raw = torch.load(key, map_location="cpu")
            if isinstance(_sd_raw, dict) and "model_state_dict" in _sd_raw:
                _sd_raw = _sd_raw["model_state_dict"]
            elif isinstance(_sd_raw, dict) and "state_dict" in _sd_raw:
                _sd_raw = _sd_raw["state_dict"]
            _detected = _detect_cnn_arch_from_sd(_sd_raw)
            if _detected != "unknown":
                exp["cnn_arch"] = _detected
                print(f"  Auto-detected CNN arch from checkpoint: {_detected}")

        model_raw = build_model(
            model_type=exp["model_type"],
            num_classes=global_args.num_classes,
            cnn_arch=exp["cnn_arch"],
            dropout=exp["dropout"],
            lstm_hidden=exp["lstm_hidden"],
            attn_hidden=exp["attn_hidden"],
        )
        loaded_models[key] = load_checkpoint(
            model_raw, key, device,
            expected_arch=exp["cnn_arch"] if exp["model_type"] == "cnn" else "",
        )

    all_mi_indices   = sorted(np.where(y == global_args.target_class)[0].tolist())
    all_norm_indices = np.where(y == 0)[0]

    if global_args.max_norm_samples > 0 and len(all_norm_indices) > global_args.max_norm_samples:
        rng = np.random.default_rng(global_args.seed)
        norm_curve_indices = np.sort(
            rng.choice(all_norm_indices, size=global_args.max_norm_samples, replace=False))
        print(f"\n[Samples] NORM population curve: using {len(norm_curve_indices)} "
              f"of {len(all_norm_indices)} (--max_norm_samples)")
    else:
        norm_curve_indices = all_norm_indices
        print(f"\n[Samples] NORM population curve: using all {len(norm_curve_indices)} patients")

    X_norm_curve = X[norm_curve_indices]
    X_mi_curve   = X[all_mi_indices]

    print(f"\n[Samples] Finding correctly classified samples (intersection across all models)...")
    per_model_correct_mi:   Dict[str, set] = {}
    per_model_correct_norm: Dict[str, set] = {}

    for key, mdl in loaded_models.items():
        correct_mi   = set(_correct_indices(mdl, X, y, global_args.target_class, device))
        correct_norm = set(_correct_indices(mdl, X, y, 0, device))
        per_model_correct_mi[key]   = correct_mi
        per_model_correct_norm[key] = correct_norm
        print(f"  {seen_ckpts[key]['model_type']:20s}  "
              f"MI correct={len(correct_mi)}/{len(all_mi_indices)}  "
              f"NORM correct={len(correct_norm)}/{len(all_norm_indices)}")

    common_mi   = set(all_mi_indices)
    common_norm = set(all_norm_indices.tolist())
    for s in per_model_correct_mi.values():
        common_mi &= s
    for s in per_model_correct_norm.values():
        common_norm &= s

    # We evaluate on every test sample (MI + NORM)
    selected: List[int] = sorted(all_mi_indices + all_norm_indices.tolist())
    if global_args.max_samples > 0:
        selected = selected[:global_args.max_samples]

    print(f"\n  All test samples: {len(selected)}"
          f"  (MI={len([i for i in selected if y[i]==1])}, "
          f"NORM={len([i for i in selected if y[i]==0])})")

    os.makedirs(global_args.out_dir, exist_ok=True)

    # curves are cached per (ckpt, method, granularity) and reused across perturb modes
    print(f"\n[Population curves] Computing NORM specificity + MI sensitivity curves...")
    print(f"  Done ONCE per (model, method, granularity). Reused across perturbation modes.")

    norm_curve_cache: Dict[Tuple, np.ndarray] = {}
    mi_curve_cache:   Dict[Tuple, np.ndarray] = {}

    for exp in experiments:
        cache_key = (exp["ckpt"], exp["attribution_method"], exp["eval_granularity"])
        if cache_key in norm_curve_cache:
            continue

        mdl = loaded_models[exp["ckpt"]]

        print(f"\n  [curves] ckpt={os.path.basename(exp['ckpt'])}  "
              f"method={exp['attribution_method']}  gran={exp['eval_granularity']}")

        pop_cfg = types.SimpleNamespace(
            attribution_method = exp["attribution_method"],
            ig_steps           = global_args.ig_steps,
            sg_samples         = exp["sg_samples"],
            sg_sigma           = exp["sg_sigma"],
            eval_granularity   = exp["eval_granularity"],
            n_segments         = exp["n_segments"],
            chunk_size         = exp["chunk_size"],
            timestep_score     = "mean_abs",
            edge_mask          = global_args.edge_mask,
        )

        set_seed(global_args.seed)
        norm_orders = compute_norm_ig_orders(mdl, X_norm_curve, device, pop_cfg)
        norm_curve  = compute_norm_specificity_curve(mdl, X_norm_curve, norm_orders, device, pop_cfg)

        set_seed(global_args.seed)
        mi_orders   = compute_mi_ig_orders(mdl, X_mi_curve, device, pop_cfg)
        mi_curve    = compute_mi_sensitivity_curve(mdl, X_mi_curve, mi_orders, device, pop_cfg)

        norm_curve_cache[cache_key] = norm_curve
        mi_curve_cache[cache_key]   = mi_curve

        print(f"    NORM specificity: step0={norm_curve[0]:.4f}  "
              f"mid={norm_curve[len(norm_curve)//2]:.4f}  last={norm_curve[-1]:.4f}")
        print(f"    MI sensitivity:   step0={mi_curve[0]:.4f}  "
              f"mid={mi_curve[len(mi_curve)//2]:.4f}  last={mi_curve[-1]:.4f}")

    # fixed plot indices for visualization
    VIS_IDX = {31, 140, 141, 153}
    viz_indices = sorted(VIS_IDX)
    print(f"\n[Viz] Fixed visualization indices: {viz_indices}")

    run_number = 0
    for exp in experiments:
        run_number += 1

        out_dir = os.path.join(
            global_args.out_dir,
            exp["folder_model"],
            exp["folder_method"],
            exp["folder_perturb"],
            exp["folder_gran"],
        )
        os.makedirs(out_dir, exist_ok=True)

        label = (f"[{run_number:02d}/{total}] "
                 f"{exp['folder_model']} | {exp['folder_method']} | "
                 f"{exp['folder_perturb']} | {exp['folder_gran']}")
        print(f"\n{'#'*65}")
        print(f"  {label}")
        print(f"  Total samples: {len(selected)}"
              f"  (MI={len([i for i in selected if y[i]==1])}, "
              f"NORM={len([i for i in selected if y[i]==0])})")
        print(f"{'#'*65}")

        cache_key = (exp["ckpt"], exp["attribution_method"], exp["eval_granularity"])
        norm_specificity_curve = norm_curve_cache[cache_key]
        mi_sensitivity_curve   = mi_curve_cache[cache_key]

        cfg = types.SimpleNamespace(
            model_type         = exp["model_type"],
            num_classes        = global_args.num_classes,
            attribution_method = exp["attribution_method"],
            ig_steps           = global_args.ig_steps,
            sg_samples         = exp["sg_samples"],
            sg_sigma           = exp["sg_sigma"],
            perturb_mode       = exp["perturb_mode"],
            baseline           = exp["baseline"],
            interp_alpha       = exp["interp_alpha"],
            eval_granularity   = exp["eval_granularity"],
            n_segments         = exp["n_segments"],
            chunk_size         = exp["chunk_size"],
            timestep_score     = "mean_abs",
            synth_method       = global_args.synth_method,
            per_lead_phase     = True,
            seed               = global_args.seed,
            target_class       = global_args.target_class,
            lead_idx           = global_args.lead_idx,
            fs                 = global_args.fs,
            edge_mask          = global_args.edge_mask,
            random_seed        = global_args.random_seed,
        )

        set_seed(global_args.seed)
        run_experiment(
            cfg=cfg,
            model=loaded_models[exp["ckpt"]],
            X=X,
            y=y,
            norm_specificity_curve=norm_specificity_curve,
            mi_sensitivity_curve=mi_sensitivity_curve,
            selected=selected,
            device=device,
            out_dir=out_dir,
            X_means=X_means,
            X_stds=X_stds,
            viz_indices=viz_indices,
        )

    n_mi_all   = len([i for i in selected if y[i] == 1])
    n_norm_all = len([i for i in selected if y[i] == 0])
    print(f"\n{'='*65}")
    print(f"  ALL {total} EXPERIMENTS COMPLETE")
    print(f"  Samples processed: {len(selected)} (MI={n_mi_all}, NORM={n_norm_all})")
    print(f"  Results -> <exp_dir>/")
    print(f"  Root: {global_args.out_dir}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the population-level ECG XAI perturbation benchmark."
        )
    

    parser.add_argument("--data_dir",  required=True,
                        help="Directory containing X_test.npy / y_test.npy")
    parser.add_argument("--cnn_ckpt",  required=True,
                        help="Path to CNN best.pt checkpoint")
    parser.add_argument("--lstm_ckpt", required=True,
                        help="Path to CNN-LSTM-Attn best.pt checkpoint")
    parser.add_argument("--out_dir",   default="results/population_v2",
                        help="Root output directory")

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ig", "smoothgrad_ig", "dtd"],
        choices=["ig", "smoothgrad_ig", "dtd"],
        help="Attribution methods to run (default: all)",
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=["zero_out", "interpolation"],
        choices=["zero_out", "interpolation"],
        help="Perturbation modes to run (default: both)",
    )
    parser.add_argument(
        "--granularities",
        nargs="+",
        default=["segment", "timestep"],
        choices=["segment", "timestep"],
        help="Evaluation granularities to run (default: both)",
    )
    parser.add_argument(
        "--models_to_run",
        nargs="+",
        default=["cnn", "cnn_lstm"],
        choices=["cnn", "cnn_lstm"],
        help="Model architectures to run (default: both)",
    )

    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--seed",        type=int, default=9999)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--split",       default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--target_class",type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max MI samples to use (-1 = all correctly classified)")
    parser.add_argument("--max_norm_samples", type=int, default=-1,
                        help="Max NORM samples for population specificity "
                             "(-1 = all). Reduce (e.g. 200) for faster runs.")

    parser.add_argument("--ig_steps",  type=int, default=50)
    parser.add_argument("--sg_samples", type=int, default=5)
    parser.add_argument("--edge_mask", type=int, default=30)

    parser.add_argument("--synth_method", default="ecgsyn",
                        choices=["ecgsyn", "simple"])

    parser.add_argument("--fs",       type=int, default=100)
    parser.add_argument("--lead_idx", type=int, default=1)

    parser.add_argument("--ig_results_dir",         default=None,
                        help="Results dir for IG ")
    parser.add_argument("--smoothgrad_results_dir",  default=None,
                        help="Results dir for SmoothGrad.")
    parser.add_argument("--dtd_results_dir",         default=None,
                        help="Results dir for DTD.")

    parser.add_argument(
        "--check_completeness",
        action="store_true",
        default=False,
        help="We Run IG completeness check instead of the full benchmark. "
             ,
    )
    parser.add_argument(
        "--completeness_samples",
        type=int,
        default=20,
        help="Number of MI samples to use for the completeness check in our experiment is  (default: 20). "
             "Set to -1 if you want  use all MI test samples.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if args.check_completeness:
        run_completeness_check_mode(args)
    else:
        run_all_experiments(args)
        generate_perturbation_comparison_plots(args.out_dir)
        run_bland_altman_analyses(
            args.out_dir,
            ig_results_dir=args.ig_results_dir,
            smoothgrad_results_dir=args.smoothgrad_results_dir,
            dtd_results_dir=args.dtd_results_dir,
        )


if __name__ == "__main__":
    main()
