
import os
import signal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .metrics import (
    predict_prob,
    compute_auc,
    compute_ba_curve_population,
    completeness_check,
    apply_edge_mask,
)
from .signal_utils import estimate_hr_bpm, make_synthetic_healthy_ecg, align_synthetic_to_mi
from .perturbation import (
    segment_importance,
    timestep_importance,
    replace_segments_stepwise,
    replace_timesteps_stepwise,
    random_order,
)
from .attribution import compute_attribution
from .plotting import (
    _method_label,
    plot_mean_curves,
    plot_auc_boxplot,
    write_final_report,
    plot_attribution_heatmap,
    plot_perturbation_comparison,
    plot_perturbation_heatmap_comparison,
    plot_perturbation_heatmap_single,
)


class _SampleTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _SampleTimeout()


def run_one(
    cfg,
    model,
    x_ecg: np.ndarray,
    cross_class_curve: np.ndarray,
    sample_idx: int,
    sample_class: int,
    device,
    true_zero: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run the perturbation benchmark for a single sample.

    MI and NORM samples follow different perturbation strategies.
    """
    p_start = predict_prob(model, x_ecg, device, sample_class)

    if sample_class == 1:
       
        hr = estimate_hr_bpm(x_ecg[cfg.lead_idx], fs=cfg.fs)
        if np.isnan(hr) or hr <= 0:
            hr = 75.0
        hr = float(np.clip(hr, 45.0, 140.0))
        x_ref = make_synthetic_healthy_ecg(
            hr=hr, fs=cfg.fs, T=x_ecg.shape[1], n_leads=x_ecg.shape[0],
            method=cfg.synth_method, per_lead_phase=cfg.per_lead_phase,
            seed=cfg.seed + sample_idx,
        )
        x_ref, shift = align_synthetic_to_mi(
            x_mi=x_ecg, x_syn=x_ref, fs=cfg.fs, lead_idx=cfg.lead_idx,
        )
        perturb_mode = cfg.perturb_mode
        baseline_arr = np.zeros_like(x_ecg) if cfg.baseline == "zero" else x_ref.copy()
    else:
       
        x_ref        = np.zeros_like(x_ecg)
        shift        = 0
        hr           = float("nan")
        perturb_mode = "zero_out"
        baseline_arr = np.zeros_like(x_ecg)

    p_ref = predict_prob(model, x_ref, device, sample_class)

    ig = compute_attribution(
        model=model, x=x_ecg, target_class=sample_class,
        device=device, baseline=baseline_arr, cfg=cfg,
    )

    if cfg.attribution_method == "ig":
        chk = completeness_check(model, x_ecg, ig, sample_class, device, baseline_arr)
    else:
        chk = {"rel_error": float("nan")}

    ig_scored = apply_edge_mask(ig, mask_size=cfg.edge_mask)

    if cfg.eval_granularity == "segment":
        order, bounds, _ = segment_importance(ig_scored, n_segments=cfg.n_segments)
        X_main = replace_segments_stepwise(
            x_ecg, x_ref, order, bounds, perturb_mode, cfg.interp_alpha,
            true_zero=true_zero)
        rand_ord = random_order(cfg.n_segments, seed=cfg.random_seed + sample_idx)
        X_rand   = replace_segments_stepwise(
            x_ecg, x_ref, rand_ord, bounds, perturb_mode, cfg.interp_alpha,
            true_zero=true_zero)
        n_steps = len(order)

    elif cfg.eval_granularity == "timestep":
        order, _ = timestep_importance(ig_scored, mode=cfg.timestep_score)
        X_main = replace_timesteps_stepwise(
            x_ecg, x_ref, order,
            chunk_size=cfg.chunk_size, perturb_mode=perturb_mode,
            interp_alpha=cfg.interp_alpha, true_zero=true_zero)
        rand_ord = random_order(x_ecg.shape[1], seed=cfg.random_seed + sample_idx)
        X_rand   = replace_timesteps_stepwise(
            x_ecg, x_ref, rand_ord,
            chunk_size=cfg.chunk_size, perturb_mode=perturb_mode,
            interp_alpha=cfg.interp_alpha, true_zero=true_zero)
        n_steps = len(X_main) - 1

    else:
        raise ValueError(f"Unknown eval_granularity: {cfg.eval_granularity!r}")

    probs_main = compute_ba_curve_population(
        model, X_main, cross_class_curve, device, sample_class=sample_class)
    probs_rand = compute_ba_curve_population(
        model, X_rand,  cross_class_curve, device, sample_class=sample_class)
    auc_main   = compute_auc(probs_main)
    auc_rand   = compute_auc(probs_rand)

    # purturbe signal between  30% and 50%  for the visualizations
    step_30_idx = max(0, min(len(X_main) - 1, int(n_steps * 0.3)))
    step_50_idx = max(0, min(len(X_main) - 1, int(n_steps * 0.5)))

    return {
        "index":                  int(sample_idx),
        "sample_class":           int(sample_class),
        "hr":                     float(hr),
        "shift":                  int(shift),
        "p_start":                float(p_start),
        "p_ref":                  float(p_ref),
        "completeness_rel_error": float(chk["rel_error"]),
        "auc_main":               float(auc_main),
        "auc_random":             float(auc_rand),
        "ig_better":              bool(auc_main < auc_rand),
        "probs_main":             probs_main,
        "probs_random":           probs_rand,
        "x_ecg":                  x_ecg,
        "x_ref":                  x_ref,
        "attr":                   ig_scored,
        "x_perturbed_30":         X_main[step_30_idx].copy(),
        "x_zero_out_50":          X_main[step_50_idx].copy() if perturb_mode == "zero_out"
                                  else None,
        "x_interp_50":            X_main[step_50_idx].copy() if perturb_mode == "interpolate"
                                  else None,
    }


def run_one_with_timeout(
    cfg,
    model,
    x_ecg: np.ndarray,
    cross_class_curve: np.ndarray,
    sample_idx: int,
    sample_class: int,
    device,
    timeout_sec: int = 120,
    true_zero: Optional[np.ndarray] = None,
):
    #Run run_one() with a hard timeout; returns None on timeout or error
    kwargs = dict(cfg=cfg, model=model, x_ecg=x_ecg,
                  cross_class_curve=cross_class_curve,
                  sample_idx=sample_idx, sample_class=sample_class,
                  device=device, true_zero=true_zero)

    if not hasattr(signal, "SIGALRM"):
        try:
            return run_one(**kwargs)
        except Exception as exc:
            print(f"  WARNING: Sample idx={sample_idx} ERROR: {exc} — SKIPPING")
            return None

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = run_one(**kwargs)
        signal.alarm(0)
        return result
    except _SampleTimeout:
        print(f"\n  WARNING: Sample idx={sample_idx} timed out after {timeout_sec}s — SKIPPING")
        return None
    except Exception as exc:
        signal.alarm(0)
        print(f"\n  WARNING: Sample idx={sample_idx} ERROR: {exc} — SKIPPING")
        return None
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def run_experiment(
    cfg,
    model,
    X: np.ndarray,
    y: np.ndarray,
    norm_specificity_curve: np.ndarray,
    mi_sensitivity_curve: np.ndarray,
    selected: List[int],
    device,
    out_dir: str,
    X_means: Optional[np.ndarray] = None,
    X_stds:  Optional[np.ndarray] = None,
    viz_indices: Optional[List[int]] = None,
):
    """
    Run the benchmark over the selected test samples.
    Class labels and precomputed cross-class curves are used to construct the population-level balanced accuracy.
    """
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, "visualizations")

    # In this part Share visualization data across perturbation modes
    method_dir   = os.path.dirname(os.path.dirname(out_dir))
    viz_cache_dir = os.path.join(method_dir, "_viz_cache", cfg.eval_granularity)
    os.makedirs(viz_cache_dir, exist_ok=True)

    rows:                 List[Dict]       = []
    curves_main:          List[np.ndarray] = []
    curves_rand:          List[np.ndarray] = []
    auc_main_list:        List[float]      = []
    auc_rand_list:        List[float]      = []
    completeness_errs:    List[float]      = []
    x_mi_samples:         List[np.ndarray] = []
    x_syn_samples:        List[np.ndarray] = []
    attr_maps:            List[np.ndarray] = []
    x_perturbed_30_samp:  List[np.ndarray] = []
    skipped_indices:      List[int]        = []

    m_label  = _method_label(cfg.attribution_method)
    _viz_set = set(viz_indices) if viz_indices is not None else set()

    for k, idx in enumerate(selected):
        sample_class = int(y[idx])
        class_label  = "MI" if sample_class == 1 else "NORM"
        cross_class_curve = (norm_specificity_curve if sample_class == 1
                             else mi_sensitivity_curve)

        print(f"    [{k+1}/{len(selected)}] idx={idx} [{class_label}]  processing...",
              flush=True)

        if X_means is not None and X_stds is not None:
            true_zero = -X_means[idx] / (X_stds[idx] + 1e-8)
        else:
            true_zero = None

        res = run_one_with_timeout(
            cfg=cfg, model=model, x_ecg=X[idx],
            cross_class_curve=cross_class_curve,
            sample_idx=idx, sample_class=sample_class,
            device=device, timeout_sec=120,
            true_zero=true_zero,
        )

        if res is None:
            skipped_indices.append(idx)
            print(f"    [{k+1}/{len(selected)}] idx={idx}  SKIPPED", flush=True)
            continue

        rows.append({
            "index":                  res["index"],
            "sample_class":           res["sample_class"],
            "eval_granularity":       cfg.eval_granularity,
            "perturb_mode":           cfg.perturb_mode,
            "attribution_method":     cfg.attribution_method,
            "hr":                     res["hr"],
            "shift":                  res["shift"],
            "p_start":                res["p_start"],
            "p_ref":                  res["p_ref"],
            "completeness_rel_error": res["completeness_rel_error"],
            "auc_main":               res["auc_main"],
            "auc_random":             res["auc_random"],
            "ig_better":              res["ig_better"],
        })
        curves_main.append(res["probs_main"])
        curves_rand.append(res["probs_random"])
        auc_main_list.append(res["auc_main"])
        auc_rand_list.append(res["auc_random"])
        completeness_errs.append(res["completeness_rel_error"])
        x_mi_samples.append(res["x_ecg"])
        x_syn_samples.append(res["x_ref"])
        attr_maps.append(res["attr"])
        x_perturbed_30_samp.append(res["x_perturbed_30"])

        
        _VIS_IDX = {21, 31, 140, 141, 153}
        if idx in _VIS_IDX:
            x_ecg_cache = os.path.join(viz_cache_dir, f"x_mi_idx{idx}.npy")
            if not os.path.exists(x_ecg_cache):
                np.save(x_ecg_cache, res["x_ecg"])

            attr_cache = os.path.join(viz_cache_dir, f"attr_idx{idx}.npy")
            if not os.path.exists(attr_cache):
                np.save(attr_cache, res["attr"])

            if res.get("x_zero_out_50") is not None:
                np.save(os.path.join(viz_cache_dir, f"zero_out_50_idx{idx}.npy"),
                        res["x_zero_out_50"])

            if res.get("x_interp_50") is not None:
                np.save(os.path.join(viz_cache_dir, f"interp_50_idx{idx}.npy"),
                        res["x_interp_50"])

        # Generate visualizations for the selected samples.
        if idx in _viz_set:
            print(f"    [Viz] Generating visualizations for dataset idx={idx} ...")
            try:
                plot_attribution_heatmap(
                    attr=res["attr"],
                    x_mi=res["x_ecg"],
                    sample_idx=idx,
                    method_label=m_label,
                    save_dir=viz_dir,
                    leads=[0, 1, 5],
                )
                print(f"    [Viz] Saved: attribution_heatmap_idx{idx}.png")
            except Exception as exc:
                print(f"    [Viz] ERROR attribution_heatmap idx={idx}: {exc}")

            x_perturbed_50 = res.get("x_zero_out_50")
            if x_perturbed_50 is None:
                x_perturbed_50 = res.get("x_interp_50")
            if x_perturbed_50 is not None:
                mode_str = "zero_out" if cfg.perturb_mode == "zero_out" else "interp"
                try:
                    plot_perturbation_heatmap_single(
                        attr=res["attr"],
                        x_original=res["x_ecg"],
                        x_perturbed=x_perturbed_50,
                        sample_idx=idx,
                        method_label=m_label,
                        perturb_mode=cfg.perturb_mode,
                        save_dir=viz_dir,
                        leads=[0, 1, 5],
                    )
                    print(f"    [Viz] Saved: perturbation_heatmap_{mode_str}_idx{idx}.png")
                except Exception as exc:
                    print(f"    [Viz] ERROR perturbation_heatmap idx={idx}: {exc}")

        n_better = sum(r["ig_better"] for r in rows)
        print(
            f"    [{k+1}/{len(selected)}] idx={idx} [{class_label}]  "
            f"AUC_m={res['auc_main']:.3f}  AUC_r={res['auc_random']:.3f}  "
            f"comp_err={res['completeness_rel_error']:.2%}  "
            f"better={n_better}/{len(rows)}", flush=True
        )

    if skipped_indices:
        print(f"\n  WARNING: Skipped {len(skipped_indices)} sample(s): {skipped_indices}")

    if _viz_set:
        processed_indices = {r["index"] for r in rows}
        for _vi in sorted(_viz_set):
            if _vi in processed_indices:
                print(f"  [Viz] idx={_vi} — visualization files GENERATED in {viz_dir}")
            elif _vi in set(skipped_indices):
                print(f"  [Viz] idx={_vi} — SKIPPED (sample timed out or errored)")
            else:
                print(f"  [Viz] idx={_vi} — NOT FOUND in selected samples")

    if not rows:
        print("  ERROR: All samples were skipped — no results to save.")
        return

    os.makedirs(out_dir, exist_ok=True)
    n = len(rows)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    plot_mean_curves(curves_main, curves_rand,
                     os.path.join(out_dir, "mean_curve.png"),
                     method_label=m_label,
                     gran=cfg.eval_granularity,
                     perturb_mode=cfg.perturb_mode)
    plot_auc_boxplot(np.asarray(auc_main_list), np.asarray(auc_rand_list),
                     os.path.join(out_dir, "auc_boxplot.png"),
                     method_label=m_label,
                     gran=cfg.eval_granularity,
                     perturb_mode=cfg.perturb_mode)
    write_final_report(os.path.join(out_dir, "final_report.txt"),
                       cfg=cfg,
                       auc_main=np.asarray(auc_main_list),
                       auc_rand=np.asarray(auc_rand_list),
                       completeness_errs=completeness_errs)
    print(f"  [all_samples] Saved to: {out_dir}  (n={n})")


def generate_perturbation_comparison_plots(
    out_dir: str,
    leads: List[int] = [0, 1, 5],
) -> None:
    
    
 #Generate perturbation comparison plots from cached results.

    print(f"\n[Perturbation comparison plots] Scanning {out_dir} ...")

    def _parse_indices(file_list, prefix):
        result = set()
        for f in file_list:
            if f.startswith(prefix) and f.endswith(".npy"):
                try:
                    result.add(int(f[len(prefix):-4]))
                except ValueError:
                    pass
        return result

    for root, dirs, _ in os.walk(out_dir):
        for d in dirs:
            if d != "_viz_cache":
                continue
            cache_parent = os.path.join(root, d)
            for gran_name in os.listdir(cache_parent):
                cache_dir = os.path.join(cache_parent, gran_name)
                if not os.path.isdir(cache_dir):
                    continue

                files_in_cache = os.listdir(cache_dir)

                zero_out_indices = _parse_indices(files_in_cache, "zero_out_50_idx")
                interp_indices   = _parse_indices(files_in_cache, "interp_50_idx")
                attr_indices     = _parse_indices(files_in_cache, "attr_idx")

                paired = sorted(zero_out_indices & interp_indices & attr_indices)
                if not paired:
                    missing = sorted(zero_out_indices | interp_indices)
                    print(f"  [skip] {cache_dir} — no fully paired samples "
                          f"(zero_out={len(zero_out_indices)}, "
                          f"interp={len(interp_indices)}, "
                          f"attr={len(attr_indices)})")
                    continue

                print(f"  [found] {cache_dir} — {len(paired)} paired samples: {paired}")

                
                method_dir = os.path.dirname(cache_parent)
                save_dir = os.path.join(method_dir, "perturbation_comparisons", gran_name)
                os.makedirs(save_dir, exist_ok=True)

                method_folder = os.path.basename(method_dir)
                m_label = _method_label(method_folder)

                VIS_IDX = {31, 140, 141, 153}
                for idx in paired:
                    x_mi_path   = os.path.join(cache_dir, f"x_mi_idx{idx}.npy")
                    attr_path   = os.path.join(cache_dir, f"attr_idx{idx}.npy")
                    zero_path   = os.path.join(cache_dir, f"zero_out_50_idx{idx}.npy")
                    interp_path = os.path.join(cache_dir, f"interp_50_idx{idx}.npy")

                    if not os.path.exists(x_mi_path):
                        print(f"    [skip idx={idx}] x_mi not found — skipping")
                        continue

                    try:
                        x_mi       = np.load(x_mi_path)
                        attr       = np.load(attr_path)
                        x_zero_out = np.load(zero_path)
                        x_interp   = np.load(interp_path)

                        # Check that zero-out only affects masked positions.
                        print(f"\n  [zero_out integrity idx={idx}]")
                        print(f"    x_mi       shape={x_mi.shape}  "
                              f"min={x_mi.min():.4f}  max={x_mi.max():.4f}")
                        print(f"    x_zero_out shape={x_zero_out.shape}  "
                              f"min={x_zero_out.min():.4f}  max={x_zero_out.max():.4f}")

                        diff            = np.abs(x_mi - x_zero_out)
                        zeroed_mask     = (x_zero_out == 0.0)
                        non_zeroed_mask = ~zeroed_mask

                        pct_zeroed = 100.0 * zeroed_mask.mean()
                        print(f"    zeroed positions : {pct_zeroed:.1f}%  (expected ~50%)")

                        if non_zeroed_mask.any():
                            max_diff  = float(diff[non_zeroed_mask].max())
                            mean_diff = float(diff[non_zeroed_mask].mean())
                            print(f"    max|original - zero_out| OUTSIDE zeroed mask "
                                  f"= {max_diff:.2e}   mean = {mean_diff:.2e}")
                            if max_diff < 1e-6:
                                print(f"    -> PASS: unperturbed timesteps are "
                                      f"bit-identical to original")
                            else:
                                print(f"    -> FAIL: signal changed outside zeroed "
                                      f"segments — wrong array or re-normalisation!")
                        else:
                            print(f"    -> WARNING: entire signal is zeroed (100%)")

                        if idx in VIS_IDX:
                            y_lim = plot_perturbation_heatmap_comparison(
                                attr=attr,
                                x_original=x_mi,
                                x_zero_out=x_zero_out,
                                x_interp=x_interp,
                                sample_idx=idx,
                                method_label=m_label,
                                save_dir=save_dir,
                                leads=leads,
                            )
                            plot_attribution_heatmap(
                                attr=attr,
                                x_mi=x_mi,
                                sample_idx=idx,
                                method_label=m_label,
                                save_dir=save_dir,
                                leads=leads,
                                y_lim=y_lim,
                            )
                    except Exception as exc:
                        print(f"    [error idx={idx}] {exc}")
