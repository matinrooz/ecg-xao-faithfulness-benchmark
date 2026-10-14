"""Visualization and report functions for the perturbation benchmark."""

import os
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .metrics import bootstrap_ci, wilcoxon_paired_test


def _plot_ecg_colored(ax, t, signal, mask, perturb_color, lw=0.8, zorder=3):
    

    #Plot the ECG and highlight the perturbed regions in a different color.
    
    T = len(signal)
    ax.plot(t, signal, color="black", linewidth=lw, zorder=zorder)

    if not mask.any():
        return

    padded = np.concatenate([[False], mask, [False]])
    starts = np.where(~padded[:-1] &  padded[1:])[0]
    ends   = np.where( padded[:-1] & ~padded[1:])[0]

    for s, e in zip(starts, ends):
        s_ext = max(0, s - 1)
        e_ext = min(T - 1, e)
        ax.plot(t[s_ext:e_ext + 1], signal[s_ext:e_ext + 1],
                color=perturb_color, linewidth=lw, zorder=zorder + 1)


def _method_label(attribution_method: str) -> str:
    # Display label for an attribution method key
    return {"ig": "IG", "smoothgrad_ig": "SmoothGrad-IG", "dtd": "DTD"}.get(
        attribution_method, attribution_method)


def _experiment_title(method_label: str, gran: str, perturb_mode: str) -> str:
    #Figure title for IG-guided Segment Perturbation (Zero-out)
    gran_str    = "Segment" if gran == "segment" else "Timestep"
    perturb_str = "Zero-out" if perturb_mode == "zero_out" else "Interpolation"
    prefix = f"{method_label}-guided" if method_label == "IG" else method_label
    return f"{prefix} {gran_str} Perturbation ({perturb_str})"


def plot_mean_curves(curves_main, curves_rand, save_path,
                     method_label="IG", gran="segment", perturb_mode="zero_out",
                     title=None, perturb_label=None):
    
    # Plot and save the mean perturbation curves.

    # Also saves the mean curves for later analysis.

    
    clean_title = _experiment_title(method_label, gran, perturb_mode)
    x_label     = f"{method_label}-guided" if method_label == "IG" else method_label

    cm_arr = np.asarray(curves_main)
    cr_arr = np.asarray(curves_rand)
    mean_main = cm_arr.mean(axis=0)
    mean_rand = cr_arr.mean(axis=0)
    x  = np.linspace(0, 1, cm_arr.shape[1])

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_main, label=f"{x_label}", color="C0", linewidth=2)
    plt.plot(x, mean_rand, label="Random order", color="C1",
             linewidth=2, linestyle="--")
    plt.xlabel("Fraction of ranked units perturbed", fontsize=11)
    plt.ylabel("Balanced Accuracy (population-level)", fontsize=11)
    plt.title(clean_title, fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    out_dir = os.path.dirname(save_path)
    np.save(os.path.join(out_dir, "mean_curve_main.npy"),   mean_main)
    np.save(os.path.join(out_dir, "mean_curve_random.npy"), mean_rand)


def plot_auc_boxplot(auc_main, auc_rand, save_path,
                     method_label="IG", gran="segment", perturb_mode="zero_out",
                     title=None):
    """Boxplot of method-guided vs random AUCs. title is a legacy no-op."""
    clean_title = _experiment_title(method_label, gran, perturb_mode)
    x_label     = f"{method_label}-guided" if method_label == "IG" else method_label

    plt.figure(figsize=(6, 5))
    plt.boxplot([auc_main, auc_rand],
                tick_labels=[x_label, "Random"],
                medianprops=dict(color="C0", linewidth=2),
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker="o", markersize=4, linestyle="none"))
    plt.ylabel("AUC of Mean Balanced Accuracy Across Samples", fontsize=11)
    plt.title(clean_title, fontsize=13, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def write_final_report(out_path: str, cfg, auc_main: np.ndarray,
                       auc_rand: np.ndarray, completeness_errs: List[float]):
    #Write the per-experiment text report config, win rate, AUC CIs and Wilcoxon)
    n            = len(auc_main)
    n_better     = int((auc_main < auc_rand).sum())
    comp_err     = np.asarray(completeness_errs, dtype=np.float64)
    method_label = _method_label(cfg.attribution_method)

    mean_m, lo_m, hi_m = bootstrap_ci(auc_main)
    mean_r, lo_r, hi_r = bootstrap_ci(auc_rand)
    delta               = auc_rand - auc_main
    mean_d, lo_d, hi_d = bootstrap_ci(delta)
    wt                  = wilcoxon_paired_test(auc_main, auc_rand)

    lines = [
        "=" * 60,
        "final report",
        
        "=" * 60,
        f"model_type            : {cfg.model_type}",
        f"attribution_method    : {cfg.attribution_method}",
        f"sg_samples            : {cfg.sg_samples}",
        f"sg_sigma              : {cfg.sg_sigma}",
        f"baseline              : {cfg.baseline}",
        f"perturb_mode          : {cfg.perturb_mode}",
        f"interp_alpha          : {cfg.interp_alpha}",
        f"synth_method          : {cfg.synth_method}",
        f"per_lead_phase        : {cfg.per_lead_phase}",
        f"edge_mask             : {cfg.edge_mask}",
        f"eval_granularity      : {cfg.eval_granularity}",
        f"n_segments            : {cfg.n_segments}",
        f"chunk_size            : {cfg.chunk_size}",
        f"timestep_score        : {cfg.timestep_score}",
        f"ig_steps              : {cfg.ig_steps}",
        f"n_samples             : {n}",
        "",
        f"Win rate ({method_label} < rand): {n_better}/{n}  ({100*n_better/n:.1f}%)",
        "",
        "Completeness check (IG only — N/A for SmoothGrad-IG):",
        f"  median rel_error    : {np.nanmedian(comp_err):.4%}",
        f"  max    rel_error    : {np.nanmax(comp_err) if not np.all(np.isnan(comp_err)) else float('nan'):.4%}",
        f"  (>5% -> --ig_steps too small; nan = SmoothGrad-IG run)",
        "",
        "AUC (lower = better):",
        f"  {method_label}-guided mean={mean_m:.4f}  std={auc_main.std():.4f}  "
        f"95% CI [{lo_m:.4f}, {hi_m:.4f}]",
        f"  Random     mean={mean_r:.4f}  std={auc_rand.std():.4f}  "
        f"95% CI [{lo_r:.4f}, {hi_r:.4f}]",
        f"  Delta(R-M) mean={mean_d:.4f}  "
        f"95% CI [{lo_d:.4f}, {hi_d:.4f}]",
        "",
        f"Wilcoxon signed-rank ({method_label} AUC < random AUC):",
        f"  statistic  : {wt['statistic']:.4f}",
        f"  p_value    : {wt['p_value']:.6e}",
        f"  n_pairs    : {wt['n']}",
    ]

    text = "\n".join(lines) + "\n"
    print(text)
    with open(out_path, "w") as f:
        f.write(text)


def plot_attribution_heatmap(
    attr: np.ndarray,
    x_mi: np.ndarray,
    sample_idx: int,
    method_label: str,
    save_dir: str,
    leads: List[int] = [0, 1, 5],
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    """
  Plot the ECG with an attribution heatmap for the selected leads.
  Uses a shared y-axis by default unless a custom range is provided.
   """
    os.makedirs(save_dir, exist_ok=True)
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    n_leads = len(leads)
    T = x_mi.shape[1]

    if y_lim is not None:
        y_min, y_max = y_lim
    else:
        all_signals = np.stack([x_mi[i] for i in leads], axis=0)
        global_min  = float(all_signals.min())
        global_max  = float(all_signals.max())
        sig_range   = global_max - global_min + 1e-8
        y_min = global_min - 0.15 * sig_range
        y_max = global_max + 0.15 * sig_range

    fig, axes = plt.subplots(n_leads, 1, figsize=(12, 3 * n_leads), squeeze=False)

    for row, lead_idx in enumerate(leads):
        ax = axes[row, 0]
        lead_name = lead_names[lead_idx] if lead_idx < len(lead_names) else f"L{lead_idx}"

        ecg_signal = x_mi[lead_idx].copy()

        attr_magnitude = np.abs(attr[lead_idx])
        attr_norm      = attr_magnitude / (attr_magnitude.max() + 1e-8)

        im = ax.imshow(
            attr_norm[np.newaxis, :],
            aspect="auto",
            extent=[0, T, y_min, y_max],
            origin="lower",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            alpha=0.85,
            zorder=1,
        )

        ax.plot(np.arange(T), ecg_signal, color="black", linewidth=0.8, zorder=2)

        ax.set_xlim(0, T)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(" Normalized Amplitude", fontsize=9)
        ax.set_title(f"Lead {lead_name}", fontsize=10, pad=3)
        ax.tick_params(labelsize=8)
        if row < n_leads - 1:
            ax.tick_params(labelbottom=False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label("Attribution\nmagnitude", fontsize=7)
        cbar.ax.tick_params(labelsize=7)

    axes[-1, 0].set_xlabel("Time Step", fontsize=9)
    fig.suptitle(
        f"{method_label} Attribution — Dataset Sample idx={sample_idx}",
        fontsize=12,
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"attribution_heatmap_idx{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [plot] Saved: {save_path}")


def plot_perturbation_heatmap_comparison(
    attr: np.ndarray,
    x_original: np.ndarray,
    x_zero_out: np.ndarray,
    x_interp: np.ndarray,
    sample_idx: int,
    method_label: str,
    save_dir: str,
    leads: List[int] = [0, 1, 5],
) -> Tuple[float, float]:
    """
    Compare the original and perturbed ECG signals across the selected leads.
    Returns the shared y-axis range used for all panels.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    n_leads  = len(leads)
    T        = x_original.shape[1]
    t        = np.arange(T)
    col_ecgs = [x_original, x_zero_out, x_interp]
    col_titles = [
        "Original ECG",
        "Zero-out (50% perturbed)",
        "Interpolation (50% perturbed)",
    ]
    highlight_perturbed = [False, True, True]

    attr_norms = {}
    for lead_idx in leads:
        a = np.abs(attr[lead_idx])
        attr_norms[lead_idx] = a / (a.max() + 1e-8)

    fig = plt.figure(figsize=(17, 3.2 * n_leads))
    gs  = fig.add_gridspec(
        n_leads, 4,
        width_ratios=[1, 1, 1, 0.04],
        hspace=0.35,
        wspace=0.25,
    )
    axes   = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(n_leads)]
    cbar_ax = fig.add_subplot(gs[:, 3])

    im_ref = None

    # shared y-limits across all leads and all three columns
    all_vals = np.concatenate([
        np.concatenate([x_original[i], x_zero_out[i], x_interp[i]])
        for i in leads
    ])
    _gr = all_vals.max() - all_vals.min() + 1e-8
    y_min = float(all_vals.min()) - 0.15 * _gr
    y_max = float(all_vals.max()) + 0.15 * _gr

    for row, lead_idx in enumerate(leads):
        lead_name = lead_names[lead_idx] if lead_idx < len(lead_names) else f"L{lead_idx}"
        attr_norm = attr_norms[lead_idx]

        # zero-out mask = positions set to 0.0; interp mask = positions that changed
        zero_mask   = (x_zero_out[lead_idx] == 0.0)
        interp_mask = (np.abs(x_interp[lead_idx] - x_original[lead_idx]) > 1e-6)
        perturb_masks = [None, zero_mask, interp_mask]

        for col, (ecg, col_title, do_highlight, pmask) in enumerate(
                zip(col_ecgs, col_titles, highlight_perturbed, perturb_masks)):
            ax         = axes[row][col]
            ecg_signal = ecg[lead_idx]

            im = ax.imshow(
                attr_norm[np.newaxis, :],
                aspect="auto",
                extent=[0, T, y_min, y_max],
                origin="lower",
                cmap="YlOrRd",
                vmin=0.0, vmax=1.0,
                interpolation="nearest",
                alpha=0.80,
                zorder=1,
            )
            im_ref = im

            perturb_colors = [None, "#e03030", "#e07800"]
            p_color = perturb_colors[col] if (do_highlight and pmask is not None) else None
            if p_color is not None:
                _plot_ecg_colored(ax, t, ecg_signal, pmask,
                                  perturb_color=p_color, lw=0.8, zorder=3)
            else:
                ax.plot(t, ecg_signal, color="black", linewidth=0.8, zorder=3)

            ax.set_xlim(0, T)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(col_title, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Lead {lead_name}\nNormalized Amplitude", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row < n_leads - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Time Step", fontsize=8)

    cbar = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.set_label(f"{method_label}\nattribution\nmagnitude", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{method_label} — Perturbation comparison — Dataset Sample idx={sample_idx}",
        fontsize=12, y=1.01,
    )

    save_path = os.path.join(
        save_dir, f"perturbation_heatmap_comparison_idx{sample_idx}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [plot] Saved: {save_path}")
    return (y_min, y_max)


def plot_perturbation_heatmap_single(
    attr: np.ndarray,
    x_original: np.ndarray,
    x_perturbed: np.ndarray,
    sample_idx: int,
    method_label: str,
    perturb_mode: str,
    save_dir: str,
    leads: List[int] = [0, 1, 5],
) -> None:
    """
    Plot the original and perturbed ECG side by side.
    """
    os.makedirs(save_dir, exist_ok=True)
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    n_leads = len(leads)
    T       = x_original.shape[1]
    t       = np.arange(T)

    is_zero_out  = (perturb_mode == "zero_out")
    perturb_color = "#e03030" if is_zero_out else "#e07800"
    perturb_title = "Zero-out (50% perturbed)" if is_zero_out \
                    else "Interpolation (50% perturbed)"

    all_vals = np.concatenate([x_original[i] for i in leads] +
                              [x_perturbed[i] for i in leads])
    gr = all_vals.max() - all_vals.min() + 1e-8
    y_min = float(all_vals.min()) - 0.15 * gr
    y_max = float(all_vals.max()) + 0.15 * gr

    attr_norms = {}
    for lead_idx in leads:
        a = np.abs(attr[lead_idx])
        attr_norms[lead_idx] = a / (a.max() + 1e-8)

    if is_zero_out:
        def _pmask(li): return x_perturbed[li] == 0.0
    else:
        def _pmask(li): return np.abs(x_perturbed[li] - x_original[li]) > 1e-6

    fig = plt.figure(figsize=(12, 3.2 * n_leads))
    gs  = fig.add_gridspec(n_leads, 3, width_ratios=[1, 1, 0.04],
                           hspace=0.35, wspace=0.25)
    axes    = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(n_leads)]
    cbar_ax =  fig.add_subplot(gs[:, 2])

    im_ref = None
    col_titles = ["Original ECG", perturb_title]

    for row, lead_idx in enumerate(leads):
        lead_name  = lead_names[lead_idx] if lead_idx < len(lead_names) else f"L{lead_idx}"
        attr_norm  = attr_norms[lead_idx]
        pmask      = _pmask(lead_idx)
        col_signals = [x_original[lead_idx], x_perturbed[lead_idx]]

        for col, (sig, col_title) in enumerate(zip(col_signals, col_titles)):
            ax = axes[row][col]

            im = ax.imshow(attr_norm[np.newaxis, :], aspect="auto",
                           extent=[0, T, y_min, y_max], origin="lower",
                           cmap="YlOrRd", vmin=0.0, vmax=1.0,
                           interpolation="nearest", alpha=0.80, zorder=1)
            im_ref = im

            if col == 1:
                _plot_ecg_colored(ax, t, sig, pmask, perturb_color, lw=0.8, zorder=3)
            else:
                ax.plot(t, sig, color="black", linewidth=0.8, zorder=3)

            ax.set_xlim(0, T)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(col_title, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Lead {lead_name}\nNormalized Amplitude", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row < n_leads - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Time Step", fontsize=8)

    cbar = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.set_label(f"{method_label}\nattribution\nmagnitude", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    mode_str = "zero_out" if is_zero_out else "interp"
    fig.suptitle(
        f"{method_label} — {perturb_title} — Dataset Sample idx={sample_idx}",
        fontsize=12, y=1.01,
    )
    save_path = os.path.join(
        save_dir, f"perturbation_heatmap_{mode_str}_idx{sample_idx}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [plot] Saved: {save_path}")


def plot_perturbation_comparison(
    x_original: np.ndarray,
    x_zero_out: np.ndarray,
    x_interp: np.ndarray,
    sample_idx: int,
    save_dir: str,
    leads: List[int] = [0, 1, 5],
) -> None:
    """Simple line plot: leads x (original | zero-out 50% | interpolation 50%)."""
    os.makedirs(save_dir, exist_ok=True)
    n_leads = len(leads)
    T = x_original.shape[1]
    t = np.arange(T)

    col_labels = ["Original", "Zero-out (50% perturbed)", "Interpolation (50% perturbed)"]
    signals    = [x_original, x_zero_out, x_interp]
    colors     = ["steelblue", "crimson", "darkorange"]

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

    fig, axes = plt.subplots(n_leads, 3, figsize=(15, 3 * n_leads), sharex=True)
    if n_leads == 1:
        axes = axes[np.newaxis, :]

    for row, lead_idx in enumerate(leads):
        lead_name = lead_names[lead_idx] if lead_idx < len(lead_names) else f"L{lead_idx}"
        for col, (sig, clr, col_label) in enumerate(zip(signals, colors, col_labels)):
            ax = axes[row, col]
            ax.plot(t, sig[lead_idx], color=clr, linewidth=0.8)
            if row == 0:
                ax.set_title(col_label, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Lead {lead_name}", fontsize=9)
            if row == n_leads - 1:
                ax.set_xlabel("Timestep")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Perturbation comparison — sample idx={sample_idx}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"perturbation_comparison_idx{sample_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [plot] Saved: {save_path}")
