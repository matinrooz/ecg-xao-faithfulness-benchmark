
#2-second cropped attribution visualization for the existing MI sample (idx=31).


import os
import json

import numpy as np
import matplotlib.pyplot as plt

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(os.path.dirname(HERE))                
RESULTS = os.path.join(ROOT, "results")
OUTDIR  = os.path.join(RESULTS, "figures_2s_crop_idx31")

IDX    = 31
FS     = 100                                                   
T      = 1000
WIN    = 200                                                    
LEADS  = [0, 1, 5]                                              
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

METHODS = [   # display label, results subpath template (relative to results/)
    ("IG",            "IG/{arch}/ig/_viz_cache/segment"),
    ("SmoothGrad-IG", "smoothgrad/{arch}/smoothgrad_ig/_viz_cache/segment"),
    ("DTD",           "dtd/{arch}/dtd/_viz_cache/segment"),
]
ARCHS = [("cnn", "CNN"), ("cnn_lstm", "CNN–BiLSTM with Attention")]


def load_attr(arch, subpath):
    d = os.path.join(RESULTS, subpath.format(arch=arch))
    attr = np.load(os.path.join(d, f"attr_idx{IDX}.npy"))
    x_mi = np.load(os.path.join(d, f"x_mi_idx{IDX}.npy"))
    assert attr.shape == (12, T) and x_mi.shape == (12, T), (attr.shape, x_mi.shape)
    return attr, x_mi


def per_lead_norm(attr):
    """|attr| normalized per lead by its full-10s max (matches the existing figure)."""
    out = np.abs(attr).astype(np.float64)
    for L in range(12):
        out[L] /= (out[L].max() + 1e-8)
    return out


def choose_window():
    """Pooled attribution peak across all methods x both archs -> fixed 200-step window."""
    curves, x_ref = [], None
    for _, sub in METHODS:
        for arch, _ in ARCHS:
            attr, x_mi = load_attr(arch, sub)
            if x_ref is None:
                x_ref = x_mi
            else:
                assert np.array_equal(x_ref, x_mi), "x_mi differs across method/arch"
            n = per_lead_norm(attr)
            curves.append(n[LEADS].mean(axis=0))               # avg over shown leads
    pooled = np.mean(curves, axis=0)
    centre = int(np.argmax(pooled))
    start, end = centre - WIN // 2, centre + WIN // 2
    if start < 0:
        start, end = 0, WIN
    if end > T:
        start, end = T - WIN, T
    return centre, start, end, x_ref


def shared_ylim(x_mi):
    sig = np.stack([x_mi[i] for i in LEADS])
    lo, hi = float(sig.min()), float(sig.max())
    r = hi - lo + 1e-8
    return lo - 0.15 * r, hi + 0.15 * r


def make_figure(arch_key, arch_title, start, end, y_lim):
    t_s = np.arange(start, end) / FS                            # seconds axis for the crop
    nrows, ncols = len(LEADS), len(METHODS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9), squeeze=False, sharex=True)
    im = None
    for c, (mlabel, sub) in enumerate(METHODS):
        attr, x_mi = load_attr(arch_key, sub)
        norm = per_lead_norm(attr)                              # normalized over FULL 10 s
        for r, L in enumerate(LEADS):
            ax = axes[r, c]
            strip = norm[L][start:end][np.newaxis, :]           # crop AFTER normalization
            im = ax.imshow(strip, aspect="auto",
                           extent=[t_s[0], t_s[-1], y_lim[0], y_lim[1]],
                           origin="lower", cmap="YlOrRd", vmin=0.0, vmax=1.0,
                           interpolation="nearest", alpha=0.85, zorder=1)
            ax.plot(t_s, x_mi[L][start:end], color="black", lw=0.9, zorder=2)
            ax.set_xlim(t_s[0], t_s[-1]); ax.set_ylim(*y_lim)
            ax.tick_params(labelsize=8)
            ax.tick_params(axis="x", labelbottom=True)   # show x tick numbers on every subplot
            if r == 0:
                ax.set_title(mlabel, fontsize=12, pad=6)
            if c == 0:
                ax.set_ylabel(f"Lead {LEAD_NAMES[L]}\nAmplitude", fontsize=9)
            if r == nrows - 1:
                ax.set_xlabel("Time (s, relative to 10-s recording)", fontsize=9)

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Attribution magnitude\n(per-lead normalized, full-10s scale)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"{arch_title} — MI sample idx={IDX}: 2-s crop "
                 f"[{start}–{end} steps = {start/FS:.2f}–{end/FS:.2f} s]",
                 fontsize=14)

    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, f"{arch_key}_2s_crop_idx{IDX}")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved:", base + ".png")
    print("saved:", base + ".pdf")


def main():
    centre, start, end, x_ref = choose_window()
    assert end - start == WIN, (start, end)
    y_lim = shared_ylim(x_ref)
    meta = {
        "sample_index": IDX, "fs_hz": FS, "n_timesteps_full": T,
        "crop_start_step": start, "crop_end_step": end, "crop_len_steps": end - start,
        "crop_start_s": start / FS, "crop_end_s": end / FS,
        "centre_step": centre, "centre_s": centre / FS,
        "leads_shown": [LEAD_NAMES[i] for i in LEADS],
        "rule": ("centre = argmax of pooled attribution magnitude across IG, SmoothGrad-IG, "
                 "DTD and both architectures (|attr| per-lead normalized by full-10s max, "
                 "averaged over leads I/II/aVF then over the 6 maps); window fixed for all "
                 "panels; coincides with a QRS R-peak at this timestep."),
        "normalization": "per-lead |attr| / full-10s per-lead max (original scale preserved)",
        "y_lim": list(y_lim),
    }
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "crop_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("crop:", json.dumps({k: meta[k] for k in
          ("sample_index", "crop_start_step", "crop_end_step", "crop_start_s", "crop_end_s", "centre_step")}))

    for arch_key, arch_title in ARCHS:
        make_figure(arch_key, arch_title, start, end, y_lim)


if __name__ == "__main__":
    main()
