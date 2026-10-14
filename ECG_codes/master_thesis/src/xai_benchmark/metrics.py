"""calculation of AUC, bootstrap CI, Wilcoxon test, completeness check, edge mask and  inference helpers"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import wilcoxon

# np.trapezoid was added in newer numpy; fall back to np.trapz
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore


@torch.no_grad()
def predict_prob(model, x: np.ndarray, device, target_class: int = 1) -> float:
    #Return P(target_class) for a single sample
    model.eval()
    x_t = torch.tensor(x[None], dtype=torch.float32, device=device)
    return float(torch.softmax(model(x_t), dim=1)[0, target_class].item())


def compute_auc(probs: np.ndarray) -> float:
    #Trapezoidal AUC of a probability curve over a [0, 1] fraction axis"""
    x = np.linspace(0.0, 1.0, len(probs))
    return float(_trapezoid(probs, x))


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    #Percentile bootstrap 95% CI for the mean. Returns (mean, lo, hi)
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng   = np.random.default_rng(seed)
    n     = len(values)
    boots = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return (
        float(values.mean()),
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


def wilcoxon_paired_test(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "less",
) -> Dict:
    #Paired Wilcoxon signed-rank test
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    try:
        stat, p = wilcoxon(a, b, alternative=alternative)
    except ValueError as e:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n": len(a), "error": str(e)}
    return {"statistic": float(stat), "p_value": float(p),
            "n": len(a), "alternative": alternative}


def completeness_check(
    model,
    x: np.ndarray,
    ig: np.ndarray,
    target_class: int,
    device,
    baseline: Optional[np.ndarray] = None,
) -> Dict:
    #Check that sum(IG) ≈ f(x) - f(baseline); returns the terms and relative error
    if baseline is None:
        baseline = np.zeros_like(x)
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x[None],        dtype=torch.float32, device=device)
        b_t = torch.tensor(baseline[None], dtype=torch.float32, device=device)
        f_x = float(model(x_t)[0, target_class].item())
        f_b = float(model(b_t)[0, target_class].item())
    ig_sum  = float(ig.sum())
    diff    = f_x - f_b
    rel_err = abs(ig_sum - diff) / (abs(diff) + 1e-8)
    return {"ig_sum": ig_sum, "f_x": f_x, "f_baseline": f_b,
            "diff": diff, "rel_error": rel_err}


def apply_edge_mask(ig: np.ndarray, mask_size: int = 30) -> np.ndarray:
    #Zero the first/last mask_size timesteps to suppress CNN boundary artifacts
    if mask_size <= 0:
        return ig
    out = ig.copy()
    out[:, :mask_size]  = 0.0
    out[:, -mask_size:] = 0.0
    return out


def compute_ba_curve_population(
    model,
    X_steps: List[np.ndarray],
    cross_class_curve: np.ndarray,
    device,
    sample_class: int = 1,
) -> np.ndarray:
    """Compute the population-level balanced-accuracy curve for one sample."""
    model.eval()
    ba_values = []
    for k, x in enumerate(X_steps):
        p_this   = predict_prob(model, x, device, sample_class)
        p_cross  = float(cross_class_curve[min(k, len(cross_class_curve) - 1)])
        ba_values.append((p_this + p_cross) / 2.0)
    return np.array(ba_values, dtype=np.float32)
