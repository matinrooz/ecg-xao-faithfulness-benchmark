"""Importance ranking and stepwise perturbation."""

from typing import List, Optional, Tuple

import numpy as np


def segment_importance(ig: np.ndarray, n_segments: int = 50):
    #Rank segments by total absolute attribution 
    T       = ig.shape[1]
    seg_len = T // n_segments
    bounds  = []
    scores  = np.empty(n_segments, dtype=np.float64)
    for i in range(n_segments):
        s = i * seg_len
        e = (i + 1) * seg_len if i < n_segments - 1 else T
        bounds.append((s, e))
        scores[i] = float(np.sum(np.abs(ig[:, s:e])))
    return np.argsort(-scores), bounds, scores


def timestep_importance(ig: np.ndarray, mode: str = "mean_abs"):
    #Aggregate attribution over the 12 leads and rank individual time steps
    if mode == "mean_abs":
        scores = np.mean(np.abs(ig), axis=0)
    elif mode == "sum_abs":
        scores = np.sum(np.abs(ig), axis=0)
    else:
        raise ValueError(f"Unknown timestep_score: {mode!r}")
    return np.argsort(-scores), scores


def replace_segments_stepwise(
    x_mi: np.ndarray,
    x_ref: np.ndarray,
    order: np.ndarray,
    bounds,
    perturb_mode: str = "zero_out",
    interp_alpha: float = 0.5,
    true_zero: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    
    #Perturb ranked segments sequentially and return each intermediate signal.

    x_cur  = x_mi.copy()
    steps  = [x_cur.copy()]
    for seg_idx in order:
        s, e = bounds[int(seg_idx)]
        if perturb_mode == "zero_out":
            if true_zero is not None:
                x_cur[:, s:e] = true_zero[:, s:e]
            else:
                x_cur[:, s:e] = 0.0
        elif perturb_mode == "interpolate":
            x_cur[:, s:e] = (1.0 - interp_alpha) * x_mi[:, s:e] + interp_alpha * x_ref[:, s:e]
        elif perturb_mode == "synthetic":
            x_cur[:, s:e] = x_ref[:, s:e]
        else:
            raise ValueError(f"Unknown perturb_mode: {perturb_mode!r}")
        steps.append(x_cur.copy())
    return steps


def replace_timesteps_stepwise(
    x_mi: np.ndarray,
    x_ref: np.ndarray,
    order: np.ndarray,
    chunk_size: int = 10,
    perturb_mode: str = "zero_out",
    interp_alpha: float = 0.5,
    true_zero: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Like replace_segments_stepwise, but perturbs chunk_size timesteps at a time in
    ranked order.
    """
    x_cur = x_mi.copy()
    steps = [x_cur.copy()]
    for k in range(0, len(order), chunk_size):
        idx = order[k:k + chunk_size]
        if perturb_mode == "zero_out":
            if true_zero is not None:
                x_cur[:, idx] = true_zero[:, idx]
            else:
                x_cur[:, idx] = 0.0
        elif perturb_mode == "interpolate":
            x_cur[:, idx] = (1.0 - interp_alpha) * x_mi[:, idx] + interp_alpha * x_ref[:, idx]
        elif perturb_mode == "synthetic":
            x_cur[:, idx] = x_ref[:, idx]
        else:
            raise ValueError(f"Unknown perturb_mode: {perturb_mode!r}")
        steps.append(x_cur.copy())
    return steps


def random_order(n_items: int, seed: int) -> np.ndarray:
    #Random permutation of n_items indices
    return np.random.default_rng(seed).permutation(n_items)
