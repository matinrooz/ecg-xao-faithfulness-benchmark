"""
Compute the population-level reference curves for the benchmark
The reference population always uses zero-out perturbation
"""

from typing import List, Tuple

import numpy as np

from .metrics import predict_prob, apply_edge_mask
from .perturbation import (
    segment_importance,
    timestep_importance,
    replace_segments_stepwise,
    replace_timesteps_stepwise,
)
from .attribution import compute_attribution


def compute_norm_ig_orders(model, X_norm: np.ndarray, device, cfg) -> List[Tuple]:
    #Attribution orders for NORM samples (target_class=0)
    return compute_population_ig_orders(model, X_norm, target_class=0, device=device, cfg=cfg)


def compute_norm_specificity_curve(
    model,
    X_norm: np.ndarray,
    norm_orders: List[Tuple],
    device,
    cfg,
) -> np.ndarray:
    #Mean P(NORM | perturbed) at each step across all NORM patients
    return compute_population_curve(
        model, X_norm, norm_orders, target_class=0, device=device, cfg=cfg)


def compute_population_ig_orders(
    model,
    X: np.ndarray,
    target_class: int,
    device,
    cfg,
) -> List[Tuple]:
    #Attribution orders (order, bounds) per sample, using a zero baseline
    label = "NORM" if target_class == 0 else "MI"
    print(f"    [{label} IG orders] Computing attribution for {len(X)} patients ...",
          flush=True)
    orders = []
    for i, x in enumerate(X):
        baseline = np.zeros_like(x)
        ig = compute_attribution(
            model=model, x=x, target_class=target_class,
            device=device, baseline=baseline, cfg=cfg,
        )
        ig_scored = apply_edge_mask(ig, mask_size=cfg.edge_mask)

        if cfg.eval_granularity == "segment":
            order, bounds, _ = segment_importance(ig_scored, n_segments=cfg.n_segments)
        elif cfg.eval_granularity == "timestep":
            order, _ = timestep_importance(ig_scored, mode=cfg.timestep_score)
            bounds = None
        else:
            raise ValueError(f"Unknown eval_granularity: {cfg.eval_granularity!r}")

        orders.append((order, bounds))

        if (i + 1) % 50 == 0 or (i + 1) == len(X):
            print(f"    [{label} IG orders] {i+1}/{len(X)} done", flush=True)

    return orders


def compute_population_curve(
    model,
    X: np.ndarray,
    orders: List[Tuple],
    target_class: int,
    device,
    cfg,
) -> np.ndarray:
    
    
    #Compute the population-level prediction curve using zero-out perturbation

    
    label = "NORM specificity" if target_class == 0 else "MI sensitivity"
    n = len(X)
    print(f"    [{label} curve] Evaluating across {n} patients ...", flush=True)

    if cfg.eval_granularity == "segment":
        n_steps = cfg.n_segments
    elif cfg.eval_granularity == "timestep":
        T = X[0].shape[1]
        n_steps = (T + cfg.chunk_size - 1) // cfg.chunk_size
    else:
        raise ValueError(f"Unknown eval_granularity: {cfg.eval_granularity!r}")

    acc = np.zeros(n_steps + 1, dtype=np.float64)
    x_ref_zeros = None

    for i, (x, (order, bounds)) in enumerate(zip(X, orders)):
        if x_ref_zeros is None or x_ref_zeros.shape != x.shape:
            x_ref_zeros = np.zeros_like(x)

        if cfg.eval_granularity == "segment":
            steps = replace_segments_stepwise(
                x, x_ref_zeros, order, bounds, perturb_mode="zero_out")
        else:
            steps = replace_timesteps_stepwise(
                x, x_ref_zeros, order,
                chunk_size=cfg.chunk_size, perturb_mode="zero_out")

        probs = np.array(
            [predict_prob(model, s, device, target_class=target_class) for s in steps],
            dtype=np.float64,
        )
        min_len = min(len(probs), n_steps + 1)
        acc[:min_len] += probs[:min_len]

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"    [{label} curve] {i+1}/{n}  "
                  f"mean step0={acc[0]/(i+1):.4f}  "
                  f"last={acc[min_len-1]/(i+1):.4f}", flush=True)

    return (acc / n).astype(np.float32)


def compute_mi_ig_orders(model, X_mi: np.ndarray, device, cfg) -> List[Tuple]:
    #Attribution orders for MI samples (target_class=1)
    return compute_population_ig_orders(model, X_mi, target_class=1, device=device, cfg=cfg)


def compute_mi_sensitivity_curve(
    model,
    X_mi: np.ndarray,
    mi_orders: List[Tuple],
    device,
    cfg,
) -> np.ndarray:
    #Mean P(MI | perturbed) at each step across all MI samples 
    return compute_population_curve(
        model, X_mi, mi_orders, target_class=1, device=device, cfg=cfg)
