#compute_attribution and  routes to IG, SmoothGrad-IG, or DTD

from typing import Optional

import numpy as np

from .ig import integrated_gradients
from .smoothgrad_ig import smoothgrad_integrated_gradients
from .dtd import deep_taylor_decomposition


def compute_attribution(
    model,
    x: np.ndarray,
    target_class: int,
    device,
    baseline: Optional[np.ndarray],
    cfg,
) -> np.ndarray:
    """
    Dispatch to the attribution method named in cfg.attribution_method
    ("ig" and "smoothgrad_ig" and  "dtd").
    """
    if cfg.attribution_method == "ig":
        return integrated_gradients(
            model=model, x=x, target_class=target_class,
            device=device, baseline=baseline, steps=cfg.ig_steps,
        )
    elif cfg.attribution_method == "smoothgrad_ig":
        return smoothgrad_integrated_gradients(
            model=model, x=x, target_class=target_class,
            device=device, baseline=baseline, steps=cfg.ig_steps,
            n_samples=cfg.sg_samples, noise_sigma=cfg.sg_sigma,
        )
    elif cfg.attribution_method == "dtd":
        return deep_taylor_decomposition(
            model=model, x=x, target_class=target_class, device=device,
        )
    else:
        raise ValueError(f"Unknown attribution_method: {cfg.attribution_method!r}")
