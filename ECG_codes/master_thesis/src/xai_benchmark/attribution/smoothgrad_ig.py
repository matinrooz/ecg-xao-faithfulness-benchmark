#SmoothGrad over Integrated Gradients

from typing import Optional

import numpy as np

from .ig import integrated_gradients


def smoothgrad_integrated_gradients(
    model,
    x: np.ndarray,
    target_class: int,
    device,
    baseline: Optional[np.ndarray] = None,
    steps: int = 50,
    n_samples: int = 20,
    noise_sigma: float = 0.05,
) -> np.ndarray:
    #Average IG over n_samples Gaussian-noised copies of x  Returns a (C, T) map
    attrs = []
    for _ in range(n_samples):
        noise = np.random.randn(*x.shape).astype(np.float32) * noise_sigma
        attr  = integrated_gradients(
            model=model, x=x + noise, target_class=target_class,
            device=device, baseline=baseline, steps=steps,
        )
        attrs.append(attr)
    return np.mean(np.stack(attrs, axis=0), axis=0).astype(np.float32)