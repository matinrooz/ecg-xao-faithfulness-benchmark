#Integrated Gradients

from typing import Optional

import numpy as np
import torch


def integrated_gradients(
    model,
    x: np.ndarray,
    target_class: int,
    device,
    baseline: Optional[np.ndarray] = None,
    steps: int = 50,
) -> np.ndarray:
    #Integrated Gradients with the midpoint rule. Returns a (C, T) attribution map
    model.eval()
    x_flat = torch.tensor(x,        dtype=torch.float32, device=device)
    b_flat  = torch.zeros_like(x_flat) if baseline is None else \
              torch.tensor(baseline, dtype=torch.float32, device=device)

    alphas = torch.linspace(0.5 / steps, 1.0 - 0.5 / steps, steps, device=device
                            ).view(-1, 1, 1)
    diff   = (x_flat - b_flat).unsqueeze(0)
    base   = b_flat.unsqueeze(0)
    scaled = (base + alphas * diff).contiguous().requires_grad_(True)

    # cuDNN needs the LSTM in train mode to get a backward pass through it
    if hasattr(model, 'lstm'):
        model.lstm.train()
    out    = model(scaled)
    target = out[:, target_class].sum()
    grads  = torch.autograd.grad(target, scaled, create_graph=False)[0]
    if hasattr(model, 'lstm'):
        model.lstm.eval()

    ig = (x_flat - b_flat) * grads.mean(dim=0)
    return ig.detach().cpu().numpy().astype(np.float32)
