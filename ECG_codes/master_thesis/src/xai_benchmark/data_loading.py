#Data and model loading utilities

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .models import ECGCNN1D_Baseline, ECGCNN1D_Deeper, ECGCNNLSTMAttn


def load_data(data_dir: str, split: str = "test"):
    """Load an ECG split from .npy files, plus per-lead means/stds if present."""
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
    means_path = os.path.join(data_dir, f"X_{split}_means.npy")
    stds_path  = os.path.join(data_dir, f"X_{split}_stds.npy")
    if os.path.exists(means_path) and os.path.exists(stds_path):
        X_means = np.load(means_path).astype(np.float32)
        X_stds  = np.load(stds_path).astype(np.float32)
    else:
        print("  WARNING: X_means/X_stds not found — zero-out will use 0.0 (z-score mean)")
        X_means = None
        X_stds  = None
    return X.astype(np.float32), y.astype(int), X_means, X_stds


def _detect_cnn_arch_from_sd(sd: dict) -> str:
    #Infer the CNN architecture from the checkpoint's key names
    keys = set(sd.keys())
    if any(k.startswith("net.") for k in keys):
        return "baseline"
    if any(k.startswith("features.") for k in keys):
        return "deeper"
    return "unknown"


def build_model(
    model_type: str,
    num_classes: int,
    cnn_arch: str = "baseline",
    dropout: float = 0.3,
    lstm_hidden: int = 64,
    attn_hidden: int = 64,
) -> nn.Module:
    #Instantiate a model by type
    if model_type == "cnn":
        if cnn_arch == "baseline":
            return ECGCNN1D_Baseline(in_ch=12, num_classes=num_classes)
        else:
            return ECGCNN1D_Deeper(in_ch=12, num_classes=num_classes, dropout=dropout)
    elif model_type == "cnn_lstm_attn":
        return ECGCNNLSTMAttn(
            in_ch=12, num_classes=num_classes,
            dropout=dropout, lstm_hidden=lstm_hidden, attn_hidden=attn_hidden,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


def load_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    device,
    expected_arch: str = "",
) -> nn.Module:
    """Load weights into model, tolerating the common checkpoint save formats."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items()}

    if expected_arch in ("baseline", "deeper"):
        detected = _detect_cnn_arch_from_sd(sd)
        if detected != "unknown" and detected != expected_arch:
            print(
                f"\n  WARNING: CNN architecture mismatch!\n"
                f"    requested : {expected_arch}\n"
                f"    checkpoint: {detected}  (ckpt={ckpt_path})\n"
            )
        else:
            print(f"  [arch check] CNN arch detected in ckpt: {detected}"
                  f"  (requested: {expected_arch}) OK")

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model
