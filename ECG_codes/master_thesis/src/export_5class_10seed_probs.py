"""
Export per-sample 5-class test probabilities for the two FINAL 10-seed experiments:
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
DATA   = os.path.join(ROOT, "data", "processed", "ptbxl_lr_5class")
RESULT = os.path.join(ROOT, "result")
OUT    = os.path.join(RESULT, "figures_5class_10seed", "predictions")

SEEDS       = [42, 123, 777, 2024, 9999, 47, 48, 49, 50, 51]
NUM_CLASSES = 5
CLASS_ORDER = ["CD", "HYP", "MI", "NORM", "STTC"]

MODELS = {
    "cnn":        "cnn_final/deeper_cnn_seed_{s}",
    "cnn_bilstm": "cnn_lstm_attn_weighted_final/cnn_lstm_attn_weighted_{s}",
}


class ECGCNN1D_Deeper(nn.Module):
    def __init__(self, in_ch=12, num_classes=5, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))


class ECGCNNLSTMAttn(nn.Module):
    def __init__(self, in_ch=12, num_classes=5, dropout=0.3, lstm_hidden=64, attn_hidden=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, lstm_hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        d = lstm_hidden * 2
        self.attn_mlp = nn.Sequential(nn.Linear(d, attn_hidden), nn.Tanh(),
                                      nn.Linear(attn_hidden, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn_mlp(out), dim=1)
        ctx = torch.sum(w * out, dim=1)
        return self.fc(self.dropout(ctx))


def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in ckpt.items()}


def build_model(model_key, sd):
    if model_key == "cnn":
        assert any(k.startswith("features.") for k in sd), "expected deeper CNN (features.*)"
        assert not any(k.startswith("lstm.") for k in sd), "unexpected lstm.* in CNN ckpt"
        return ECGCNN1D_Deeper(num_classes=NUM_CLASSES)
    assert any(k.startswith("lstm.") for k in sd), "expected CNN-BiLSTM (lstm.*)"
    return ECGCNNLSTMAttn(num_classes=NUM_CLASSES)


def main():
    required = [os.path.join(DATA, f) for f in ("X_test.npy", "y_test.npy", "meta.json")]
    for tmpl in MODELS.values():
        required += [os.path.join(RESULT, tmpl.format(s=s), "best.pt") for s in SEEDS]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("STOP - missing required files:")
        for m in missing:
            print("  ", m)
        sys.exit(1)

    meta = json.load(open(os.path.join(DATA, "meta.json")))
    order = [meta["idx_to_class"][str(i)] for i in range(NUM_CLASSES)]
    assert order == CLASS_ORDER, f"class order mismatch: {order} != {CLASS_ORDER}"

    X = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA, "y_test.npy")).astype(np.int64)
    assert X.ndim == 3 and X.shape[1:] == (12, 1000), X.shape
    assert y.shape == (X.shape[0],) and set(np.unique(y).tolist()) <= set(range(NUM_CLASSES))

    os.makedirs(OUT, exist_ok=True)
    xt = torch.from_numpy(X)
    with torch.no_grad():
        for key, tmpl in MODELS.items():
            for s in SEEDS:
                sd = load_state_dict(os.path.join(RESULT, tmpl.format(s=s), "best.pt"))
                model = build_model(key, sd)
                model.load_state_dict(sd, strict=True)
                model.eval()
                logits = [model(xt[i:i + 256]) for i in range(0, len(xt), 256)]
                prob = torch.softmax(torch.cat(logits), dim=1).numpy().astype(np.float32)
                assert prob.shape == (len(y), NUM_CLASSES), prob.shape
                np.save(os.path.join(OUT, f"{key}_seed_{s}_probs.npy"), prob)
                np.save(os.path.join(OUT, f"{key}_seed_{s}_true.npy"), y)
                print(f"  loaded+inferred {key:11s} seed {s:5d}  strict=OK  probs {prob.shape}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
