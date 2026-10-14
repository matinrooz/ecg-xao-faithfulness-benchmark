#evaluation for CNN_Bilstm + attention model 
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize


class NPYECGDataset(Dataset):
    def __init__(self, x_path, y_path, mmap=True):
        self.X = np.load(x_path, mmap_mode="r" if mmap else None)
        self.y = np.load(y_path, mmap_mode="r" if mmap else None)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


class ECGCNNLSTMAttn(nn.Module):
    def __init__(
        self,
        in_ch=12,
        num_classes=5,
        dropout=0.3,
        lstm_hidden=64,
        attn_hidden=64,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        lstm_out_dim = lstm_hidden * 2

        self.attn_mlp = nn.Sequential(
            nn.Linear(lstm_out_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)          # (B, 128, T')
        x = x.permute(0, 2, 1)   # (B, T', 128)

        out, _ = self.lstm(x)    # (B, T', 2H)

        attn_scores = self.attn_mlp(out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * out, dim=1)

        context = self.dropout(context)
        logits = self.fc(context)
        return logits


def get_class_names(num_classes):
    if num_classes == 2:
        return {
            0: "NORM",
            1: "MI",
        }
    elif num_classes == 5:
        return {
            0: "NORM",
            1: "MI",
            2: "STTC",
            3: "CD",
            4: "HYP",
        }
    else:
        return {i: f"class_{i}" for i in range(num_classes)}


def load_state_dict_flexible(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]

    return ckpt


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for x, y in loader:
        x = x.to(device)
        out = model(x)

        prob = torch.softmax(out, dim=1).cpu().numpy()
        pred = out.argmax(dim=1).cpu().numpy()

        y_true.extend(y.numpy())
        y_pred.extend(pred)
        y_prob.extend(prob)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    return y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(x) for x in per_class_f1],
        "confusion_matrix": cm.tolist(),
    }

    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            ap = average_precision_score(y_true, y_prob[:, 1])
            metrics["auc"] = float(auc)
            metrics["average_precision"] = float(ap)
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            macro_auc = roc_auc_score(
                y_true_bin,
                y_prob,
                average="macro",
                multi_class="ovr",
            )
            macro_ap = average_precision_score(
                y_true_bin,
                y_prob,
                average="macro",
            )
            metrics["macro_auc"] = float(macro_auc)
            metrics["macro_average_precision"] = float(macro_ap)
    except ValueError:
        if num_classes == 2:
            metrics["auc"] = None
            metrics["average_precision"] = None
        else:
            metrics["macro_auc"] = None
            metrics["macro_average_precision"] = None

    if num_classes == 2:
        precision_mi = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall_mi = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_mi = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        metrics["precision_mi"] = float(precision_mi)
        metrics["recall_mi"] = float(recall_mi)
        metrics["f1_mi"] = float(f1_mi)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--attn_hidden", type=int, default=64)

    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        choices=[2, 5],
        help="2 for binary, 5 for full setup"
    )

    parser.add_argument(
        "--save_txt",
        action="store_true",
        help="Also save a *_eval.txt next to the checkpoint."
    )

    args = parser.parse_args()

    x_path = os.path.join(args.data_dir, f"X_{args.split}.npy")
    y_path = os.path.join(args.data_dir, f"y_{args.split}.npy")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing file: {y_path}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")

    ds = NPYECGDataset(x_path, y_path, mmap=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = ECGCNNLSTMAttn(
        in_ch=12,
        num_classes=args.num_classes,
        dropout=args.dropout,
        lstm_hidden=args.lstm_hidden,
        attn_hidden=args.attn_hidden,
    ).to(device)

    state_dict = load_state_dict_flexible(args.ckpt, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    y_true, y_pred, y_prob = run_eval(model, dl, device)
    metrics = compute_metrics(y_true, y_pred, y_prob, args.num_classes)
    class_names = get_class_names(args.num_classes)

    lines = []
    lines.append(f"Split: {args.split}")
    lines.append("Arch: cnn_lstm_attn")
    lines.append(f"Num classes: {args.num_classes}")
    lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"Macro F1: {metrics['macro_f1']:.4f}")

    if args.num_classes == 2:
        auc_val = metrics["auc"]
        ap_val = metrics["average_precision"]
        lines.append(f"AUC: {auc_val:.4f}" if auc_val is not None else "AUC: could not be computed")
        lines.append(f"Average Precision: {ap_val:.4f}" if ap_val is not None else "Average Precision: could not be computed")
        lines.append(f"Precision (MI): {metrics['precision_mi']:.4f}")
        lines.append(f"Recall (MI): {metrics['recall_mi']:.4f}")
        lines.append(f"F1 (MI): {metrics['f1_mi']:.4f}")
    else:
        auc_val = metrics["macro_auc"]
        ap_val = metrics["macro_average_precision"]
        lines.append(f"Macro AUC (OvR): {auc_val:.4f}" if auc_val is not None else "Macro AUC (OvR): could not be computed")
        lines.append(f"Macro Average Precision: {ap_val:.4f}" if ap_val is not None else "Macro Average Precision: could not be computed")

    lines.append("Per-class F1: " + str([round(float(x), 3) for x in metrics["per_class_f1"]]))
    lines.append("Class mapping: " + str(class_names))
    lines.append("Confusion matrix (rows=true, cols=pred):")
    lines.append(str(np.array(metrics["confusion_matrix"])))

    print("\n".join(lines))

    out_dir = os.path.dirname(args.ckpt)
    metrics_out = {
        "split": args.split,
        "arch": "cnn_lstm_attn",
        "num_classes": args.num_classes,
        "class_mapping": class_names,
        **metrics,
    }

    out_json = os.path.join(out_dir, f"metrics_{args.split}.json")
    with open(out_json, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Saved metrics to: {out_json}")

    if args.save_txt:
        out_txt = os.path.join(out_dir, f"{args.split}_eval.txt")
        with open(out_txt, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("Saved eval text to:", out_txt)


if __name__ == "__main__":
    main()
