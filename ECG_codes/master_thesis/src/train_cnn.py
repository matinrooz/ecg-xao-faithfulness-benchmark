import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class ECGCNN1D_Baseline(nn.Module):
    def __init__(self, in_ch=12, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class ECGCNN1D_Deeper(nn.Module):
    def __init__(self, in_ch=12, num_classes=5, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
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

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        pred = out.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        choices=[2, 5],
        help="Number of classes: 2 for binary (NORM/MI), 5 for full setup"
    )

    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class-weighted CrossEntropyLoss"
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="baseline",
        choices=["baseline", "deeper"],
        help="Model architecture"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout for deeper model"
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value; set <=0 to disable"
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed)

    xtr = os.path.join(args.data_dir, "X_train.npy")
    ytr = os.path.join(args.data_dir, "y_train.npy")
    xva = os.path.join(args.data_dir, "X_val.npy")
    yva = os.path.join(args.data_dir, "y_val.npy")

    for p in [xtr, ytr, xva, yva]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    ds_tr = NPYECGDataset(xtr, ytr, mmap=True)
    ds_va = NPYECGDataset(xva, yva, mmap=True)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.arch == "baseline":
        print(f"Using architecture: baseline | num_classes={args.num_classes}")
        model = ECGCNN1D_Baseline(
            in_ch=12,
            num_classes=args.num_classes
        ).to(device)
    else:
        print(
            f"Using architecture: deeper | "
            f"num_classes={args.num_classes} | dropout={args.dropout}"
        )
        model = ECGCNN1D_Deeper(
            in_ch=12,
            num_classes=args.num_classes,
            dropout=args.dropout
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    num_classes = args.num_classes

    if args.use_class_weights:
        print("Using class-weighted CrossEntropyLoss")
        y_train_np = np.load(ytr).astype(np.int64)
        class_counts = np.bincount(y_train_np, minlength=num_classes)

        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

        print("class_counts:", class_counts.tolist())
        print("class_weights:", weights.tolist())

        criterion = nn.CrossEntropyLoss(weight=weights_t)
    else:
        print("Using unweighted CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    best_path = os.path.join(args.out_dir, "best.pt")
    log_path = os.path.join(args.out_dir, "train_log.txt")
    config_path = os.path.join(args.out_dir, "train_config.json")

    config = {
        "arch": args.arch,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "use_class_weights": bool(args.use_class_weights),
        "grad_clip": args.grad_clip,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    with open(log_path, "w") as log_f:
        log_f.write(f"seed={args.seed}\n")
        log_f.write(f"arch={args.arch}\n")
        log_f.write(f"num_classes={args.num_classes}\n")
        log_f.write(f"use_class_weights={args.use_class_weights}\n")
        log_f.write(f"epochs={args.epochs}\n")
        log_f.write(f"batch_size={args.batch_size}\n")
        log_f.write(f"lr={args.lr}\n")
        log_f.write(f"weight_decay={args.weight_decay}\n")
        log_f.write(f"dropout={args.dropout}\n")
        log_f.write(f"grad_clip={args.grad_clip}\n")
        log_f.write("-" * 60 + "\n")

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0

            for x, y in dl_tr:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()

                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                running_loss += loss.item() * x.size(0)

            avg_loss = running_loss / len(ds_tr)
            train_acc = evaluate_accuracy(model, dl_tr, device)
            val_acc = evaluate_accuracy(model, dl_va, device)

            scheduler.step(val_acc)

            line = (
                f"epoch={epoch+1:03d} | "
                f"loss={avg_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )
            print(line)
            log_f.write(line + "\n")
            log_f.flush()

            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), best_path)
                print(f"Saved new best model to: {best_path}")

        final_line = f"best_val_acc={best_val:.4f}"
        print(final_line)
        log_f.write("-" * 60 + "\n")
        log_f.write(final_line + "\n")

    print("Training finished.")
    print("Best model saved to:", best_path)
    print("Train log saved to:", log_path)
    print("Config saved to:", config_path)


if __name__ == "__main__":
    main()
