import os
import ast
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt


DATA_ROOT_DEFAULT = "/project/data/shared_all/PTB-XL-ECG-dataset"

# binary NORM-vs-MI task, fixed label mapping for reproducibility
TARGET_CLASSES = ["NORM", "MI"]
CLASS_TO_IDX = {
    "NORM": 0,
    "MI": 1,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


def bandpass_filter(sig, fs=100, low=0.5, high=40.0, order=4):
    #Band-pass filter a (T, 12) signal along time
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=0)


def zscore_normalize(sig, eps=1e-8):
    #Per-lead z-score normalization
    mean = sig.mean(axis=0, keepdims=True)
    std = sig.std(axis=0, keepdims=True)
    return (sig - mean) / (std + eps)


def load_sig(data_root, filename_lr):
    #Load a record; returns sig (T, 12)
    rec_path = os.path.join(data_root, filename_lr)
    sig, meta = wfdb.rdsamp(rec_path)
    return sig.astype(np.float32), meta


def to_model_shape(sig):
    #(T, 12) -> (12, T) as expected by the models
    return sig.T.astype(np.float32)


def read_ptbxl_tables(data_root):
    db_path = os.path.join(data_root, "ptbxl_database.csv")
    scp_path = os.path.join(data_root, "scp_statements.csv")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Missing: {db_path}")
    if not os.path.exists(scp_path):
        raise FileNotFoundError(f"Missing: {scp_path}")

    df = pd.read_csv(db_path)
    scp = pd.read_csv(scp_path, index_col=0)
    return df, scp


def parse_scp_codes(x):
    if isinstance(x, dict):
        return x
    if not isinstance(x, str) or x.strip() == "":
        return {}
    return ast.literal_eval(x)


def build_code_to_class_map(scp):
    #Map diagnostic SCP codes to their superclass, restricted to TARGET_CLASSES
    scp_local = scp.copy()

    if "diagnostic" in scp_local.columns:
        scp_local = scp_local[scp_local["diagnostic"] == 1]

    if "diagnostic_class" not in scp_local.columns:
        raise KeyError("scp_statements.csv lacks 'diagnostic_class'")

    scp_local = scp_local[scp_local["diagnostic_class"].isin(TARGET_CLASSES)]
    return scp_local["diagnostic_class"].to_dict()


def assign_single_label(scp_dict, code_to_class):
    #Return the single TARGET_CLASSES superclass for a sample, or None if not exactly one
    classes = set()
    for code in scp_dict.keys():
        cls = code_to_class.get(code)
        if cls is not None:
            classes.add(cls)

    if len(classes) == 1:
        return next(iter(classes))
    return None


def build_arrays(
    df_subset,
    data_root,
    low=0.5,
    high=40.0,
    order=4,
    max_n=None,
    verbose_every=500,
):
    #Build X (N, 12, 1000) and y (N,) for a split
    if max_n is not None:
        df_subset = df_subset.iloc[:max_n].copy()

    n = len(df_subset)
    X = np.zeros((n, 12, 1000), dtype=np.float32)
    y = df_subset["y"].to_numpy(dtype=np.int64)

    for i, (_, row) in enumerate(df_subset.iterrows()):
        sig, meta = load_sig(data_root, row["filename_lr"])
        fs = int(meta.get("fs", 100))

        sig_f = bandpass_filter(sig, fs=fs, low=low, high=high, order=order)
        sig_n = zscore_normalize(sig_f)
        x = to_model_shape(sig_n)  # (12, 1000)

        if x.shape != (12, 1000):
            raise ValueError(f"Bad shape {x.shape} for {row['filename_lr']}")

        X[i] = x

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  built {i+1}/{n}")

    return X, y


def print_label_counts(name, df_part):
    counts = df_part["label"].value_counts().to_dict()
    print(f"{name} label counts: {counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=DATA_ROOT_DEFAULT)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.expanduser("~/master_thesis/data/processed/ptbxl_lr_binary_norm_mi"),
    )
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_val", type=int, default=None)
    ap.add_argument("--max_test", type=int, default=None)
    ap.add_argument("--low", type=float, default=0.5)
    ap.add_argument("--high", type=float, default=40.0)
    ap.add_argument("--order", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tables...")
    df, scp = read_ptbxl_tables(args.data_root)

    print("Parsing scp_codes...")
    df["scp_codes_parsed"] = df["scp_codes"].apply(parse_scp_codes)

    print("Building code->class map...")
    code_to_class = build_code_to_class_map(scp)

    print("Assigning single-label superclass...")
    df["label"] = df["scp_codes_parsed"].apply(lambda d: assign_single_label(d, code_to_class))

    df_clean = df[df["label"].isin(TARGET_CLASSES)].copy()
    df_clean["y"] = df_clean["label"].map(CLASS_TO_IDX).astype(int)

    if "strat_fold" not in df_clean.columns:
        raise KeyError("Missing 'strat_fold' in ptbxl_database.csv")

    # PTB-XL official split
    train_df = df_clean[df_clean["strat_fold"] <= 8].copy()
    val_df = df_clean[df_clean["strat_fold"] == 9].copy()
    test_df = df_clean[df_clean["strat_fold"] == 10].copy()

    print("Counts:", {"train": len(train_df), "val": len(val_df), "test": len(test_df)})
    print("class_to_idx:", CLASS_TO_IDX)
    print_label_counts("TRAIN", train_df)
    print_label_counts("VAL", val_df)
    print_label_counts("TEST", test_df)

    cols = ["ecg_id", "filename_lr", "label", "y", "strat_fold"]
    train_df[cols].to_csv(out_dir / "train_index.csv", index=False)
    val_df[cols].to_csv(out_dir / "val_index.csv", index=False)
    test_df[cols].to_csv(out_dir / "test_index.csv", index=False)

    meta = {
        "data_root": args.data_root,
        "target_classes": TARGET_CLASSES,
        "classes_present": TARGET_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "preprocess": {
            "bandpass": [args.low, args.high],
            "order": args.order,
            "normalize": "zscore_per_lead",
        },
        "split": {
            "train": "<=8",
            "val": "==9",
            "test": "==10",
        },
        "task": "binary_classification",
        "positive_class": "MI",
        "negative_class": "NORM",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\nBuilding TRAIN arrays...")
    X_train, y_train = build_arrays(
        train_df,
        args.data_root,
        low=args.low,
        high=args.high,
        order=args.order,
        max_n=args.max_train,
    )

    print("\nBuilding VAL arrays...")
    X_val, y_val = build_arrays(
        val_df,
        args.data_root,
        low=args.low,
        high=args.high,
        order=args.order,
        max_n=args.max_val,
    )

    print("\nBuilding TEST arrays...")
    X_test, y_test = build_arrays(
        test_df,
        args.data_root,
        low=args.low,
        high=args.high,
        order=args.order,
        max_n=args.max_test,
    )

    print("\nSaving .npy files...")
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)

    print("\nSaved files:")
    print(" ", out_dir / "X_train.npy")
    print(" ", out_dir / "y_train.npy")
    print(" ", out_dir / "X_val.npy")
    print(" ", out_dir / "y_val.npy")
    print(" ", out_dir / "X_test.npy")
    print(" ", out_dir / "y_test.npy")
    print(" ", out_dir / "train_index.csv")
    print(" ", out_dir / "val_index.csv")
    print(" ", out_dir / "test_index.csv")
    print(" ", out_dir / "meta.json")

    print("\nDone. Output:", out_dir)


if __name__ == "__main__":
    main()
