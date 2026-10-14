
#Reproduce the band-pass filtering illustration (thesis Figure 14):


import os
import argparse

import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


DATA_ROOT_DEFAULT = "/project/data/shared_all/PTB-XL-ECG-dataset"


def bandpass_filter(sig, fs=100, low=0.5, high=40.0, order=4):
    """4th-order Butterworth band-pass, zero-phase, along time (sig shape (T, 12))."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT_DEFAULT,
                    help="PTB-XL root (contains ptbxl_database.csv and the records)")
    ap.add_argument("--record", default="records100/00000/00001_lr",
                    help="Record path relative to data_root (100 Hz, without extension)")
    ap.add_argument("--lead", type=int, default=0, help="Lead index (0 = Lead I)")
    ap.add_argument("--low", type=float, default=0.5)
    ap.add_argument("--high", type=float, default=40.0)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--out", default="bandpass_example.png")
    args = ap.parse_args()

    rec_path = os.path.join(args.data_root, args.record)
    sig, meta = wfdb.rdsamp(rec_path)          # sig: (T, 12)
    fs = int(meta.get("fs", 100))

    sig_f = bandpass_filter(sig, fs=fs, low=args.low, high=args.high, order=args.order)

    raw      = sig[:, args.lead]
    filtered = sig_f[:, args.lead]
    lead_name = ["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"][args.lead]

    plt.figure(figsize=(12, 4))
    plt.plot(raw,      label="Original ECG")
    plt.plot(filtered, label="Band-pass Filtered ECG")
    plt.title(f"Lead {lead_name}: Original ECG vs Band-pass Filtered ECG")
    plt.xlabel("Time step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
