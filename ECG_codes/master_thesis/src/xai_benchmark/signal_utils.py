"""ECG signal utilities for R-peak detection, HR estimation, synthetic healthy ECG and calculation of alignment"""

from typing import Tuple

import numpy as np
from scipy.signal import find_peaks

try:
    import neurokit2 as nk
except ImportError as exc:
    raise ImportError("Install neurokit2 first:  pip install neurokit2") from exc


def shift_with_padding(x: np.ndarray, shift: int) -> np.ndarray:
    #Shift a (C, T) signal in time; edges are padded with the boundary value, not wrapped
    if shift == 0:
        return x.copy()
    out = np.empty_like(x)
    if shift > 0:
        out[:, shift:] = x[:, :-shift]
        out[:, :shift] = x[:, :1]
    else:
        s = -shift
        out[:, :-s] = x[:, s:]
        out[:, -s:] = x[:, -1:]
    return out


def detect_r_peaks(
    signal_1d: np.ndarray,
    fs: int = 100,
    min_rr_sec: float = 0.5,
    prominence: float = 0.5,
) -> np.ndarray:
    #R-peak positions in a 1D ECG via scipy find_peaks
    peaks, _ = find_peaks(
        signal_1d,
        distance=int(fs * min_rr_sec),
        prominence=prominence,
    )
    return peaks


def estimate_hr_bpm(
    signal_1d: np.ndarray,
    fs: int = 100,
    min_rr_sec: float = 0.5,
) -> float:
    #Heart rate (BPM) from the mean R-R interval and NaN if fewer than two peaks
    peaks = detect_r_peaks(signal_1d, fs=fs, min_rr_sec=min_rr_sec)
    if len(peaks) < 2:
        return float("nan")
    mean_rr = float(np.mean(np.diff(peaks) / fs))
    return float("nan") if mean_rr <= 0 else 60.0 / mean_rr


def make_synthetic_healthy_ecg(
    hr: float,
    fs: int = 100,
    T: int = 1000,
    n_leads: int = 12,
    method: str = "ecgsyn",
    per_lead_phase: bool = True,
    seed: int = 0,
) -> np.ndarray:
    
    #Generate a synthetic healthy ECG with optional lead-wise phase shifts.
    
    duration = T / fs
    try:
        ecg_1d = nk.ecg_simulate(
            duration=duration, sampling_rate=fs,
            heart_rate=hr, method=method,
        )
    except Exception:
        ecg_1d = nk.ecg_simulate(
            duration=duration, sampling_rate=fs,
            heart_rate=hr, method="simple",
        )

    ecg_1d = np.asarray(ecg_1d, dtype=np.float32)
    if len(ecg_1d) > T:
        ecg_1d = ecg_1d[:T]
    elif len(ecg_1d) < T:
        ecg_1d = np.pad(ecg_1d, (0, T - len(ecg_1d)))

    if per_lead_phase:
        rng = np.random.default_rng(seed)
        leads = [np.roll(ecg_1d, int(rng.integers(-5, 6))) for _ in range(n_leads)]
        x = np.stack(leads, axis=0).astype(np.float32)
    else:
        x = np.tile(ecg_1d, (n_leads, 1)).astype(np.float32)

    mean = x.mean(axis=1, keepdims=True)
    std  = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def align_synthetic_to_mi(
    x_mi: np.ndarray,
    x_syn: np.ndarray,
    fs: int = 100,
    lead_idx: int = 1,
    prominence: float = 0.5,
) -> Tuple[np.ndarray, int]:
    """
    Align the synthetic ECG to the MI signal.
    Returns the aligned signal and applied shift.
    """
    peaks_mi = detect_r_peaks(x_mi[lead_idx], fs=fs, prominence=prominence)
    try:
        _, info = nk.ecg_peaks(x_syn[lead_idx], sampling_rate=fs)
        peaks_syn = np.array(info["ECG_R_Peaks"])
    except Exception:
        peaks_syn = detect_r_peaks(x_syn[lead_idx], fs=fs, prominence=prominence)

    if len(peaks_mi) == 0 or len(peaks_syn) == 0:
        return x_syn.copy(), 0

    ref_mi  = int(peaks_mi[len(peaks_mi) // 2])
    ref_syn = int(peaks_syn[len(peaks_syn) // 2])
    shift   = ref_mi - ref_syn
    return shift_with_padding(x_syn, shift), shift
