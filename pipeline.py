"""
pipeline.py
===========
Data loading, dataset, training, and evaluation utilities for the
high-to-low rate profile reconstruction benchmark on BatteryOCV.

Main entry points:
  - build_dataloaders(...)        : in-distribution train/val/test loaders
  - build_crossmodel_bundle(...)  : zero-shot cross-model evaluation bundle
  - train_one_epoch / validate_one_epoch / evaluate_ae_predictions_aux

Both loaders read BatteryOCV.h5 + BatteryOCV_split.json from the working
directory by default; pass `h5_path=` and `split_json_path=` to override.
"""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import find_peaks, savgol_filter
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


def smooth_peak_signal(
    y: np.ndarray,
    window: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Smooth a 1D profile before dV/dQ peak counting."""
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1 or y.size < 3:
        return y

    win = int(window)
    if win % 2 == 0:
        win += 1
    win = min(win, y.size if y.size % 2 == 1 else y.size - 1)
    if win <= polyorder or win < 3:
        return y
    return savgol_filter(y, window_length=win, polyorder=min(polyorder, win - 1), mode="interp")

# =========================================================
# CONFIG
# =========================================================
DB_ROOT_PUBLIC_H5  = Path("BatteryOCV.h5")
SPLIT_JSON_PATH    = Path("BatteryOCV_split.json")

RANDOM_SEED  = 42

DEFAULT_CHANNEL_VARIANT = "voltage_temp_current_channel"

WINDOW_SIZE      = 64
USE_HALF_PROFILE = True
BATCH_SIZE       = 32
NUM_WORKERS      = 0

VOLTAGE_MIN = 2.5
VOLTAGE_MAX = 4.4
TEMP_MIN    = -20.0
TEMP_MAX    =  60.0


# =========================================================
# BASIC UTILS
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def minmax_scale_voltage(v: np.ndarray) -> np.ndarray:
    return (v - VOLTAGE_MIN) / (VOLTAGE_MAX - VOLTAGE_MIN)


def minmax_scale_temp_scalar(v: float) -> float:
    return (float(v) - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)


def minmax_scale_temp(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)


def downsample_128_to_64(x: np.ndarray) -> np.ndarray:
    return x[::2]


def safe_zero_pad_rows(arr: np.ndarray, left_pad: int, right_pad: int) -> np.ndarray:
    if left_pad == 0 and right_pad == 0:
        return arr
    return np.pad(arr, ((left_pad, right_pad), (0, 0)),
                  mode="constant", constant_values=0.0)


def safe_scalar(x, default: float = 0.0) -> np.float32:
    x = np.float32(x)
    return x if np.isfinite(x) else np.float32(default)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# COLUMN INDEX (BatteryOCV.h5 layout)
# =========================================================
# Each row in BatteryOCV.h5 has 775 columns:
#   col 0 : model_idx (0=A, 1=B, 2=C, 3=D, 4=E)
#   col 1 : cell_idx  (integer cell identifier; matches BatteryOCV_split.json)
#   col 2 : cycle
#   col 3 : charge_capacity     (normalized by nominal capacity)
#   col 4 : discharge_capacity  (normalized by nominal capacity)
#   col 5 : charge_temp         (median chamber temp during charge, deg C)
#   col 6 : discharge_temp
#   col 7..134  : charge_voltage_profile     (128 points)
#   col 135..262: discharge_voltage_profile  (128 points)
#   col 263..390: charge_temp_profile        (128 points)
#   col 391..518: discharge_temp_profile     (128 points)
#   col 519..646: charge_current_profile     (128 points, C-rate)
#   col 647..774: discharge_current_profile  (128 points, C-rate)
#
# Normalization notes:
#   - Voltage profiles are min-max scaled to the operating window [2.5, 4.4] V.
#     Recorded values may slightly exceed [0, 1] at the boundaries because of
#     instrument sampling near the voltage cutoffs.
#   - Capacities are normalized by each cell's nominal (design) capacity,
#     not by the measured peak. Healthy cells whose measured capacity exceeds
#     the design value report values above 1.0 (typically up to ~1.15).
COL = {
    "model":                      0,
    "cell":                       1,
    "cycle":                      2,
    "charge_capacity":            3,
    "discharge_capacity":         4,
    "charge_temp":                5,
    "discharge_temp":             6,
    "charge_voltage_profile":     slice(7,   135),
    "discharge_voltage_profile":  slice(135, 263),
    "charge_temp_profile":        slice(263, 391),
    "discharge_temp_profile":     slice(391, 519),
    "charge_current_profile":     slice(519, 647),
    "discharge_current_profile":  slice(647, 775),
}


# =========================================================
# H5 / SPLIT LOADERS
# =========================================================
def load_h5_full(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load entire highrate_cycle and lowrate_cycle arrays from h5.

    Returns
    -------
    hr_all : np.ndarray of shape (N_hr, 775)
    lr_all : np.ndarray of shape (N_lr, 775)
    """
    with h5py.File(str(h5_path), "r") as f:
        hr_all = f["highrate_cycle"][:]
        lr_all = f["lowrate_cycle"][:]
    return np.asarray(hr_all, dtype=np.float32), np.asarray(lr_all, dtype=np.float32)


def load_public_split(split_json_path: Path) -> Dict:
    """Load BatteryOCV_split.json.

    Returns dict with keys:
        in_distribution : {train, val, test}  (each is list of cell_idx)
        cross_model     : {test}              (list of cell_idx)
        model_map       : {A:0, B:1, ..., E:4}
        random_state    : int
    """
    with open(split_json_path) as f:
        return json.load(f)


# =========================================================
# PROFILE / AUX EXTRACTION
# =========================================================
def build_voltage_profile_from_row(
    row: np.ndarray, use_half_profile: bool = True
) -> np.ndarray:
    """Returns (128,) = charge64 + discharge64.
    Values in the dataset are already min-max normalized to [0, 1]; pass through to avoid double normalization.
    """
    ch  = row[COL["charge_voltage_profile"]].astype(np.float32)
    dis = row[COL["discharge_voltage_profile"]].astype(np.float32)
    if use_half_profile:
        ch  = downsample_128_to_64(ch)
        dis = downsample_128_to_64(dis)
    return np.nan_to_num(np.concatenate([ch, dis]).astype(np.float32), nan=0.0)


def build_channel_profile_from_row(
    row: np.ndarray,
    profile_type: str,
    use_half_profile: bool = True,
) -> np.ndarray:
    """Returns charge+discharge profile for one input channel.

    Voltage is already normalized in DB and is used as-is. Temperature is raw
    Celsius and must be scaled to [-20, 60]. Current is c-rate and is used as-is.
    """
    if profile_type == "voltage":
        ch = row[COL["charge_voltage_profile"]].astype(np.float32)
        dis = row[COL["discharge_voltage_profile"]].astype(np.float32)
    elif profile_type == "temp":
        ch = minmax_scale_temp(row[COL["charge_temp_profile"]].astype(np.float32))
        dis = minmax_scale_temp(row[COL["discharge_temp_profile"]].astype(np.float32))
    elif profile_type == "current":
        ch = row[COL["charge_current_profile"]].astype(np.float32)
        dis = row[COL["discharge_current_profile"]].astype(np.float32)
    elif profile_type == "capacity":
        raw_len = len(row[COL["charge_voltage_profile"]])
        ch = np.linspace(
            0.0, safe_scalar(row[COL["charge_capacity"]]), raw_len, dtype=np.float32)
        dis = np.linspace(
            0.0, safe_scalar(row[COL["discharge_capacity"]]), raw_len, dtype=np.float32)
    else:
        raise ValueError(f"Unknown profile_type: {profile_type}")

    if use_half_profile:
        ch = downsample_128_to_64(ch)
        dis = downsample_128_to_64(dis)
    return np.nan_to_num(np.concatenate([ch, dis]).astype(np.float32), nan=0.0)


def build_multi_channel_input_from_row(
    row: np.ndarray,
    variant: str = DEFAULT_CHANNEL_VARIANT,
    use_half_profile: bool = True,
) -> np.ndarray:
    """Returns channel input [C, L] for channel variants."""
    if variant == "voltage_channel":
        channels = ["voltage"]
    elif variant == "voltage_current_channel":
        channels = ["voltage", "current"]
    elif variant == "voltage_temp_channel":
        channels = ["voltage", "temp"]
    elif variant == "voltage_temp_current_channel":
        channels = ["voltage", "temp", "current"]
    elif variant == "voltage_capacity_channel":
        channels = ["voltage", "capacity"]
    elif variant == "voltage_temp_capacity_channel":
        channels = ["voltage", "temp", "capacity"]
    elif variant == "voltage_current_capacity_channel":
        channels = ["voltage", "current", "capacity"]
    elif variant == "voltage_temp_current_capacity_channel":
        channels = ["voltage", "temp", "current", "capacity"]
    else:
        raise ValueError(f"Unknown channel variant: {variant}")

    return np.stack([
        build_channel_profile_from_row(row, ch, use_half_profile)
        for ch in channels
    ], axis=0).astype(np.float32)


def is_channel_variant(variant: str) -> bool:
    return variant.endswith("_channel")


def extract_aux_features(row: np.ndarray, variant: str) -> np.ndarray:
    """
    aux_dim per variant:
        voltage               -> 2  (zeros, placeholder)
        voltage_temp          -> 2  [ch_temp_norm, dis_temp_norm]
        voltage_current       -> 2  [ch_cur_mean,  dis_cur_mean]
        voltage_capacity      -> 2  [ch_cap, dis_cap]
        voltage_temp_current  -> 4  [ch_temp, dis_temp, ch_cur_mean, dis_cur_mean]
        voltage_temp_capacity -> 4  [ch_temp, dis_temp, ch_cap, dis_cap]
        voltage_current_capacity -> 4  [ch_cur_mean, dis_cur_mean, ch_cap, dis_cap]
        voltage_temp_current_capacity -> 6
    """
    if variant == "voltage":
        return np.array([0.0, 0.0], dtype=np.float32)

    elif variant == "voltage_temp":
        return np.array([
            minmax_scale_temp_scalar(safe_scalar(row[COL["charge_temp"]])),
            minmax_scale_temp_scalar(safe_scalar(row[COL["discharge_temp"]])),
        ], dtype=np.float32)

    elif variant == "voltage_current":
        ch_cur  = row[COL["charge_current_profile"]].astype(np.float32)
        dis_cur = row[COL["discharge_current_profile"]].astype(np.float32)
        return np.array([float(np.nanmean(ch_cur)),
                         float(np.nanmean(dis_cur))], dtype=np.float32)

    elif variant == "voltage_capacity":
        return np.array([safe_scalar(row[COL["charge_capacity"]]),
                         safe_scalar(row[COL["discharge_capacity"]])],
                        dtype=np.float32)

    elif variant == "voltage_temp_current":
        ch_cur  = row[COL["charge_current_profile"]].astype(np.float32)
        dis_cur = row[COL["discharge_current_profile"]].astype(np.float32)
        return np.array([
            minmax_scale_temp_scalar(safe_scalar(row[COL["charge_temp"]])),
            minmax_scale_temp_scalar(safe_scalar(row[COL["discharge_temp"]])),
            float(np.nanmean(ch_cur)),
            float(np.nanmean(dis_cur)),
        ], dtype=np.float32)

    elif variant == "voltage_temp_capacity":
        return np.array([
            minmax_scale_temp_scalar(safe_scalar(row[COL["charge_temp"]])),
            minmax_scale_temp_scalar(safe_scalar(row[COL["discharge_temp"]])),
            safe_scalar(row[COL["charge_capacity"]]),
            safe_scalar(row[COL["discharge_capacity"]]),
        ], dtype=np.float32)

    elif variant == "voltage_current_capacity":
        ch_cur  = row[COL["charge_current_profile"]].astype(np.float32)
        dis_cur = row[COL["discharge_current_profile"]].astype(np.float32)
        return np.array([
            float(np.nanmean(ch_cur)),
            float(np.nanmean(dis_cur)),
            safe_scalar(row[COL["charge_capacity"]]),
            safe_scalar(row[COL["discharge_capacity"]]),
        ], dtype=np.float32)

    elif variant == "voltage_temp_current_capacity":
        ch_cur  = row[COL["charge_current_profile"]].astype(np.float32)
        dis_cur = row[COL["discharge_current_profile"]].astype(np.float32)
        return np.array([
            minmax_scale_temp_scalar(safe_scalar(row[COL["charge_temp"]])),
            minmax_scale_temp_scalar(safe_scalar(row[COL["discharge_temp"]])),
            float(np.nanmean(ch_cur)),
            float(np.nanmean(dis_cur)),
            safe_scalar(row[COL["charge_capacity"]]),
            safe_scalar(row[COL["discharge_capacity"]]),
        ], dtype=np.float32)

    else:
        raise ValueError(
            f"Unknown variant: {variant}. Choose from "
            f"['voltage', 'voltage_temp', 'voltage_current', "
            f"'voltage_capacity', 'voltage_temp_current', "
            f"'voltage_temp_capacity', 'voltage_current_capacity', "
            f"'voltage_temp_current_capacity']"
        )


# =========================================================
# WINDOW BUILDER
# =========================================================
def get_centered_window_with_padding(
    src_arr: np.ndarray, target_cycle: int, window_size: int = 64,) -> np.ndarray:
    # W=1 special case: return just the single target cycle row.
    if window_size == 1:
        cycle_values = src_arr[:, COL["cycle"]].astype(np.int64)
        idx = np.searchsorted(cycle_values, target_cycle)
        idx = int(np.clip(idx, 0, len(src_arr) - 1))
        return src_arr[idx : idx + 1]  # shape (1, n_features)

    assert window_size % 2 == 0
    half = window_size // 2
    cycle_values = src_arr[:, COL["cycle"]].astype(np.int64)
    idxs = np.where(cycle_values == int(target_cycle))[0]
    if len(idxs) == 0:
        return np.zeros((window_size, src_arr.shape[1]), dtype=np.float32)

    center_idx = int(idxs[0])
    start, end = center_idx - half, center_idx + half
    left_pad  = max(0, -start)
    right_pad = max(0, end - len(src_arr))
    seg = src_arr[max(0, start):min(len(src_arr), end)]
    seg = safe_zero_pad_rows(seg, left_pad, right_pad)

    if seg.shape[0] != window_size:
        raise ValueError(f"Window mismatch: got {seg.shape[0]}, expected {window_size}")
    return seg.astype(np.float32)


# =========================================================
# PAIR BUILDER
# =========================================================
def build_src_tgt_pairs(
    cell_idx_list:    List[int],
    h5_path:          Path,
    variant:          str  = "voltage",
    window_size:      int  = WINDOW_SIZE,
    use_half_profile: bool = USE_HALF_PROFILE,
    hr_all:           Optional[np.ndarray] = None,
    lr_all:           Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """Build (X_main, X_aux, Y_prof, Y_cap, meta) pairs for the given cells.

    Each low-rate target row is paired with a window of `window_size`
    high-rate rows centered on the target cycle (zero-padded at boundaries).

    Parameters
    ----------
    cell_idx_list : list of integer cell indices (matches BatteryOCV_split.json).
    h5_path       : path to BatteryOCV.h5.
    hr_all, lr_all: optional pre-loaded full arrays to avoid re-reading h5
                    when this function is called multiple times for different
                    splits.
    """
    if hr_all is None or lr_all is None:
        hr_all, lr_all = load_h5_full(h5_path)

    cell_col_hr = hr_all[:, COL["cell"]].astype(int)
    cell_col_lr = lr_all[:, COL["cell"]].astype(int)

    X_main_list, X_aux_list = [], []
    Y_prof_list, Y_cap_list = [], []
    meta_list = []

    for cell_idx in cell_idx_list:
        src_arr = hr_all[cell_col_hr == int(cell_idx)]
        tgt_arr = lr_all[cell_col_lr == int(cell_idx)]
        if len(src_arr) == 0 or len(tgt_arr) == 0:
            continue
        model_idx = int(src_arr[0, COL["model"]])

        for i in range(len(tgt_arr)):
            tgt_row   = tgt_arr[i]
            tgt_cycle = int(tgt_row[COL["cycle"]])
            if tgt_cycle == 0:
                tgt_cycle = 1

            src_window = get_centered_window_with_padding(
                src_arr, tgt_cycle, window_size)

            x_main = np.stack(
                [build_voltage_profile_from_row(r, use_half_profile)
                 for r in src_window], axis=0
            ).astype(np.float32)

            x_aux = np.stack(
                [extract_aux_features(r, variant) for r in src_window], axis=0
            ).astype(np.float32)

            y_prof = build_voltage_profile_from_row(
                tgt_row, use_half_profile)[None, :]
            y_cap  = np.array([
                tgt_row[COL["charge_capacity"]],
                tgt_row[COL["discharge_capacity"]],
            ], dtype=np.float32)

            X_main_list.append(x_main)
            X_aux_list.append(x_aux)
            Y_prof_list.append(y_prof)
            Y_cap_list.append(y_cap)
            meta_list.append({
                "cell": int(cell_idx),
                "model_idx": model_idx,
                "target_cycle": tgt_cycle,
            })

    if not X_main_list:
        raise ValueError(
            f"No valid pairs were created from h5 for cells {cell_idx_list}."
        )

    def to_tensor(lst):
        return torch.nan_to_num(
            torch.from_numpy(np.stack(lst).astype(np.float32)),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

    return (to_tensor(X_main_list), to_tensor(X_aux_list),
            to_tensor(Y_prof_list), to_tensor(Y_cap_list), meta_list)


def build_src_tgt_pairs_channel(
    cell_idx_list:    List[int],
    h5_path:          Path,
    variant:          str  = DEFAULT_CHANNEL_VARIANT,
    window_size:      int  = WINDOW_SIZE,
    use_half_profile: bool = USE_HALF_PROFILE,
    hr_all:           Optional[np.ndarray] = None,
    lr_all:           Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """Channel-mode variant of `build_src_tgt_pairs` — auxiliary signals
    (temperature, current, capacity) are stacked as additional channels of
    `x_main` instead of being fused as scalar features."""
    if hr_all is None or lr_all is None:
        hr_all, lr_all = load_h5_full(h5_path)

    cell_col_hr = hr_all[:, COL["cell"]].astype(int)
    cell_col_lr = lr_all[:, COL["cell"]].astype(int)

    X_main_list = []
    Y_prof_list, Y_cap_list = [], []
    meta_list = []

    for cell_idx in cell_idx_list:
        src_arr = hr_all[cell_col_hr == int(cell_idx)]
        tgt_arr = lr_all[cell_col_lr == int(cell_idx)]
        if len(src_arr) == 0 or len(tgt_arr) == 0:
            continue
        model_idx = int(src_arr[0, COL["model"]])

        for i in range(len(tgt_arr)):
            tgt_row = tgt_arr[i]
            tgt_cycle = int(tgt_row[COL["cycle"]])
            if tgt_cycle == 0:
                tgt_cycle = 1

            src_window = get_centered_window_with_padding(
                src_arr, tgt_cycle, window_size)

            x_main = np.stack([
                build_multi_channel_input_from_row(r, variant, use_half_profile)
                for r in src_window
            ], axis=0).astype(np.float32)

            y_prof = build_voltage_profile_from_row(
                tgt_row, use_half_profile)[None, :]
            y_cap = np.array([
                tgt_row[COL["charge_capacity"]],
                tgt_row[COL["discharge_capacity"]],
            ], dtype=np.float32)

            X_main_list.append(x_main)
            Y_prof_list.append(y_prof)
            Y_cap_list.append(y_cap)
            meta_list.append({
                "cell": int(cell_idx),
                "model_idx": model_idx,
                "target_cycle": tgt_cycle,
            })

    if not X_main_list:
        raise ValueError(
            f"No valid channel pairs were created from h5 for cells {cell_idx_list}."
        )

    def to_tensor(lst):
        return torch.nan_to_num(
            torch.from_numpy(np.stack(lst).astype(np.float32)),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

    X_main = to_tensor(X_main_list)
    X_aux = torch.zeros(
        (X_main.shape[0], X_main.shape[1], 0), dtype=torch.float32)
    return X_main, X_aux, to_tensor(Y_prof_list), to_tensor(Y_cap_list), meta_list


# =========================================================
# DATASET
# =========================================================
class BatteryPairDataset(Dataset):
    def __init__(self, X_main, X_aux, Y_prof, Y_cap, meta_list):
        assert len(X_main) == len(X_aux) == len(Y_prof) == len(Y_cap) == len(meta_list)
        self.X_main = X_main; self.X_aux = X_aux
        self.Y_prof = Y_prof; self.Y_cap = Y_cap; self.meta = meta_list

    def __len__(self): return self.X_main.size(0)

    def __getitem__(self, idx):
        return (self.X_main[idx], self.X_aux[idx],
                self.Y_prof[idx], self.Y_cap[idx], self.meta[idx])


# =========================================================
# BUILD DATALOADERS
# =========================================================
def build_dataloaders(
    variant:          str  = DEFAULT_CHANNEL_VARIANT,
    window_size:      int  = WINDOW_SIZE,
    use_half_profile: bool = USE_HALF_PROFILE,
    batch_size:       int  = BATCH_SIZE,
    num_workers:      int  = NUM_WORKERS,
    h5_path:          Optional[Path] = None,
    split_json_path:  Optional[Path] = None,
) -> Dict:
    """Build train / val / test DataLoaders from BatteryOCV.h5.

    Reads BatteryOCV_split.json for the cell-disjoint train/val/test split
    over Model E (in-distribution). Cross-model evaluation on Models A-D
    is handled separately by `build_crossmodel_bundle`.

    Note: this function does not call set_seed. Resetting the seed inside
    data loading would homogenize model initialization across variants and
    contaminate ablation comparisons; the calling script should call
    set_seed(...) once at the start.
    """
    _h5_path    = Path(h5_path)         if h5_path is not None         else DB_ROOT_PUBLIC_H5
    _split_path = Path(split_json_path) if split_json_path is not None else SPLIT_JSON_PATH

    split_dict  = load_public_split(_split_path)
    train_cells = list(split_dict["in_distribution"]["train"])
    val_cells   = list(split_dict["in_distribution"]["val"])
    test_cells  = list(split_dict["in_distribution"]["test"])

    # Load h5 once and reuse for all three splits
    hr_all, lr_all = load_h5_full(_h5_path)

    def _build(cells):
        if is_channel_variant(variant):
            return build_src_tgt_pairs_channel(
                cells, _h5_path, variant, window_size, use_half_profile,
                hr_all=hr_all, lr_all=lr_all)
        return build_src_tgt_pairs(
            cells, _h5_path, variant, window_size, use_half_profile,
            hr_all=hr_all, lr_all=lr_all)

    X_tr, Xa_tr, Yp_tr, Yc_tr, m_tr = _build(train_cells)
    X_v,  Xa_v,  Yp_v,  Yc_v,  m_v  = _build(val_cells)
    X_te, Xa_te, Yp_te, Yc_te, m_te = _build(test_cells)

    train_ds = BatteryPairDataset(X_tr, Xa_tr, Yp_tr, Yc_tr, m_tr)
    val_ds   = BatteryPairDataset(X_v,  Xa_v,  Yp_v,  Yc_v,  m_v)
    test_ds  = BatteryPairDataset(X_te, Xa_te, Yp_te, Yc_te, m_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers, pin_memory=True)

    print("=" * 60)
    print(f"Variant      : {variant}")
    print(f"Window size  : {window_size}")
    print(f"Train cells  : {len(train_cells):4d} | samples: {len(train_ds):5d}")
    print(f"Val cells    : {len(val_cells):4d} | samples: {len(val_ds):5d}")
    print(f"Test cells   : {len(test_cells):4d} | samples: {len(test_ds):5d}")
    print(f"Profile len  : {X_tr.shape[-1]}")
    if is_channel_variant(variant):
        print(f"In channels  : {X_tr.shape[2]}")
    else:
        print(f"Aux dim      : {Xa_tr.shape[-1]}")
    print("=" * 60)

    return {
        "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
        "train_dataset": train_ds, "val_dataset": val_ds, "test_dataset": test_ds,
        "train_cells": train_cells, "val_cells": val_cells, "test_cells": test_cells,
        "X_main_train": X_tr,  "X_aux_train": Xa_tr,
        "Y_prof_train": Yp_tr, "Y_cap_train": Yc_tr, "meta_train": m_tr,
        "X_main_val":   X_v,   "X_aux_val":   Xa_v,
        "Y_prof_val":   Yp_v,  "Y_cap_val":   Yc_v,  "meta_val":   m_v,
        "X_main_test":  X_te,  "X_aux_test":  Xa_te,
        "Y_prof_test":  Yp_te, "Y_cap_test":  Yc_te, "meta_test":  m_te,
        "profile_len": X_tr.shape[-1],
        "aux_dim":     Xa_tr.shape[-1],
        "in_channels": X_tr.shape[2] if X_tr.dim() == 4 else 1,
        "variant":     variant,
    }


# =========================================================
# BUILD CROSSMODEL BUNDLE
# =========================================================
def build_crossmodel_bundle(
    variant:          str  = DEFAULT_CHANNEL_VARIANT,
    window_size:      int  = WINDOW_SIZE,
    use_half_profile: bool = USE_HALF_PROFILE,
    batch_size:       int  = BATCH_SIZE,
    num_workers:      int  = NUM_WORKERS,
    h5_path:          Optional[Path] = None,
    split_json_path:  Optional[Path] = None,
) -> Dict:
    """Build a cross-model evaluation bundle for the held-out cells (Models A-D).

    Cross-model cells are listed under split["cross_model"]["test"] in
    BatteryOCV_split.json. Inference only — no train/val partition is
    formed; all cells are placed in `test_loader`.
    """
    _h5_path    = Path(h5_path)         if h5_path is not None         else DB_ROOT_PUBLIC_H5
    _split_path = Path(split_json_path) if split_json_path is not None else SPLIT_JSON_PATH

    split_dict = load_public_split(_split_path)
    crossmodel_cells = list(split_dict["cross_model"]["test"])

    # Compute per-model cell counts using model_idx in h5 for an informative print
    hr_all, lr_all = load_h5_full(_h5_path)
    cell_to_model = {}
    cell_set = set(crossmodel_cells)
    for row in hr_all:
        c = int(row[COL["cell"]])
        if c in cell_set and c not in cell_to_model:
            cell_to_model[c] = int(row[COL["model"]])
    idx_to_letter = {v: k for k, v in split_dict.get("model_map", {}).items()}
    model_counts = {}
    for c in crossmodel_cells:
        if c in cell_to_model:
            letter = idx_to_letter.get(cell_to_model[c], str(cell_to_model[c]))
            model_counts[letter] = model_counts.get(letter, 0) + 1

    print(f"[Cross-model bundle]")
    print(f"  Cells per model: {model_counts}")
    print(f"  Total cells    : {len(crossmodel_cells)}")

    if is_channel_variant(variant):
        X_te, Xa_te, Yp_te, Yc_te, m_te = build_src_tgt_pairs_channel(
            cell_idx_list=crossmodel_cells,
            h5_path=_h5_path,
            variant=variant,
            window_size=window_size,
            use_half_profile=use_half_profile,
            hr_all=hr_all, lr_all=lr_all,
        )
    else:
        X_te, Xa_te, Yp_te, Yc_te, m_te = build_src_tgt_pairs(
            cell_idx_list=crossmodel_cells,
            h5_path=_h5_path,
            variant=variant,
            window_size=window_size,
            use_half_profile=use_half_profile,
            hr_all=hr_all, lr_all=lr_all,
        )

    test_ds = BatteryPairDataset(X_te, Xa_te, Yp_te, Yc_te, m_te)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers, pin_memory=True,
    )

    print(f"  Total samples  : {len(test_ds)}")

    return {
        "test_loader":      test_loader,
        "test_dataset":     test_ds,
        "X_main_test":      X_te,
        "X_aux_test":       Xa_te,
        "Y_prof_test":      Yp_te,
        "Y_cap_test":       Yc_te,
        "meta_test":        m_te,
        "profile_len":      X_te.shape[-1],
        "aux_dim":          Xa_te.shape[-1],
        "in_channels":      X_te.shape[2] if X_te.dim() == 4 else 1,
        "crossmodel_cells": crossmodel_cells,
    }


# =========================================================
# BATCH TO DEVICE
# =========================================================
def move_batch_to_device(batch, device: torch.device):
    x_main, x_aux, y_prof, y_cap, meta = batch
    return (x_main.to(device, non_blocking=True),
            x_aux.to(device,  non_blocking=True),
            y_prof.to(device, non_blocking=True),
            y_cap.to(device,  non_blocking=True), meta)


def unpack_ae_outputs(outputs):
    """Return common AE outputs from deterministic or VAE-style model tuples."""
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise ValueError("Model output must be a tuple/list with at least y_prof and y_cap.")
    z = outputs[2] if len(outputs) > 2 else None
    return outputs[0], outputs[1], z


# =========================================================
# SEGMENT BIAS
# =========================================================
class SegmentBias(nn.Module):
    def __init__(self, length: int = 128, split: int = 64):
        super().__init__()
        assert split * 2 == length
        self.split = split
        self.seg = nn.Parameter(torch.zeros(2, split))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seg_bias = torch.cat([self.seg[0], self.seg[1]], dim=0)
        view_shape = (1,) * (x.dim() - 1) + (seg_bias.shape[0],)
        return x + seg_bias.view(view_shape)


# =========================================================
# MODEL
# =========================================================
class Conv1dLow2HighWithAux(nn.Module):
    """x_main: [B, W, 128], x_aux: [B, W, aux_dim]"""
    def __init__(
        self,
        profile_len:      int  = 128,
        window_size:      int  = 64,
        aux_dim:          int  = 2,
        latent_dim:       int  = 128,
        aux_embed_dim:    int  = 8,
        out_activation:   str  = "sigmoid",
        use_segment_bias: bool = False,
    ):
        super().__init__()
        self.aux_dim       = aux_dim
        self.aux_embed_dim = aux_embed_dim
        self.window_size   = window_size

        self.seg_bias = (SegmentBias(length=profile_len, split=profile_len // 2)
                         if use_segment_bias else nn.Identity())

        self.main_encoder = nn.Sequential(
            nn.Conv1d(window_size, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, 16, 3, stride=2, padding=1),          nn.ReLU(),
            nn.Conv1d(16,  8, 3, stride=2, padding=1),          nn.ReLU(),
        )
        main_feat_dim = 8 * 16
        self.main_to_latent = nn.Sequential(
            nn.Flatten(), nn.Linear(main_feat_dim, latent_dim), nn.ReLU(),
        )
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, aux_embed_dim), nn.ReLU(),
            nn.Linear(aux_embed_dim, aux_embed_dim), nn.ReLU(),
        )
        self.aux_to_latent = nn.Sequential(
            nn.Linear(window_size * aux_embed_dim, latent_dim // 2), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 2, latent_dim), nn.ReLU(),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, main_feat_dim), nn.ReLU(),
        )
        self.profile_decoder = nn.Sequential(
            nn.ConvTranspose1d(8,  16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32,  1, 3, stride=2, padding=1, output_padding=1),
        )
        self.cap_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), nn.ReLU(),
            nn.Linear(latent_dim // 2, 2),
        )
        self.out_act = (nn.Sigmoid() if out_activation == "sigmoid"
                        else nn.Tanh() if out_activation == "tanh"
                        else nn.Identity())

    def forward(self, x_main: torch.Tensor, x_aux: torch.Tensor):
        x_main = self.seg_bias(x_main)
        z_main = self.main_to_latent(self.main_encoder(x_main))
        b, w, _ = x_aux.shape
        z_aux   = self.aux_to_latent(self.aux_encoder(x_aux).reshape(b, -1))
        z       = self.fusion(torch.cat([z_main, z_aux], dim=1))
        h_dec   = self.from_latent(z).view(b, 8, 16)
        y_prof  = self.out_act(self.profile_decoder(h_dec))
        y_cap   = self.cap_head(z)
        return y_prof, y_cap, z


class Conv1dChannelAE(nn.Module):
    """Channel baseline. x_main: [B, W, C, L] -> [B, W*C, L]."""
    def __init__(
        self,
        profile_len:      int  = 128,
        window_size:      int  = 64,
        in_channels:      int  = 3,
        latent_dim:       int  = 128,
        out_activation:   str  = "sigmoid",
        use_segment_bias: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.total_in_ch = window_size * in_channels

        self.seg_bias = (SegmentBias(length=profile_len, split=profile_len // 2)
                         if use_segment_bias else nn.Identity())

        self.encoder = nn.Sequential(
            nn.Conv1d(self.total_in_ch, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, 3, stride=2, padding=1),               nn.ReLU(),
            nn.Conv1d(32, 16, 3, stride=2, padding=1),               nn.ReLU(),
        )
        enc_len = max(1, profile_len // 8)
        enc_dim = 16 * enc_len
        self.to_latent = nn.Sequential(
            nn.Flatten(), nn.Linear(enc_dim, latent_dim), nn.ReLU(),
        )
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, enc_dim), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64,  1, 3, stride=2, padding=1, output_padding=1),
        )
        self.cap_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), nn.ReLU(),
            nn.Linear(latent_dim // 2, 2),
        )
        self.out_act = (nn.Sigmoid() if out_activation == "sigmoid"
                        else nn.Tanh() if out_activation == "tanh"
                        else nn.Identity())

    def forward(self, x_main: torch.Tensor, x_aux: Optional[torch.Tensor] = None):
        if x_main.dim() == 3:
            b, w, l = x_main.shape
            x = x_main
        elif x_main.dim() == 4:
            b, w, c, l = x_main.shape
            x = x_main.reshape(b, w * c, l)
        else:
            raise ValueError(f"Expected x_main [B,W,L] or [B,W,C,L], got {x_main.shape}")

        x = self.seg_bias(x)
        z = self.to_latent(self.encoder(x))
        h = self.from_latent(z).view(b, 16, max(1, l // 8))
        y_prof = self.out_act(self.decoder(h))
        if y_prof.shape[-1] != l:
            y_prof = torch.nn.functional.interpolate(
                y_prof, size=l, mode="linear", align_corners=False)
        y_cap = self.cap_head(z)
        return y_prof, y_cap, z


# =========================================================
# LOSS
# =========================================================
@dataclass
class AELossConfig:
    prof_weight:   float = 100.0
    diff_weight:   float = 0.0
    cap_weight:    float = 5.0
    cap_loss_type: str   = "smoothl1"
    cap_beta:      float = 1.0
    use_diff_loss: bool  = False
    diff_order:    int   = 1


class AELoss(nn.Module):
    def __init__(self, cfg: AELossConfig):
        super().__init__()
        self.cfg = cfg
        self.prof_crit = nn.MSELoss()
        self.cap_crit  = (nn.SmoothL1Loss(beta=cfg.cap_beta)
                          if cfg.cap_loss_type.lower() == "smoothl1" else nn.MSELoss())

    def forward(self, yprof_pred, yprof_true, ycap_pred, ycap_true):
        loss_prof = self.prof_crit(yprof_pred, yprof_true)
        loss_cap  = self.cap_crit(ycap_pred, ycap_true)
        loss_diff = (self.prof_crit(torch.diff(yprof_pred, dim=-1),
                                   torch.diff(yprof_true, dim=-1))
                     if self.cfg.use_diff_loss
                     else torch.zeros((), device=yprof_pred.device))
        total = (self.cfg.prof_weight * loss_prof
                 + self.cfg.diff_weight * loss_diff
                 + self.cfg.cap_weight  * loss_cap)
        return {
            "loss_total":       total,
            "loss_prof":        loss_prof,
            "loss_diff":        loss_diff,
            "loss_cap":         loss_cap,
            "loss_prof_scaled": self.cfg.prof_weight * loss_prof,
            "loss_diff_scaled": self.cfg.diff_weight * loss_diff,
            "loss_cap_scaled":  self.cfg.cap_weight  * loss_cap,
        }


# =========================================================
# TRAIN / VALIDATION
# =========================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device,
                    grad_clip_norm: float = 1.0) -> Dict:
    model.train()
    sums = {k: 0.0 for k in ["loss_total", "loss_prof", "loss_diff", "loss_cap",
                               "loss_prof_scaled", "loss_diff_scaled", "loss_cap_scaled"]}
    n = 0
    for batch in loader:
        x_main, x_aux, y_prof, y_cap, _ = move_batch_to_device(batch, device)
        y_prof_pred, y_cap_pred, _ = unpack_ae_outputs(model(x_main, x_aux))
        loss_dict = loss_fn(y_prof_pred, y_prof, y_cap_pred, y_cap)
        optimizer.zero_grad(set_to_none=True)
        loss_dict["loss_total"].backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        bs = x_main.size(0)
        for k in sums: sums[k] += loss_dict[k].item() * bs
        n += bs
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device) -> Dict:
    model.eval()
    sums = {k: 0.0 for k in ["loss_total", "loss_prof", "loss_diff", "loss_cap",
                               "loss_prof_scaled", "loss_diff_scaled", "loss_cap_scaled"]}
    n = 0
    for batch in loader:
        x_main, x_aux, y_prof, y_cap, _ = move_batch_to_device(batch, device)
        y_prof_pred, y_cap_pred, _ = unpack_ae_outputs(model(x_main, x_aux))
        loss_dict = loss_fn(y_prof_pred, y_prof, y_cap_pred, y_cap)
        bs = x_main.size(0)
        for k in sums: sums[k] += loss_dict[k].item() * bs
        n += bs
    return {k: v / max(n, 1) for k, v in sums.items()}


# =========================================================
# CHECKPOINT
# =========================================================
def _move_opt_state(optimizer, device):
    for st in optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v): st[k] = v.to(device)


def save_ckpt_ae(path, epoch, model, optimizer, train_history, val_history,
                 best_val, loss_cfg, scheduler=None, extra=None):
    state = {
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_history": train_history, "val_history": val_history,
        "best_val": best_val, "loss_cfg": asdict(loss_cfg), "extra": extra or {},
    }
    if scheduler is not None: state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


def load_ckpt_ae(path, model, optimizer=None, scheduler=None, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        _move_opt_state(optimizer, device)
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return (ckpt.get("epoch", 0), ckpt.get("train_history", []),
            ckpt.get("val_history", []), ckpt.get("best_val", float("inf")),
            ckpt.get("loss_cfg", {}), ckpt.get("extra", {}))


# =========================================================
# LOSS PLOT
# =========================================================
def plot_loss_components(train_history, val_history, save_path=None):
    keys = ["loss_prof_scaled", "loss_diff_scaled", "loss_cap_scaled"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, k in zip(axes, keys):
        ax.plot([d[k] for d in train_history], label="Train")
        ax.plot([d[k] for d in val_history],   label="Val")
        ax.set_title(k); ax.set_yscale("log"); ax.grid(True); ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# =========================================================
# EVALUATION
# =========================================================
@torch.no_grad()
def evaluate_ae_predictions_aux(
    model, X_main, X_aux, Yprof, Ycap, device,
    meta_val=None, charge_len=64, prominence=0.02,
    peak_smooth_window=7, peak_smooth_polyorder=2,
    charge_peak_ylim=(0, 2), discharge_peak_ylim=(-2, 0),
    plot_best_worst_cell_profile=True, plot_cap_scatter=True,
    plot_best_worst_cap_aligned_profile=True, plot_best_worst_peak=True,
    plot_one_cell_trend=False, trend_cell_value=None,
    normalize_one_cell_from_first=False,
    plot_all_cell_trends=False, normalize_all_cells_from_first=False,
    max_cells_to_plot=None, alpha_all_cells=0.25,
    plot_cell_slope_scatter=True,
    save_dir=None,
    result_csv_name="evaluation_summary.csv",
    cell_csv_name="evaluation_per_cell.csv",
    sample_csv_name="evaluation_per_sample.csv",
    trend_cell_key="cell", trend_cycle_key="target_cycle",
):
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(fig, filename):
        if save_dir is not None:
            fig.savefig(save_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def to_np(x):
        if isinstance(x, np.ndarray): return x
        if torch.is_tensor(x): return x.detach().cpu().numpy()
        return np.asarray(x)

    def cap_aligned_rmse(yt, ct, yp, cp, n_grid=128):
        xmax = min(ct, cp)
        if xmax <= 0: return np.nan
        xg = np.linspace(0, xmax, n_grid)
        return np.sqrt(np.mean(
            (np.interp(xg, np.linspace(0, ct, len(yt)), yt) -
             np.interp(xg, np.linspace(0, cp, len(yp)), yp)) ** 2))

    def get_peaks(x, y, prominence=0.02):
        y_sm = smooth_peak_signal(
            y, window=peak_smooth_window, polyorder=peak_smooth_polyorder)
        dydx = np.diff(y_sm) / np.diff(x)
        pks  = find_peaks(dydx, prominence=prominence)[0]
        return x[1:], dydx, pks

    def greedy_match(xt, xp):
        matched, used = [], set()
        for t in xt:
            cands = [(j, abs(t - p)) for j, p in enumerate(xp) if j not in used]
            if not cands: continue
            j, d = min(cands, key=lambda z: z[1]); used.add(j); matched.append(d)
        return matched

    def peak_metrics(yt, yp, ct, cp, charge_len, prominence):
        out = {}
        for side, y_t, y_p, c_t, c_p in [
            ("ch",  yt[:charge_len], yp[:charge_len], ct[0], cp[0]),
            ("dis", yt[charge_len:], yp[charge_len:], ct[1], cp[1]),
        ]:
            xd,  _, pk  = get_peaks(np.linspace(0, c_t, charge_len), y_t, prominence)
            xdp, _, pkp = get_peaks(np.linspace(0, c_p, charge_len), y_p, prominence)
            m = greedy_match(xd[pk], xdp[pkp])
            out[f"count_loss_{side}"] = abs(len(pk) - len(pkp))
            out[f"pos_loss_{side}"]   = np.mean(m) if m else np.nan
        out["count_loss_mean"] = np.mean([out["count_loss_ch"], out["count_loss_dis"]])
        out["pos_loss_mean"]   = np.nanmean([out["pos_loss_ch"], out["pos_loss_dis"]])
        return out

    def cell_slope_metrics(meta_val, Ycap_np, ycap_pred, cell_key, cycle_key, min_points=2):
        df = pd.DataFrame(meta_val).copy()
        df["true_cap_ch"]  = Ycap_np[:, 0]; df["pred_cap_ch"]  = ycap_pred[:, 0]
        df["true_cap_dis"] = Ycap_np[:, 1]; df["pred_cap_dis"] = ycap_pred[:, 1]
        rows = []
        for cv, sub in df.groupby(cell_key):
            sub = sub.sort_values(cycle_key)
            if len(sub) < min_points: continue
            x = sub[cycle_key].values.astype(float); xr = x - x[0]
            if np.allclose(xr, 0): continue
            def slope(col):
                return np.polyfit(xr, sub[col].values.astype(float) - sub[col].values[0], 1)[0]
            tch, pch = slope("true_cap_ch"), slope("pred_cap_ch")
            tdi, pdi = slope("true_cap_dis"), slope("pred_cap_dis")
            rows.append({"cell": cv, "n_points": len(sub),
                         "true_slope_ch": tch, "pred_slope_ch": pch,
                         "true_slope_dis": tdi, "pred_slope_dis": pdi,
                         "slope_err_ch": pch-tch, "slope_err_dis": pdi-tdi,
                         "cycle_min": x.min(), "cycle_max": x.max()})
        slope_df = pd.DataFrame(rows)
        if len(slope_df) == 0:
            return slope_df, {k: np.nan for k in [
                "cell_slope_rmse_ch", "cell_slope_rmse_dis", "cell_slope_rmse_mean"]}
        rch  = np.sqrt(np.mean((slope_df["pred_slope_ch"]  - slope_df["true_slope_ch"])  ** 2))
        rdis = np.sqrt(np.mean((slope_df["pred_slope_dis"] - slope_df["true_slope_dis"]) ** 2))
        return slope_df, {"cell_slope_rmse_ch": rch, "cell_slope_rmse_dis": rdis,
                          "cell_slope_rmse_mean": np.mean([rch, rdis])}

    # ── prediction ──
    model.eval()
    X_main_t = X_main if torch.is_tensor(X_main) else torch.tensor(X_main, dtype=torch.float32)
    if X_aux is None:
        X_aux_t = None
    else:
        X_aux_t = X_aux if torch.is_tensor(X_aux) else torch.tensor(X_aux, dtype=torch.float32)
    yprof_pred_t, ycap_pred_t, _ = unpack_ae_outputs(
        model(X_main_t.to(device), X_aux_t.to(device) if X_aux_t is not None else None)
    )
    yprof_pred = yprof_pred_t.detach().cpu().numpy()
    ycap_pred  = ycap_pred_t.detach().cpu().numpy()
    Yprof_np   = to_np(Yprof); Ycap_np = to_np(Ycap)
    meta_df    = pd.DataFrame(meta_val).copy() if meta_val is not None else pd.DataFrame()

    # ── overall metrics ──
    profile_rmse            = np.sqrt(np.mean((Yprof_np - yprof_pred) ** 2))
    profile_rmse_per_sample = np.sqrt(np.mean((Yprof_np - yprof_pred) ** 2, axis=(1, 2)))
    cap_rmse     = np.sqrt(np.mean((Ycap_np - ycap_pred) ** 2))
    cap_rmse_ch  = np.sqrt(np.mean((Ycap_np[:,0] - ycap_pred[:,0]) ** 2))
    cap_rmse_dis = np.sqrt(np.mean((Ycap_np[:,1] - ycap_pred[:,1]) ** 2))

    cal_losses = np.array([
        cap_aligned_rmse(Yprof_np[i,0,:charge_len], Ycap_np[i,0],
                         yprof_pred[i,0,:charge_len], ycap_pred[i,0])
        for i in range(len(Yprof_np))], dtype=float)
    cap_aligned_profile_rmse_mean = np.nanmean(cal_losses)

    pclch, pplch, pcldi, ppldi, pclm, pplm = [], [], [], [], [], []
    for i in range(len(Yprof_np)):
        pm = peak_metrics(Yprof_np[i,0], yprof_pred[i,0],
                          Ycap_np[i], ycap_pred[i], charge_len, prominence)
        pclch.append(pm["count_loss_ch"]); pplch.append(pm["pos_loss_ch"])
        pcldi.append(pm["count_loss_dis"]); ppldi.append(pm["pos_loss_dis"])
        pclm.append(pm["count_loss_mean"]); pplm.append(pm["pos_loss_mean"])

    peak_count_loss_ch_mean  = np.mean(pclch)
    peak_count_loss_dis_mean = np.mean(pcldi)
    peak_count_loss_mean     = np.mean(pclm)
    peak_pos_loss_ch_mean    = np.nanmean(pplch)
    peak_pos_loss_dis_mean   = np.nanmean(ppldi)
    peak_pos_loss_mean       = np.nanmean(pplm)

    # ── per-sample / per-cell ──
    if meta_val is not None:
        per_sample_df = pd.DataFrame(meta_val).copy()
        per_sample_df["profile_rmse"] = profile_rmse_per_sample
        per_sample_df["cap_rmse"]     = np.sqrt(np.mean((Ycap_np - ycap_pred) ** 2, axis=1))
        per_sample_df["cap_rmse_ch"]  = np.abs(Ycap_np[:,0] - ycap_pred[:,0])
        per_sample_df["cap_rmse_dis"] = np.abs(Ycap_np[:,1] - ycap_pred[:,1])
        per_sample_df["cap_aligned_profile_rmse"] = cal_losses
        per_sample_df["peak_count_loss_ch"]       = np.array(pclch, dtype=float)
        per_sample_df["peak_pos_loss_ch"]         = np.array(pplch, dtype=float)
        per_sample_df["peak_count_loss_dis"]      = np.array(pcldi, dtype=float)
        per_sample_df["peak_pos_loss_dis"]        = np.array(ppldi, dtype=float)
        per_sample_df["peak_count_loss_mean"]     = np.array(pclm,  dtype=float)
        per_sample_df["peak_pos_loss_mean"]       = np.array(pplm,  dtype=float)

        per_cell_df = (per_sample_df.groupby(trend_cell_key, dropna=False).agg(
            n_samples=("profile_rmse","size"),
            profile_rmse_mean=("profile_rmse","mean"),
            cap_rmse_mean=("cap_rmse","mean"),
            cap_rmse_ch_mean=("cap_rmse_ch","mean"),
            cap_rmse_dis_mean=("cap_rmse_dis","mean"),
            cap_aligned_profile_rmse_mean=("cap_aligned_profile_rmse","mean"),
            peak_count_loss_ch_mean=("peak_count_loss_ch","mean"),
            peak_pos_loss_ch_mean=("peak_pos_loss_ch","mean"),
            peak_count_loss_dis_mean=("peak_count_loss_dis","mean"),
            peak_pos_loss_dis_mean=("peak_pos_loss_dis","mean"),
            peak_count_loss_mean=("peak_count_loss_mean","mean"),
            peak_pos_loss_mean=("peak_pos_loss_mean","mean"),
        ).reset_index())
    else:
        per_sample_df = pd.DataFrame({"profile_rmse": profile_rmse_per_sample})
        per_cell_df   = pd.DataFrame()

    best_cell = worst_cell = best_idx = worst_idx = None
    if len(per_cell_df) > 0:
        per_cell_df = per_cell_df.sort_values("profile_rmse_mean").reset_index(drop=True)
        best_cell   = per_cell_df.iloc[0][trend_cell_key]
        worst_cell  = per_cell_df.iloc[-1][trend_cell_key]
        best_idx  = per_sample_df[per_sample_df[trend_cell_key] == best_cell].sort_values("profile_rmse").index[0]
        worst_idx = per_sample_df[per_sample_df[trend_cell_key] == worst_cell].sort_values("profile_rmse").index[0]

    slope_df = None
    slope_summary = {k: np.nan for k in [
        "cell_slope_rmse_ch", "cell_slope_rmse_dis", "cell_slope_rmse_mean"]}
    if meta_val is not None:
        try:
            slope_df, slope_summary = cell_slope_metrics(
                meta_val, Ycap_np, ycap_pred, trend_cell_key, trend_cycle_key)
        except Exception as e:
            print(f"[WARN] cell slope metric skipped: {e}")

    if slope_df is not None and len(slope_df) > 0 and len(per_cell_df) > 0:
        per_cell_df = per_cell_df.merge(
            slope_df.rename(columns={"cell": trend_cell_key}),
            on=trend_cell_key, how="left")

    # ── plots ──
    def plot_profile_by_index(idx, title_prefix, filename):
        fig = plt.figure(figsize=(8, 4))
        plt.plot(Yprof_np[idx,0], label="GT")
        plt.plot(yprof_pred[idx,0], "--", label="Pred")
        plt.title(f"{title_prefix}\n"
                  f"{trend_cell_key}={meta_df.iloc[idx][trend_cell_key]}, "
                  f"RMSE={per_sample_df.iloc[idx]['profile_rmse']:.6f}")
        plt.xlabel("Index"); plt.ylabel("Voltage"); plt.grid(True); plt.legend()
        save_fig(fig, filename)

    def plot_cap_aligned_by_index(idx, title_prefix, filename):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, Ycap_np[idx,0], charge_len),
                 Yprof_np[idx,0,:charge_len], marker="o", markersize=3, label="GT")
        plt.plot(np.linspace(0, ycap_pred[idx,0], charge_len),
                 yprof_pred[idx,0,:charge_len], marker="o", markersize=3,
                 linestyle="--", label="Pred")
        plt.xlabel("Capacity"); plt.ylabel("Voltage"); plt.grid(True); plt.legend()
        plt.title(f"{title_prefix}\n"
                  f"cap-aligned RMSE={per_sample_df.iloc[idx]['cap_aligned_profile_rmse']:.6f}")
        save_fig(fig, filename)

    def plot_peak_by_index(idx, title_prefix, filename):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, side, yt, yp, ct, cp, ylim in [
            (axes[0], "Charge",    Yprof_np[idx,0,:charge_len],  yprof_pred[idx,0,:charge_len],
             Ycap_np[idx,0], ycap_pred[idx,0], charge_peak_ylim),
            (axes[1], "Discharge", Yprof_np[idx,0,charge_len:],  yprof_pred[idx,0,charge_len:],
             Ycap_np[idx,1], ycap_pred[idx,1], discharge_peak_ylim),
        ]:
            x = np.linspace(0, ct, charge_len); xp_arr = np.linspace(0, cp, charge_len)
            xd, dd, pk   = get_peaks(x,      yt, prominence)
            xdp,ddp,pkp  = get_peaks(xp_arr, yp, prominence)
            ax.plot(xd, dd, label="GT dV/dQ")
            ax.scatter(xd[pk], dd[pk], color="red", label="GT peaks")
            ax.plot(xdp, ddp, "--", label="Pred dV/dQ")
            ax.scatter(xdp[pkp], ddp[pkp], color="blue", alpha=0.6, label="Pred peaks")
            ax.set_title(side); ax.set_ylim(*ylim); ax.grid(True); ax.legend()
        fig.suptitle(f"{title_prefix}")
        plt.tight_layout(); save_fig(fig, filename)

    def plot_slope_scatter(slope_df, filename="cell_slope_scatter.png"):
        if slope_df is None or len(slope_df) == 0: return
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, tc, pc, title in [
            (axes[0], "true_slope_ch",  "pred_slope_ch",  "Charge"),
            (axes[1], "true_slope_dis", "pred_slope_dis", "Discharge"),
        ]:
            x, y = slope_df[tc].values, slope_df[pc].values
            mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
            ax.scatter(x, y, alpha=0.7); ax.plot([mn,mx],[mn,mx],"k--")
            ax.set_title(f"Cell slope ({title})")
            ax.set_xlabel("True slope"); ax.set_ylabel("Pred slope"); ax.grid(True)
        plt.tight_layout(); save_fig(fig, filename)

    if plot_best_worst_cell_profile and best_idx is not None:
        plot_profile_by_index(best_idx,  "Best cell",  "best_cell_profile.png")
        plot_profile_by_index(worst_idx, "Worst cell", "worst_cell_profile.png")
    if plot_cap_scatter:
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(Ycap_np[:,0], ycap_pred[:,0], alpha=0.5, label="Charge")
        plt.scatter(Ycap_np[:,1], ycap_pred[:,1], alpha=0.5, label="Discharge")
        mn = min(Ycap_np.min(), ycap_pred.min()); mx = max(Ycap_np.max(), ycap_pred.max())
        plt.plot([mn,mx],[mn,mx],"k--"); plt.xlabel("True"); plt.ylabel("Pred")
        plt.title("Capacity prediction"); plt.grid(True); plt.legend()
        save_fig(fig, "capacity_scatter.png")
    if plot_best_worst_cap_aligned_profile and best_idx is not None:
        plot_cap_aligned_by_index(best_idx,  "Best cell cap-aligned",  "best_cap_aligned.png")
        plot_cap_aligned_by_index(worst_idx, "Worst cell cap-aligned", "worst_cap_aligned.png")
    if plot_best_worst_peak and best_idx is not None:
        plot_peak_by_index(best_idx,  "Best cell peak",  "best_cell_peak.png")
        plot_peak_by_index(worst_idx, "Worst cell peak", "worst_cell_peak.png")
    if plot_cell_slope_scatter and slope_df is not None:
        try: plot_slope_scatter(slope_df)
        except Exception as e: print(f"[WARN] slope scatter skipped: {e}")

    # ── save csv ──
    summary_row = {
        "profile_rmse": profile_rmse, "cap_rmse": cap_rmse,
        "cap_rmse_ch": cap_rmse_ch, "cap_rmse_dis": cap_rmse_dis,
        "cap_aligned_profile_rmse_mean": cap_aligned_profile_rmse_mean,
        "peak_count_loss_ch_mean":  peak_count_loss_ch_mean,
        "peak_count_loss_dis_mean": peak_count_loss_dis_mean,
        "peak_count_loss_mean":     peak_count_loss_mean,
        "peak_pos_loss_ch_mean":    peak_pos_loss_ch_mean,
        "peak_pos_loss_dis_mean":   peak_pos_loss_dis_mean,
        "peak_pos_loss_mean":       peak_pos_loss_mean,
        "peak_smooth_window":       peak_smooth_window,
        "peak_smooth_polyorder":    peak_smooth_polyorder,
        "cell_slope_rmse_ch":   slope_summary["cell_slope_rmse_ch"],
        "cell_slope_rmse_dis":  slope_summary["cell_slope_rmse_dis"],
        "cell_slope_rmse_mean": slope_summary["cell_slope_rmse_mean"],
        "best_cell": best_cell, "worst_cell": worst_cell,
    }
    summary_df = pd.DataFrame([summary_row])
    if save_dir is not None:
        summary_df.to_csv(save_dir / result_csv_name, index=False)
        if len(per_cell_df)   > 0: per_cell_df.to_csv(save_dir / cell_csv_name, index=False)
        if len(per_sample_df) > 0: per_sample_df.to_csv(save_dir / sample_csv_name, index=False)

    print(f"Profile RMSE                : {profile_rmse:.6f}")
    print(f"Capacity RMSE (overall)     : {cap_rmse:.6f}")
    print(f"Capacity RMSE (charge)      : {cap_rmse_ch:.6f}")
    print(f"Capacity RMSE (discharge)   : {cap_rmse_dis:.6f}")
    print(f"Cap-aligned Profile RMSE    : {cap_aligned_profile_rmse_mean:.6f}")
    print(f"Peak count loss (charge)    : {peak_count_loss_ch_mean:.6f}")
    print(f"Peak count loss (discharge) : {peak_count_loss_dis_mean:.6f}")
    print(f"Peak count loss (mean)      : {peak_count_loss_mean:.6f}")
    print(f"Peak pos loss (charge)      : {peak_pos_loss_ch_mean:.6f}")
    print(f"Peak pos loss (discharge)   : {peak_pos_loss_dis_mean:.6f}")
    print(f"Peak pos loss (mean)        : {peak_pos_loss_mean:.6f}")
    print(f"Peak smoothing              : Savitzky-Golay window={peak_smooth_window}, polyorder={peak_smooth_polyorder}")
    print(f"Cell slope RMSE (charge)    : {slope_summary['cell_slope_rmse_ch']:.6f}")
    print(f"Cell slope RMSE (discharge) : {slope_summary['cell_slope_rmse_dis']:.6f}")
    print(f"Cell slope RMSE (mean)      : {slope_summary['cell_slope_rmse_mean']:.6f}")
    print(f"Best cell by RMSE           : {best_cell}")
    print(f"Worst cell by RMSE          : {worst_cell}")
    if save_dir is not None: print(f"Saved results to            : {save_dir}")

    return {
        "yprof_pred": yprof_pred, "ycap_pred": ycap_pred,
        "profile_rmse": profile_rmse, "profile_rmse_per_sample": profile_rmse_per_sample,
        "cap_rmse": cap_rmse, "cap_rmse_ch": cap_rmse_ch, "cap_rmse_dis": cap_rmse_dis,
        "cap_aligned_profile_rmse_mean": cap_aligned_profile_rmse_mean,
        "cap_aligned_profile_rmse_per_sample": cal_losses,
        "peak_count_loss_ch_mean": peak_count_loss_ch_mean,
        "peak_count_loss_dis_mean": peak_count_loss_dis_mean,
        "peak_count_loss_mean": peak_count_loss_mean,
        "peak_pos_loss_ch_mean": peak_pos_loss_ch_mean,
        "peak_pos_loss_dis_mean": peak_pos_loss_dis_mean,
        "peak_pos_loss_mean": peak_pos_loss_mean,
        "peak_count_losses_ch": np.array(pclch, dtype=float),
        "peak_count_losses_dis": np.array(pcldi, dtype=float),
        "peak_pos_losses_ch": np.array(pplch, dtype=float),
        "peak_pos_losses_dis": np.array(ppldi, dtype=float),
        "cell_slope_df": slope_df,
        "cell_slope_rmse_ch":   slope_summary["cell_slope_rmse_ch"],
        "cell_slope_rmse_dis":  slope_summary["cell_slope_rmse_dis"],
        "cell_slope_rmse_mean": slope_summary["cell_slope_rmse_mean"],
        "per_sample_df": per_sample_df, "per_cell_df": per_cell_df,
        "summary_df": summary_df,
        "best_cell": best_cell, "worst_cell": worst_cell,
        "best_idx": best_idx, "worst_idx": worst_idx,
    }


# =========================================================
# Training utilities (shared across all notebooks)
# =========================================================
PROFILE_RMSE_SCALE = 100.0
OBJECTIVE_WEIGHT_PEAK_COUNT = 1.0
OBJECTIVE_WEIGHT_PEAK_POS = 10.0
OBJECTIVE_WEIGHT_CAP = 5.0
OBJECTIVE_WEIGHT_PROFILE = 30.0


class EarlyStopping:
    def __init__(self, patience: int = 50, min_delta: float = 1e-6):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def step(self, score: float, epoch: int) -> bool:
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter    = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Run one epoch. Pass an optimizer to train, or `None` to validate.

    Returns (total_loss, profile_rmse) averaged over the loader.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss      = 0.0
    total_prof_rmse = 0.0
    n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            x_main, x_aux, y_prof, y_cap, _ = move_batch_to_device(batch, device)
            bs = x_main.size(0)

            y_prof_pred, y_cap_pred, _ = unpack_ae_outputs(model(x_main, x_aux))
            loss_dict = loss_fn(y_prof_pred, y_prof, y_cap_pred, y_cap)
            loss = loss_dict["loss_total"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss      += loss.item() * bs
            total_prof_rmse += torch.sqrt(
                ((y_prof_pred - y_prof) ** 2).mean()
            ).item() * bs
            n += bs

    return total_loss / max(n, 1), total_prof_rmse / max(n, 1)


@torch.no_grad()
def compute_objective_metric(
    model: nn.Module,
    X_main: torch.Tensor,
    X_aux: Optional[torch.Tensor],
    Y_prof: torch.Tensor,
    Y_cap: torch.Tensor,
    device: torch.device,
    charge_len: int = 64,
    prominence: float = 0.02,
    peak_smooth_window: int = 7,
    peak_smooth_polyorder: int = 2,
    batch_size: int = 128,
) -> float:
    """Optuna objective = M_cnt + 10*M_pos + 5*M_cap + 30*M_profile."""
    model.eval()
    all_pred_prof = []
    all_pred_cap = []
    N = X_main.size(0)

    for i in range(0, N, batch_size):
        xm = X_main[i:i+batch_size].to(device)
        xa = X_aux[i:i+batch_size].to(device) if X_aux is not None else None
        pred, pred_cap, _ = unpack_ae_outputs(model(xm, xa))
        all_pred_prof.append(pred.squeeze(1).cpu().numpy())
        all_pred_cap.append(pred_cap.cpu().numpy())

    pred_np   = np.concatenate(all_pred_prof, axis=0)
    pred_cap_np = np.concatenate(all_pred_cap, axis=0)
    target_np = Y_prof.detach().cpu().squeeze(1).numpy() if torch.is_tensor(Y_prof) else np.asarray(Y_prof).squeeze(1)
    cap_np = Y_cap.detach().cpu().numpy() if torch.is_tensor(Y_cap) else np.asarray(Y_cap)

    profile_rmse = float(np.sqrt(np.mean((pred_np - target_np) ** 2)))
    cap_rmse = float(np.sqrt(np.mean((pred_cap_np - cap_np) ** 2)))

    def greedy_match(xt, xp):
        matched, used = [], set()
        for t in xt:
            cands = [(j, abs(t - p)) for j, p in enumerate(xp) if j not in used]
            if not cands:
                continue
            j, d = min(cands, key=lambda z: z[1])
            used.add(j)
            matched.append(d)
        return matched

    count_losses = []
    pos_losses = []
    for i in range(N):
        for side, sl, cap_idx in [
            ("ch",  slice(0, charge_len),   0),
            ("dis", slice(charge_len, None), 1),
        ]:
            cap_gt = max(float(cap_np[i, cap_idx]), 1e-6)
            cap_pr = max(float(pred_cap_np[i, cap_idx]), 1e-6)
            x_gt = np.linspace(0, cap_gt, charge_len)
            x_pr = np.linspace(0, cap_pr, charge_len)
            y_gt = smooth_peak_signal(
                target_np[i, sl],
                window=peak_smooth_window,
                polyorder=peak_smooth_polyorder,
            )
            y_pr = smooth_peak_signal(
                pred_np[i, sl],
                window=peak_smooth_window,
                polyorder=peak_smooth_polyorder,
            )
            dydx_gt = np.diff(y_gt) / np.diff(x_gt)
            pk_gt   = find_peaks(dydx_gt, prominence=prominence)[0]
            dydx_pr = np.diff(y_pr) / np.diff(x_pr)
            pk_pr   = find_peaks(dydx_pr, prominence=prominence)[0]
            count_losses.append(abs(len(pk_gt) - len(pk_pr)))
            matches = greedy_match(x_gt[1:][pk_gt], x_pr[1:][pk_pr])
            pos_losses.append(np.mean(matches) if matches else np.nan)

    m_cnt = float(np.mean(count_losses))
    m_pos = float(np.nanmean(pos_losses)) if np.any(~np.isnan(pos_losses)) else 0.0
    m_cap = cap_rmse
    m_profile = profile_rmse
    return (
        OBJECTIVE_WEIGHT_PEAK_COUNT * m_cnt
        + OBJECTIVE_WEIGHT_PEAK_POS * m_pos
        + OBJECTIVE_WEIGHT_CAP * m_cap
        + OBJECTIVE_WEIGHT_PROFILE * m_profile
    )


def load_best_window_from_ablation(
    window_ablation_dir: Path = Path('window_ablation_results'),
) -> int:
    csv_path = window_ablation_dir / 'window_ablation_summary.csv'
    if not csv_path.exists():
        print(f'[WARNING] window_ablation_summary.csv not found. Using W=64.')
        return 64
    df = pd.read_csv(csv_path)
    if 'test_objective' not in df.columns:
        print('[WARNING] test_objective column not found. Using W=64.')
        return 64
    best_row = df.loc[df['test_objective'].idxmin()]
    best_w   = int(best_row['window_size'])
    print(f'[INFO] Best window size: W={best_w} '
          f'(test_objective={best_row["test_objective"]:.4f})')
    return best_w


def load_best_seg_diff_from_loss_ablation(
    loss_ablation_dir: Path = Path('loss_ablation_results_v2'),
) -> dict:
    csv_path = loss_ablation_dir / 'loss_ablation_summary.csv'
    defaults = {'seg_bias': True, 'diff_loss': True}
    if not csv_path.exists():
        print('[WARNING] loss_ablation_summary.csv not found. Using defaults.')
        return defaults
    df   = pd.read_csv(csv_path)
    if 'test_objective' not in df.columns:
        return defaults
    best = df.loc[df['test_objective'].idxmin()]
    result = {
        'seg_bias':  bool(best.get('seg_bias',  True)),
        'diff_loss': bool(best.get('diff_loss', True)),
    }
    print(f'[INFO] Best loss config: ablation={best.get("ablation","?")} '
          f'| seg={result["seg_bias"]} | diff={result["diff_loss"]}')
    return result
