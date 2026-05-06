"""
Quick sanity inspection of BatteryOCV.h5 and BatteryOCV_split.json.

Run this before any training script to verify the dataset is in the expected
shape and that the cell-disjoint splits match the numbers reported in the
paper. Prints a summary, draws a few example profiles, and writes the
figure to inspect_data.png.

Usage
-----
    python 00_inspect_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


H5_PATH    = Path("BatteryOCV.h5")
SPLIT_PATH = Path("BatteryOCV_split.json")
FIG_PATH   = Path("inspect_data.png")

# Column layout — must match COL in pipeline.py
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


def header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def main():
    if not H5_PATH.exists():
        raise FileNotFoundError(
            f"{H5_PATH} not found. Download BatteryOCV.h5 from Harvard Dataverse "
            f"(DOI: 10.7910/DVN/ZK8PGV) and place it in the working directory."
        )
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH} not found. Download from the same Dataverse entry."
        )

    # ------------------------------------------------------------------ h5
    header("1. HDF5 file structure")
    with h5py.File(str(H5_PATH), "r") as f:
        print(f"  File             : {H5_PATH}  ({H5_PATH.stat().st_size / 1e6:.1f} MB)")
        print(f"  Datasets         : {list(f.keys())}")
        for name in ["highrate_cycle", "lowrate_cycle"]:
            if name in f:
                ds = f[name]
                print(f"  {name:<16s}: shape={ds.shape}, dtype={ds.dtype}")
        attrs = dict(f.attrs)
        if attrs:
            print(f"  File attrs       :")
            for k, v in attrs.items():
                v_str = str(v)
                if len(v_str) > 80:
                    v_str = v_str[:77] + "..."
                print(f"    {k}: {v_str}")
        hr = np.asarray(f["highrate_cycle"][:], dtype=np.float32)
        lr = np.asarray(f["lowrate_cycle"][:], dtype=np.float32)

    if hr.shape[1] != 775 or lr.shape[1] != 775:
        raise ValueError(
            f"Unexpected column count: highrate {hr.shape}, lowrate {lr.shape}. "
            f"Each row should have 775 columns; see COL definition above."
        )

    # ------------------------------------------------------------------ split
    header("2. Cell-disjoint splits (BatteryOCV_split.json)")
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    print(f"  description : {split.get('description', '(none)')}")
    print(f"  random_state: {split.get('random_state')}")
    print(f"  model_map   : {split.get('model_map')}")

    splits = {
        "Train (Model E)":    split["in_distribution"]["train"],
        "Val   (Model E)":    split["in_distribution"]["val"],
        "Test  (Model E)":    split["in_distribution"]["test"],
        "Cross (Models A-D)": split["cross_model"]["test"],
    }
    print()
    print(f"  {'Split':<22s} {'#cells':>7s}  {'#highrate':>10s}  {'#lowrate':>9s}")
    cell_col_hr = hr[:, COL["cell"]].astype(int)
    cell_col_lr = lr[:, COL["cell"]].astype(int)
    for name, cells in splits.items():
        cell_set = set(cells)
        n_hr = int(np.isin(cell_col_hr, list(cell_set)).sum())
        n_lr = int(np.isin(cell_col_lr, list(cell_set)).sum())
        print(f"  {name:<22s} {len(cells):>7d}  {n_hr:>10d}  {n_lr:>9d}")

    # cell-disjointness check
    all_id = (set(splits["Train (Model E)"]) | set(splits["Val   (Model E)"])
              | set(splits["Test  (Model E)"]))
    cross  = set(splits["Cross (Models A-D)"])
    overlap = all_id & cross
    print()
    print(f"  Cell-disjoint check (Model E vs A-D): "
          f"{'PASS' if not overlap else f'FAIL — overlap on cells {sorted(overlap)}'}")

    # ------------------------------------------------------------------ stats
    header("3. Profile sanity statistics")
    print(f"  highrate cycle counts per cell (Model E train), "
          f"first 5 cells:")
    train_cells = splits["Train (Model E)"][:5]
    for c in train_cells:
        n = int((cell_col_hr == c).sum())
        m = int((cell_col_lr == c).sum())
        print(f"    cell_idx={c:4d} : highrate={n:5d}, lowrate={m:3d}")

    nan_hr = int(np.isnan(hr).sum())
    nan_lr = int(np.isnan(lr).sum())
    print(f"\n  NaN entries: highrate={nan_hr}, lowrate={nan_lr}")

    v_min_hr = float(hr[:, COL["charge_voltage_profile"]].min())
    v_max_hr = float(hr[:, COL["charge_voltage_profile"]].max())
    v_min_lr = float(lr[:, COL["charge_voltage_profile"]].min())
    v_max_lr = float(lr[:, COL["charge_voltage_profile"]].max())
    print(f"  charge_voltage_profile range:")
    print(f"    highrate: [{v_min_hr:.4f}, {v_max_hr:.4f}]")
    print(f"    lowrate : [{v_min_lr:.4f}, {v_max_lr:.4f}]")
    print(f"    Note: voltages are min-max scaled to the operating window")
    print(f"    [2.5, 4.4] V. Recorded values may slightly exceed [0, 1] at")
    print(f"    the boundaries due to instrument sampling near cutoffs.")

    cap_hr = hr[:, COL["charge_capacity"]]
    cap_lr = lr[:, COL["charge_capacity"]]
    print(f"\n  charge_capacity (normalized by nominal):")
    print(f"    highrate: mean={cap_hr.mean():.4f}, [{cap_hr.min():.4f}, {cap_hr.max():.4f}]")
    print(f"    lowrate : mean={cap_lr.mean():.4f}, [{cap_lr.min():.4f}, {cap_lr.max():.4f}]")
    print(f"    Note: nominal capacity is a design value, so healthy cells")
    print(f"    may report measured capacity above 1.0 (typically up to ~1.15).")

    # ------------------------------------------------------------------ plot
    header("4. Example profiles → inspect_data.png")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Pick a random training cell and plot its low-rate profiles
    rng = np.random.default_rng(42)
    cell = int(rng.choice(splits["Train (Model E)"]))
    lr_cell = lr[cell_col_lr == cell]
    hr_cell = hr[cell_col_hr == cell]

    ax = axes[0, 0]
    for row in lr_cell:
        ax.plot(row[COL["charge_voltage_profile"]], alpha=0.6)
    ax.set_title(f"Low-rate charge V profiles, cell_idx={cell} (n={len(lr_cell)})")
    ax.set_xlabel("Sample index (capacity-uniform, 128 pts)")
    ax.set_ylabel("Voltage (normalized)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for row in lr_cell:
        ax.plot(row[COL["discharge_voltage_profile"]], alpha=0.6)
    ax.set_title(f"Low-rate discharge V profiles, cell_idx={cell}")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Voltage (normalized)")
    ax.grid(True, alpha=0.3)

    # Plot a sparse subset of high-rate profiles (every Nth) to show degradation trend
    ax = axes[1, 0]
    n_plot = min(50, len(hr_cell))
    idx = np.linspace(0, len(hr_cell) - 1, n_plot, dtype=int)
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(idx):
        ax.plot(hr_cell[k, COL["charge_voltage_profile"]],
                color=cmap(i / max(n_plot - 1, 1)), alpha=0.6, linewidth=0.7)
    ax.set_title(f"High-rate charge V (every {len(hr_cell) // n_plot}-th cycle, "
                 f"colors = early→late)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Voltage (normalized)")
    ax.grid(True, alpha=0.3)

    # Capacity trajectory across cycles for that cell
    ax = axes[1, 1]
    cycles = hr_cell[:, COL["cycle"]]
    caps   = hr_cell[:, COL["charge_capacity"]]
    order = np.argsort(cycles)
    ax.plot(cycles[order], caps[order], ".", markersize=2, alpha=0.5,
            label="high-rate")
    cycles_lr = lr_cell[:, COL["cycle"]]
    caps_lr   = lr_cell[:, COL["charge_capacity"]]
    ax.plot(cycles_lr, caps_lr, "rx", markersize=8, label="low-rate (RPT)")
    ax.set_title(f"Capacity trajectory, cell_idx={cell}")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Charge capacity (normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=120, bbox_inches="tight")
    print(f"  Saved figure: {FIG_PATH}")

    print()
    print("All checks done. If the cell counts above match the paper "
          "(265 / 39 / 76 / 24) and capacity ranges look physical, the "
          "dataset is ready to use. Run any of the 0X_*.py scripts next.")


if __name__ == "__main__":
    main()
