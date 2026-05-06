# BatteryOCV: A Paired Dataset and Benchmark for High-to-Low Rate Profile Reconstruction

Code accompanying the NeurIPS 2026 Evaluations & Datasets Track submission **"BatteryOCV: A Paired Dataset and Benchmark for High-to-Low Rate Profile Reconstruction"**.

## Overview

BatteryOCV is a paired dataset and benchmark for reconstructing low-rate (0.05C) reference voltage profiles from high-rate (0.33–0.5C) observations in lithium-ion batteries. The benchmark formulates high-to-low rate profile reconstruction as a window-based conditional regression problem with capacity prediction as a primary output, and proposes DVA-aware evaluation metrics (peak count, peak position) that complement profile RMSE.

## Dataset

The dataset will be released on Harvard Dataverse under a CC BY 4.0 license upon acceptance.

For double-blind review, an anonymous preview URL is provided in the OpenReview submission alongside this code repository. Reviewers can download `BatteryOCV.h5`, `BatteryOCV_split.json`, and `BatteryOCV_croissant.json` from that URL.

The dataset comprises:
- **404 industrial NCM/graphite cells**
- **1,894 operator-verified low-rate (0.05C) reference profiles**
- **182,519 high-rate cycle records** (0.33–0.5C, 25–45 °C chamber temperatures)

Files included in the release:
- `BatteryOCV.h5` (~600 MB) — main HDF5 file with `highrate_cycle` and `lowrate_cycle` datasets
- `BatteryOCV_split.json` (5 KB) — cell-disjoint splits (Model E in-distribution 70/10/20, Models A–D held out)
- `BatteryOCV_croissant.json` (16 KB) — Croissant metadata (Core + Responsible AI fields)

The HDF5 column layout is documented in Appendix A.2 (Table 7) of the paper and in the `COL` dictionary at the top of `pipeline.py`.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE                          # MIT
│
├── pipeline.py                      # Data loading, training/eval utilities,
│                                    # AE loss, evaluation pipeline.
├── models.py                        # Discriminative architectures (Conv, MLP,
│                                    # LSTM, BiLSTM, Transformer, UNet, Conv+LSTM).
├── generative_models.py             # Generative architectures (Conv VAE, LSTM
│                                    # VAE, Transformer VAE, Conv GAN).
│
├── 00_inspect_data.py               # Sanity check the HDF5 file and splits
├── 01_window_ablation.py            # Stage 1: window size ablation
├── 02_design_ablation.py            # Stage 2: design components (Seg / Diff)
├── 03_input_ablation.py             # Stage 3: input representation ablation
├── 04_architecture_comparison.py    # Stage 4: 7 discriminative architectures
└── 05_generative_baselines.py       # Stage 5: VAE / GAN baselines
```

The `0X_*.py` scripts implement the experiments reported in §5 and §B of the paper, in the order they appear (window → design → input → architecture comparison → generative baselines).

## Reproducing the paper results

| Paper section | Result | Script |
|---|---|---|
| —             | Dataset / split sanity check | `python 00_inspect_data.py` |
| §5.2 / Table 3 | Window size ablation (W ∈ {1, 16, 32, 64}, Conv AE, voltage-only) | `python 01_window_ablation.py` |
| §5.2 / Table 4 | Design components ablation (Base, Seg, Diff, Seg+Diff at W=64) | `python 02_design_ablation.py` |
| §B.3 / Table 8 | Input representation ablation (V, V+T, V+I, V+Q in scalar / channel modes) | `python 03_input_ablation.py` |
| §5.4 / Table 5 | Architecture comparison (7 discriminative architectures, Optuna 20 trials each) | `python 04_architecture_comparison.py` |
| §5.5 / Table 9 | Generative baselines (Conv/LSTM/Transformer VAE, Conv GAN, Optuna 20 trials each) | `python 05_generative_baselines.py` |
| §5.6 / Table 6 / Figure 6 | Cross-model zero-shot evaluation on Models A–D | Produced inside `04_*` and `05_*` via `build_crossmodel_bundle(...)` |

All scripts use seed 42 and the fixed cell-disjoint split distributed in `BatteryOCV_split.json`. Each training script writes per-epoch logs, best checkpoints, evaluation summaries, and aggregated CSVs into its own `stage*` output directory.

### Note on multi-seed results

Tables 3, 4, and 8 in the paper additionally report mean ± std over seeds 42, 0, 1. Those multi-seed runs were performed during development with seed-dependent cell-disjoint splits derived from the original (now anonymized) cell identifiers. Because the public release uses a single fixed split (`BatteryOCV_split.json`) for clarity and reproducibility, the released code reproduces only the seed-42 row of these tables; the seed-42 numbers match the paper exactly. We provide the fixed split because it is what reviewers and downstream users need to compare new methods against the same cells we report on.

## Installation

```bash
git clone <this-repo>
cd batteryocv-benchmark
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.0+. Experiments in the paper were run on a single NVIDIA RTX 3070 Ti (8 GB VRAM); the full pipeline takes approximately 15 GPU hours.

## Quick start

1. **Download the dataset.** Place `BatteryOCV.h5` and `BatteryOCV_split.json` into the repository root. During review, both files can be downloaded from the anonymous preview URL provided in the OpenReview submission. (Or pass `h5_path=` and `split_json_path=` to the loaders to point elsewhere.)

2. **Sanity-check the data.** This prints split sizes, profile statistics, and writes a few example profiles to `inspect_data.png`:
   ```bash
   python 00_inspect_data.py
   ```
   The split sizes should print as 265 / 39 / 76 cells (Model E train/val/test) and 24 cells (Models A–D cross-model).

3. **Run a single ablation.** For example, the window-size ablation:
   ```bash
   python 01_window_ablation.py
   ```

4. **Inspect results.** Each script produces a summary CSV in its `stage*` directory (e.g., `stage1_window_ablation/window_ablation_summary.csv`).

The loaders read `BatteryOCV.h5` and `BatteryOCV_split.json` from the working directory by default; no additional configuration is required.

## Reproducibility notes

- **Seed.** All released scripts run with seed 42. Multi-seed averaging is not provided in the released code (see "Note on multi-seed results" above).
- **Optimizer.** AdamW with `lr=1e-3`, `betas=(0.9, 0.95)`, `weight_decay=1e-2`, up to 3000 epochs, with checkpoint selection by validation profile RMSE.
- **Optuna.** TPE sampler with median pruner, 20 trials per architecture, composite objective `O = M_cnt + 10·M_pos + 5·M_cap + 30·M_profile` (Eq. 7 in the paper).
- **Evaluation.** dV/dQ peaks are detected on Savitzky–Golay smoothed profiles (window 7, polyorder 2, prominence 0.02); see Appendix B.1 for full metric definitions.
- **Data splits.** Cell-disjoint, fixed seed 42, distributed in `BatteryOCV_split.json`. Model E (in-distribution): 265/39/76 cells (train/val/test); Models A–D (24 cells total) held out for cross-model evaluation.

## Adding a new architecture

To benchmark a new discriminative model on this task, add a class with the following interface to `models.py`:

```python
class YourModel(nn.Module):
    def forward(self, x_main, x_aux):
        # x_main : [B, W, 128]   high-rate window (W consecutive cycles)
        # x_aux  : [B, W, aux_dim] auxiliary scalar features (optional)
        ...
        return y_prof, y_cap, z
        # y_prof : [B, 1, 128]   reconstructed low-rate profile
        # y_cap  : [B, 2]        (charge capacity, discharge capacity)
        # z      : [B, latent_dim] latent representation
```

Then register it in `MODEL_REGISTRY` in `models.py` and add a hyperparameter block in `04_architecture_comparison.py`. The training loop, loss, and evaluation pipeline are reused unchanged.

## Troubleshooting

**Windows: training terminates silently after one epoch with `OMP: Error #15`**

If you see a message like `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`, this is a known OpenMP runtime conflict between PyTorch and other scientific libraries (commonly triggered when `numpy`/`scipy` are mixed from `conda` and `pip`). Set the environment variable before running:

```bat
set KMP_DUPLICATE_LIB_OK=TRUE      :: Windows cmd
$env:KMP_DUPLICATE_LIB_OK="TRUE"   # Windows PowerShell
export KMP_DUPLICATE_LIB_OK=TRUE   # Linux / macOS
```

This is a widely used workaround and does not affect numerical results for the PyTorch + NumPy combination used here. If you prefer a permanent fix, reinstall `numpy`, `scipy`, and `mkl` from a single package source (all `conda` or all `pip`).

## License

- **Code**: MIT License (see `LICENSE`)
- **Dataset**: CC BY 4.0 (release pending acceptance; anonymous preview URL provided in OpenReview)

