# %%
"""
Input representation ablation (Stage 3).

Reproduces Table 8 (paper §B.3): single and combined auxiliary inputs
{V, V+T, V+I, V+Q, ...} in scalar (MLP-fused) and channel-stacked
modes at the fixed configuration W=64, Seg+Diff, seed 42.

Outputs
-------
stage3_input_ablation/
    Per-input-variant directories with checkpoints, evaluation summaries,
    and an aggregate `input_ablation_summary.csv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import shutil
import time

import numpy as np
import pandas as pd
import torch

from pipeline import (
    RANDOM_SEED, USE_HALF_PROFILE, BATCH_SIZE, NUM_WORKERS,
    DB_ROOT_PUBLIC_H5, SPLIT_JSON_PATH,
    set_seed, ensure_dir, build_dataloaders,
    AELossConfig, AELoss, train_one_epoch, validate_one_epoch,
    save_ckpt_ae, load_ckpt_ae, evaluate_ae_predictions_aux,
    plot_loss_components, compute_objective_metric, PROFILE_RMSE_SCALE,
)
from models import build_model, count_parameters

# =========================================================
# STAGE 3: Unified input ablation after best window/loss
# Aux and channel inputs use the same build_dataloaders + build_model path.
# =========================================================
WINDOW_SIZE = 64
USE_SEG_BIAS = True
USE_DIFF_LOSS = True

LAMBDA_P = 100.0
LAMBDA_D = 100.0
LAMBDA_C = 5.0

LATENT_DIM = 128
AUX_EMBED_DIM = 8
LR = 1e-3
WEIGHT_DECAY = 1e-2
FULL_EPOCHS = 3000

# Set to None to run all input ablations. Keep voltage_only for quick retraining.
RUN_CONFIG_NAMES = ["voltage_only", 'aux_temp', 'aux_current', 'aux_capacity', 'channel_temp', 'channel_current', 'channel_capacity']

RESULT_DIR = Path("stage3_input_ablation")


@dataclass
class InputConfig:
    name: str
    variant: str
    mode: str
    features: str


INPUT_CONFIGS = [
    InputConfig("voltage_only", "voltage", "baseline", "voltage"),

    InputConfig("aux_temp", "voltage_temp", "aux", "temp"),
    InputConfig("aux_current", "voltage_current", "aux", "current"),
    InputConfig("aux_capacity", "voltage_capacity", "aux", "capacity"),
    InputConfig("aux_temp_current", "voltage_temp_current", "aux", "temp+current"),
    InputConfig("aux_temp_capacity", "voltage_temp_capacity", "aux", "temp+capacity"),
    InputConfig("aux_current_capacity", "voltage_current_capacity", "aux", "current+capacity"),
    InputConfig("aux_temp_current_capacity", "voltage_temp_current_capacity", "aux", "temp+current+capacity"),

    InputConfig("channel_temp", "voltage_temp_channel", "channel", "temp"),
    InputConfig("channel_current", "voltage_current_channel", "channel", "current"),
    InputConfig("channel_capacity", "voltage_capacity_channel", "channel", "capacity"),
    InputConfig("channel_temp_current", "voltage_temp_current_channel", "channel", "temp+current"),
    InputConfig("channel_temp_capacity", "voltage_temp_capacity_channel", "channel", "temp+capacity"),
    InputConfig("channel_current_capacity", "voltage_current_capacity_channel", "channel", "current+capacity"),
    InputConfig("channel_temp_current_capacity", "voltage_temp_current_capacity_channel", "channel", "temp+current+capacity"),
]


def make_loss_fn(diff_loss: bool) -> tuple[AELossConfig, AELoss]:
    cfg = AELossConfig(
        prof_weight=LAMBDA_P,
        diff_weight=LAMBDA_D if diff_loss else 0.0,
        cap_weight=LAMBDA_C,
        cap_loss_type="smoothl1",
        use_diff_loss=diff_loss,
    )
    return cfg, AELoss(cfg)


def backup_existing_result_dir(ckpt_dir: Path) -> None:
    ckpt_best = ckpt_dir / "best.pt"
    if not ckpt_best.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ckpt_dir.parent / f"{ckpt_dir.name}_backup_{timestamp}"
    shutil.copytree(ckpt_dir, backup_dir)
    print(f"[BACKUP] Existing result copied to: {backup_dir}")


def run_input_experiment(cfg: InputConfig, window_size: int, loss_setting: dict, device: torch.device) -> dict:
    print(f"\n{'='*80}\n[Stage 3 INPUT] {cfg.name} | {cfg.mode} | {cfg.features} | W={window_size} | loss={loss_setting['ablation']}\n{'='*80}")
    set_seed(RANDOM_SEED)
    bundle = build_dataloaders(
        variant=cfg.variant,
        window_size=window_size,
        use_half_profile=USE_HALF_PROFILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        split_json_path=SPLIT_JSON_PATH,
        h5_path=DB_ROOT_PUBLIC_H5,
    )
    model = build_model(
        model_type="conv",
        profile_len=bundle["profile_len"], window_size=window_size,
        in_channels=bundle["in_channels"], aux_dim=bundle["aux_dim"],
        latent_dim=LATENT_DIM, aux_embed_dim=AUX_EMBED_DIM,
        out_activation="sigmoid", use_segment_bias=loss_setting["seg_bias"],
    ).to(device)
    n_params = count_parameters(model)
    loss_cfg, loss_fn = make_loss_fn(loss_setting["diff_loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)

    ckpt_dir = RESULT_DIR / cfg.name
    backup_existing_result_dir(ckpt_dir)
    ensure_dir(ckpt_dir)
    ckpt_best = ckpt_dir / "best.pt"
    log_csv = ckpt_dir / "train_log.csv"
    best_val_rmse = float("inf")
    best_epoch = 0
    train_history, val_history = [], []
    t0 = time.time()

    for epoch in range(1, FULL_EPOCHS + 1):
        train_metrics = train_one_epoch(model, bundle["train_loader"], optimizer, loss_fn, device)
        val_metrics = validate_one_epoch(model, bundle["val_loader"], loss_fn, device)
        val_prof_rmse = float((val_metrics["loss_prof_scaled"] / max(LAMBDA_P, 1e-6)) ** 0.5)
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        if val_prof_rmse < best_val_rmse:
            best_val_rmse = val_prof_rmse
            best_epoch = epoch
            save_ckpt_ae(
                ckpt_best, epoch, model, optimizer, train_history, val_history,
                best_val_rmse, loss_cfg,
                extra={
                    "stage": "3", "input_name": cfg.name, "variant": cfg.variant,
                    "mode": cfg.mode, "features": cfg.features, "window_size": window_size,
                    "loss_ablation": loss_setting["ablation"],
                    "use_seg_bias": loss_setting["seg_bias"],
                    "use_diff_loss": loss_setting["diff_loss"],
                    "best_epoch": epoch,
                },
            )

        pd.DataFrame([{
            "epoch": epoch, "input_name": cfg.name,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_prof_rmse": val_prof_rmse, "best_val_rmse": best_val_rmse,
        }]).to_csv(log_csv, mode="a", header=not log_csv.exists(), index=False)

        if epoch % 500 == 0 or epoch == 1:
            print(f"  [{cfg.name}][{epoch:04d}] train_prof={train_metrics['loss_prof_scaled']:.4f} | val_rmse={val_prof_rmse:.6f} | best={best_val_rmse:.6f}")

    elapsed_min = (time.time() - t0) / 60
    plot_loss_components(train_history, val_history)
    load_ckpt_ae(ckpt_best, model=model, device=device)
    model.eval()

    results = evaluate_ae_predictions_aux(
        model=model,
        X_main=bundle["X_main_test"], X_aux=bundle["X_aux_test"],
        Yprof=bundle["Y_prof_test"], Ycap=bundle["Y_cap_test"],
        device=device, meta_val=bundle["meta_test"],
        charge_len=64, prominence=0.02,
        charge_peak_ylim=(0, 2), discharge_peak_ylim=(-2, 0),
        plot_best_worst_cell_profile=True, plot_cap_scatter=True,
        plot_best_worst_cap_aligned_profile=True, plot_best_worst_peak=True,
        plot_one_cell_trend=False, plot_all_cell_trends=False,
        plot_cell_slope_scatter=True,
        save_dir=ckpt_dir / "evaluation",
        result_csv_name="evaluation_summary.csv",
        cell_csv_name="evaluation_per_cell.csv",
        sample_csv_name="evaluation_per_sample.csv",
    )
    test_obj = compute_objective_metric(
        model=model,
        X_main=bundle["X_main_test"], X_aux=bundle["X_aux_test"],
        Y_prof=bundle["Y_prof_test"], Y_cap=bundle["Y_cap_test"], device=device,
    )
    summary = results.get("summary_df", pd.DataFrame())
    row = {
        "stage": "3", "input_name": cfg.name, "variant": cfg.variant,
        "mode": cfg.mode, "features": cfg.features, "window_size": window_size,
        "loss_ablation": loss_setting["ablation"],
        "seg_bias": loss_setting["seg_bias"], "diff_loss": loss_setting["diff_loss"],
        "in_channels": bundle["in_channels"], "aux_dim": bundle["aux_dim"],
        "n_params": n_params, "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch, "elapsed_min": round(elapsed_min, 2),
        "test_objective": test_obj,
    }
    if len(summary):
        for col in summary.columns:
            if col not in ["best_cell", "worst_cell"]:
                row[col] = summary.iloc[0][col]
    print(f"  [{cfg.name}] test_objective={test_obj:.4f}")
    return row


set_seed(RANDOM_SEED)
ensure_dir(RESULT_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window_size = WINDOW_SIZE
loss_setting = {"ablation": "seg_bias_diff", "seg_bias": USE_SEG_BIAS, "diff_loss": USE_DIFF_LOSS}
print(f"Device: {device}")
print(f"Objective = M_cnt + 10*M_pos + 5*M_cap + 30*M_profile")

run_configs = INPUT_CONFIGS if RUN_CONFIG_NAMES is None else [cfg for cfg in INPUT_CONFIGS if cfg.name in RUN_CONFIG_NAMES]
print("Run configs:", [cfg.name for cfg in run_configs])
rows = [run_input_experiment(cfg, window_size, loss_setting, device) for cfg in run_configs]
df = pd.DataFrame(rows).sort_values("test_objective").reset_index(drop=True)
df.to_csv(RESULT_DIR / "input_ablation_summary.csv", index=False)
print(df)
print(f"\n[DONE] Saved to: {RESULT_DIR}")

