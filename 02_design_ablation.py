# %%
"""
Design components ablation (Stage 2).

Reproduces Table 4 (paper §5.2): four configurations
{Base, Seg, Diff, Seg+Diff} at the fixed configuration W=64 with
voltage-only input, seed 42.

Outputs
-------
stage2_design_ablation/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
# STAGE 2: Loss / segment-bias ablation after best window
# NOTE: Existing codebase implements segment as learnable segment bias, not a separate loss term.
# =========================================================
VARIANT = "voltage"
WINDOW_SIZE = 64

LAMBDA_P = 100.0
LAMBDA_D = 100.0
LAMBDA_C = 5.0

LATENT_DIM = 128
AUX_EMBED_DIM = 8
LR = 1e-3
WEIGHT_DECAY = 1e-2
FULL_EPOCHS = 3000

RESULT_DIR = Path("stage2_design_ablation")


@dataclass
class LossAblationConfig:
    name: str
    use_segment_bias: bool
    use_diff_loss: bool


ABLATION_CONFIGS = [
    LossAblationConfig("base", False, False),
    LossAblationConfig("seg_bias_only", True, False),
    LossAblationConfig("diff_only", False, True),
    LossAblationConfig("seg_bias_diff", True, True),
]


def make_loss_fn(cfg: LossAblationConfig) -> tuple[AELossConfig, AELoss]:
    loss_cfg = AELossConfig(
        prof_weight=LAMBDA_P,
        diff_weight=LAMBDA_D if cfg.use_diff_loss else 0.0,
        cap_weight=LAMBDA_C,
        cap_loss_type="smoothl1",
        use_diff_loss=cfg.use_diff_loss,
    )
    return loss_cfg, AELoss(loss_cfg)


def run_loss_ablation(cfg: LossAblationConfig, window_size: int, bundle: dict, device: torch.device) -> dict:
    print(f"\n{'='*80}\n[Stage 2 LOSS] {cfg.name} | W={window_size} | seg_bias={cfg.use_segment_bias} diff={cfg.use_diff_loss}\n{'='*80}")
    set_seed(RANDOM_SEED)
    model = build_model(
        model_type="conv",
        profile_len=bundle["profile_len"], window_size=window_size,
        in_channels=bundle["in_channels"], aux_dim=bundle["aux_dim"],
        latent_dim=LATENT_DIM, aux_embed_dim=AUX_EMBED_DIM,
        out_activation="sigmoid", use_segment_bias=cfg.use_segment_bias,
    ).to(device)
    n_params = count_parameters(model)
    loss_cfg, loss_fn = make_loss_fn(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)

    ckpt_dir = RESULT_DIR / cfg.name
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
                    "stage": "2", "variant": VARIANT, "window_size": window_size,
                    "ablation": cfg.name, "use_seg_bias": cfg.use_segment_bias,
                    "use_diff_loss": cfg.use_diff_loss, "best_epoch": epoch,
                },
            )

        pd.DataFrame([{
            "epoch": epoch, "ablation": cfg.name,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_prof_rmse": val_prof_rmse, "best_val_rmse": best_val_rmse,
        }]).to_csv(log_csv, mode="a", header=not log_csv.exists(), index=False)

        if epoch % 500 == 0 or epoch == 1:
            print(f"  [{cfg.name}][{epoch:04d}] train_prof={train_metrics['loss_prof_scaled']:.4f} | train_diff={train_metrics['loss_diff_scaled']:.4f} | val_rmse={val_prof_rmse:.6f} | best={best_val_rmse:.6f}")

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
        "stage": "2", "variant": VARIANT, "window_size": window_size,
        "ablation": cfg.name, "seg_bias": cfg.use_segment_bias,
        "diff_loss": cfg.use_diff_loss, "n_params": n_params,
        "best_val_rmse": best_val_rmse, "best_epoch": best_epoch,
        "elapsed_min": round(elapsed_min, 2), "test_objective": test_obj,
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
bundle = build_dataloaders(
    variant=VARIANT, window_size=window_size, use_half_profile=USE_HALF_PROFILE,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    split_json_path=SPLIT_JSON_PATH, h5_path=DB_ROOT_PUBLIC_H5,
)
print(f"Device: {device}")
print(f"Objective = M_cnt + 10*M_pos + 5*M_cap + 30*M_profile")

rows = [run_loss_ablation(cfg, window_size, bundle, device) for cfg in ABLATION_CONFIGS]
df = pd.DataFrame(rows).sort_values("test_objective").reset_index(drop=True)
df.to_csv(RESULT_DIR / "design_ablation_summary.csv", index=False)
print(df)
print(f"\n[DONE] Saved to: {RESULT_DIR}")

