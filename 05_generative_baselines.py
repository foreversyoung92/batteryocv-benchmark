# %%
"""
Generative baselines (Stage 5).

Reproduces Tables 9 (generative rows) and 10 (paper §5.5 / §B.6):
{Conv, LSTM, Transformer} VAE and Conv GAN, each independently
tuned with Optuna (TPE sampler, 20 trials, median pruner) under
the default configuration (W=64, Seg+Diff, voltage-only).
Cross-model evaluation on Models A-D follows the in-distribution
best checkpoint per architecture.

Outputs
-------
stage5_generative_baselines/
    Per-architecture Optuna study DBs, best HPs, full retraining
    checkpoints, and aggregate comparison CSVs.
"""

from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import find_peaks
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from pipeline import (
    RANDOM_SEED, WINDOW_SIZE, USE_HALF_PROFILE,
    BATCH_SIZE, NUM_WORKERS,
    DB_ROOT_PUBLIC_H5, SPLIT_JSON_PATH,
    set_seed, ensure_dir,
    build_dataloaders,
    build_crossmodel_bundle,
        AELossConfig, AELoss,
    move_batch_to_device,
    save_ckpt_ae, load_ckpt_ae,
    evaluate_ae_predictions_aux,
    EarlyStopping, compute_objective_metric,
    PROFILE_RMSE_SCALE,
)
from generative_models import (
    build_generative_model,
    VAELoss, GANLoss,
    run_epoch_vae, run_epoch_gan,
    count_parameters,
)

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================
VARIANT       = "voltage"
USE_SEG_BIAS  = True
USE_DIFF_LOSS = True
LATENT_DIM    = 128
BEST_WINDOW   = 64

N_TRIALS         = 20
MAX_EPOCHS       = 2000
ES_PATIENCE      = 200
ES_MIN_DELTA     = 1e-6

FULL_EPOCHS      = 3000
FULL_ES_PATIENCE = None  # full train: no early stopping

BETA_MAX    = 1.0
BETA_WARMUP = 200

MODEL_TYPES = [
    "conv_vae",
    "lstm_vae",
    "transformer_vae",
    "conv_gan",
]

RESULT_DIR = Path("stage5_generative_baselines")


# =========================================================
# β annealing
# =========================================================
def get_beta(epoch: int) -> float:
    if epoch <= BETA_WARMUP:
        return BETA_MAX * (epoch / max(BETA_WARMUP, 1))
    return BETA_MAX


# =========================================================
# Validation profile RMSE — common early-stopping criterion across all models
# =========================================================
@torch.no_grad()
def get_val_prof_rmse(model, val_loader, device) -> float:
    """Raw profile RMSE (independent of loss weights); used as the early-stopping criterion."""
    model.eval()
    total_rmse = 0.0
    n = 0
    for batch in val_loader:
        x_main, x_aux, y_prof, y_cap, _ = move_batch_to_device(batch, device)
        out = model(x_main, x_aux)
        y_pred = out[0]   # forward returns (y_prof, y_cap, z, ...); y_prof is always first
        bs = x_main.size(0)
        total_rmse += torch.sqrt(
            ((y_pred - y_prof) ** 2).mean()
        ).item() * bs
        n += bs
    return total_rmse / max(n, 1)


# =========================================================
# Hyperparameter search space
# =========================================================
def suggest_hyperparams(trial: optuna.Trial, model_type: str) -> Dict:
    hp = {
        "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "lambda_p": trial.suggest_categorical("lambda_p", [100, 300, 500]),
        "lambda_d": trial.suggest_categorical("lambda_d", [100, 300, 500]),
        "lambda_c": trial.suggest_categorical("lambda_c", [5, 10, 20]),
    }

    if model_type == "conv_vae":
        hp["beta"]          = trial.suggest_float("beta", 0.01, 1.0, log=True)
        hp["aux_embed_dim"] = trial.suggest_categorical("aux_embed_dim", [4, 8, 16])

    elif model_type == "lstm_vae":
        hp["beta"]       = trial.suggest_float("beta", 0.01, 1.0, log=True)
        hp["num_layers"] = trial.suggest_int("num_layers", 1, 3)
        hp["dropout"]    = trial.suggest_float("dropout", 0.0, 0.3)

    elif model_type == "transformer_vae":
        hp["beta"]               = trial.suggest_float("beta", 0.01, 1.0, log=True)
        hp["nhead"]              = trial.suggest_categorical("nhead", [2, 4, 8])
        hp["num_encoder_layers"] = trial.suggest_int("num_encoder_layers", 2, 5)
        hp["dim_feedforward"]    = trial.suggest_categorical("dim_feedforward", [128, 256, 512])
        hp["dropout"]            = trial.suggest_float("dropout", 0.0, 0.3)

    elif model_type == "conv_gan":
        hp["adv_weight"]      = trial.suggest_float("adv_weight", 0.01, 1.0, log=True)
        hp["aux_embed_dim"]   = trial.suggest_categorical("aux_embed_dim", [4, 8, 16])
        hp["disc_base_ch"]    = trial.suggest_categorical("disc_base_ch", [8, 16, 32])
        hp["n_disc_steps"]    = trial.suggest_categorical("n_disc_steps", [1, 2])
        hp["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.2)

    return hp


def build_model_from_hp(model_type, hp, bundle, device):
    key_map = {
        "conv_vae":        ["aux_embed_dim"],
        "lstm_vae":        ["num_layers", "dropout"],
        "transformer_vae": ["nhead", "num_encoder_layers", "dim_feedforward", "dropout"],
        "conv_gan":        ["aux_embed_dim", "disc_base_ch"],
    }
    kwargs = {k: hp[k] for k in key_map.get(model_type, []) if k in hp}
    return build_generative_model(
        model_type=model_type,
        profile_len=bundle["profile_len"],
        window_size=BEST_WINDOW,
        in_channels=bundle["in_channels"],
        aux_dim=bundle["aux_dim"],
        latent_dim=LATENT_DIM,
        out_activation="sigmoid",
        use_segment_bias=USE_SEG_BIAS,
        **kwargs,
    ).to(device)


def make_loss_fn(model_type, hp):
    lp = float(hp["lambda_p"])
    ld = float(hp["lambda_d"]) if USE_DIFF_LOSS else 0.0
    lc = float(hp["lambda_c"])

    if model_type in ("conv_vae", "lstm_vae", "transformer_vae"):
        return VAELoss(
            prof_weight=lp, cap_weight=lc, diff_weight=ld,
            beta=hp.get("beta", BETA_MAX),
            use_diff_loss=USE_DIFF_LOSS,
        )

    if model_type == "conv_gan":
        return GANLoss(
            prof_weight=lp, cap_weight=lc,
            adv_weight=float(hp.get("adv_weight", 0.1)),
            diff_weight=ld, use_diff_loss=USE_DIFF_LOSS,
            label_smoothing=float(hp.get("label_smoothing", 0.1)),
        )

    raise ValueError(f"Unsupported generative model_type: {model_type}")


# =========================================================
# Optuna objective
# =========================================================
def make_objective(model_type, bundle, device):
    def objective(trial):
        hp = suggest_hyperparams(trial, model_type)

        try:
            model = build_model_from_hp(model_type, hp, bundle, device)
        except Exception as e:
            raise optuna.exceptions.TrialPruned(f"Model build failed: {e}")

        loss_fn = make_loss_fn(model_type, hp)
        if model_type == "conv_gan":
            opt_G = torch.optim.AdamW(
                model.generator_params(),
                lr=hp["lr"], weight_decay=hp["weight_decay"], betas=(0.5, 0.999))
            opt_D = torch.optim.AdamW(
                model.discriminator_params(),
                lr=hp["lr"], weight_decay=hp["weight_decay"], betas=(0.5, 0.999))
            opt = None
        else:
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=hp["lr"], weight_decay=hp["weight_decay"], betas=(0.9, 0.95))
            opt_G = opt_D = None
        es = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)
        best_val_rmse = float("inf")
        best_state = None

        for epoch in range(1, MAX_EPOCHS + 1):
            beta = get_beta(epoch)
            if model_type == "conv_gan":
                run_epoch_gan(
                    model, bundle["train_loader"], loss_fn, device,
                    opt_G, opt_D, n_disc_steps=int(hp.get("n_disc_steps", 1)))
                val_prof_rmse = get_val_prof_rmse(model, bundle["val_loader"], device)
            else:
                run_epoch_vae(model, bundle["train_loader"], loss_fn, device, opt, beta=beta)
                val_prof_rmse = get_val_prof_rmse(model, bundle["val_loader"], device)

            trial.report(val_prof_rmse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_prof_rmse < best_val_rmse:
                best_val_rmse = val_prof_rmse
                best_state = copy.deepcopy(model.state_dict())

            if es.step(val_prof_rmse, epoch):
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return compute_objective_metric(
            model=model,
            X_main=bundle["X_main_val"],
            X_aux=bundle["X_aux_val"],
            Y_prof=bundle["Y_prof_val"],
            Y_cap=bundle["Y_cap_val"],
            device=device,
        )

    return objective


# =========================================================
# Full train
# =========================================================
def full_train(model_type, best_hp, bundle, device, save_dir):
    set_seed(RANDOM_SEED)
    model = build_model_from_hp(model_type, best_hp, bundle, device)
    loss_fn = make_loss_fn(model_type, best_hp)

    print(f"  [full train] {model_type}")
    print(f"    params={count_parameters(model):,}")
    print(f"  lambda_p={best_hp['lambda_p']} lambda_d={best_hp['lambda_d']} lambda_c={best_hp['lambda_c']}")

    ckpt_dir = save_dir / f"full_train_{model_type}"
    ensure_dir(ckpt_dir)
    ckpt_best = ckpt_dir / "best.pt"
    log_csv = ckpt_dir / "train_log.csv"

    if model_type == "conv_gan":
        opt_G = torch.optim.AdamW(
            model.generator_params(),
            lr=best_hp["lr"], weight_decay=best_hp["weight_decay"], betas=(0.5, 0.999))
        opt_D = torch.optim.AdamW(
            model.discriminator_params(),
            lr=best_hp["lr"], weight_decay=best_hp["weight_decay"], betas=(0.5, 0.999))
        opt = None
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=best_hp["lr"], weight_decay=best_hp["weight_decay"], betas=(0.9, 0.95))
        opt_G = opt_D = None
    best_val_metric = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, FULL_EPOCHS + 1):
        beta = get_beta(epoch)
        if model_type == "conv_gan":
            train_dict = run_epoch_gan(
                model, bundle["train_loader"], loss_fn, device,
                opt_G, opt_D, n_disc_steps=int(best_hp.get("n_disc_steps", 1)))
            val_prof_rmse = get_val_prof_rmse(model, bundle["val_loader"], device)
        else:
            train_dict = run_epoch_vae(
                model, bundle["train_loader"], loss_fn, device, opt, beta=beta)
            val_prof_rmse = get_val_prof_rmse(model, bundle["val_loader"], device)

        if val_prof_rmse < best_val_metric:
            best_val_metric = val_prof_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_best)

        history.append({
            "epoch": epoch,
            "beta": beta,
            **{f"train_{k}": v for k, v in train_dict.items()},
            "val_prof_rmse": val_prof_rmse,
            "best_val_rmse": best_val_metric,
        })

        if epoch % 500 == 0 or epoch == 1:
            print(f"  [{model_type}][{epoch:04d}] beta={beta:.2f} "
                  f"train_prof={train_dict.get('loss_prof_scaled', 0):.4f} "
                  f"train_total={train_dict.get('loss_total', train_dict.get('loss_G_total', 0)):.4f} "
                  f"kl={train_dict.get('loss_kl_scaled', 0):.4f} "
                  f"val_metric={val_prof_rmse:.6f} best={best_val_metric:.6f} "
                  f"(ep {best_epoch})")


    pd.DataFrame(history).to_csv(log_csv, index=False)
    print(f"  [{model_type}] Training done. best_epoch={best_epoch}")

    if history:
        import matplotlib.pyplot as plt
        h_df = pd.DataFrame(history)
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        if "train_loss_total" in h_df.columns:
            ax.plot(h_df["epoch"], h_df["train_loss_total"], label="train")
        ax2 = ax.twinx()
        ax2.plot(h_df["epoch"], h_df["val_prof_rmse"], color="orange", label="val_prof_rmse")
        ax2.set_ylabel("val prof RMSE")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.set_title(f"{model_type} full train loss")
        fig.tight_layout()
        fig.savefig(ckpt_dir / "loss_curve.png", dpi=120)
        plt.close(fig)

    if ckpt_best.exists():
        model.load_state_dict(torch.load(ckpt_best, map_location=device))
        print(f"  Loaded best checkpoint (epoch {best_epoch})")

    model.eval()
    print(f"\n  [{model_type}] Evaluating on test set...")
    results = evaluate_ae_predictions_aux(
        model=model,
        X_main=bundle["X_main_test"],
        X_aux=bundle["X_aux_test"],
        Yprof=bundle["Y_prof_test"],
        Ycap=bundle["Y_cap_test"],
        device=device,
        meta_val=bundle["meta_test"],
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

    test_objective = compute_objective_metric(
        model=model,
        X_main=bundle["X_main_test"],
        X_aux=bundle["X_aux_test"],
        Y_prof=bundle["Y_prof_test"],
        Y_cap=bundle["Y_cap_test"],
        device=device,
    )
    print(f"  [{model_type}] Test objective: {test_objective:.4f}")

    return {
        "model": model,
        "best_val_rmse": best_val_metric,
        "best_epoch": best_epoch,
        "eval_results": results,
        "test_objective": test_objective,
    }


# =========================================================
# Result summary
# =========================================================
def save_comparison_table(all_results, save_dir):
    rows = []
    for r in all_results:
        summary = r.get("eval_results", {}).get("summary_df", pd.DataFrame())
        row = {
            "model_type":     r["model_type"],
            "params":         r["params"],
            "best_trial_obj": r["best_trial_obj"],
            "test_objective": r.get("test_objective", np.nan),
            "best_epoch":     r.get("best_epoch", np.nan),
            "best_hp":        str(r["best_hp"]),
        }
        if len(summary) > 0:
            for col in summary.columns:
                if col not in ["best_cell", "worst_cell"]:
                    row[col] = summary.iloc[0][col]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        "test_objective", na_position="last"
    ).reset_index(drop=True)
    df.to_csv(save_dir / "generative_comparison.csv", index=False)

    print("\n" + "=" * 80)
    print("GENERATIVE MODEL COMPARISON (sorted by test_objective)")
    print("=" * 80)
    cols = ["model_type", "params", "best_trial_obj", "test_objective",
            "profile_rmse", "cap_rmse", "cap_aligned_profile_rmse_mean",
            "peak_count_loss_mean", "peak_pos_loss_mean", "best_epoch"]
    cols_exist = [c for c in cols if c in df.columns]
    print(df[cols_exist].to_string(index=False))
    return df


# =========================================================
# MAIN
# =========================================================
set_seed(RANDOM_SEED)
ensure_dir(RESULT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

USE_SEG_BIAS  = True
USE_DIFF_LOSS = True
best_window   = BEST_WINDOW
print(f"[INFO] Fixed config: input={VARIANT}, "
      f"seg={USE_SEG_BIAS}, diff={USE_DIFF_LOSS}, W={best_window}")

bundle = build_dataloaders(
    variant=VARIANT,
    window_size=best_window,
    use_half_profile=USE_HALF_PROFILE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    split_json_path=SPLIT_JSON_PATH,
    h5_path=DB_ROOT_PUBLIC_H5,
)

print("=" * 80)
print(f"[STAGE 5 GENERATIVE MODEL SEARCH]")
print(f"  input      : {VARIANT}")
print(f"  objective  : M_cnt + 10*M_pos + 5*M_cap + 30*M_profile")
print(f"  λp/λd      : [100, 300, 500]")
print(f"  λc         : [5, 10, 20]")
print(f"  beta warmup: {BETA_WARMUP} epochs (VAE only)")
print(f"  models     : {MODEL_TYPES}")
print(f"  train/val/test: {len(bundle['train_dataset'])} / "
      f"{len(bundle['val_dataset'])} / {len(bundle['test_dataset'])}")
print("=" * 80)

all_results = []

for model_type in MODEL_TYPES:
    model_dir = RESULT_DIR / model_type
    ensure_dir(model_dir)

    study_path = model_dir / "optuna_study.db"
    if study_path.exists():
        study_path.unlink()
        print(f"  [{model_type}] Removed old optuna study DB.")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        storage=f"sqlite:///{study_path}",
        study_name=model_type,
    )

    print(f"\n{'='*80}")
    print(f"[OPTUNA SEARCH] {model_type}  (N_TRIALS={N_TRIALS})")
    print(f"{'='*80}")

    study.optimize(make_objective(model_type, bundle, device), n_trials=N_TRIALS, show_progress_bar=True)
    study.trials_dataframe().to_csv(model_dir / "optuna_trials.csv", index=False)

    best_trial = study.best_trial
    best_obj   = best_trial.value
    best_hp    = best_trial.params

    print(f"  [{model_type}] Best trial obj={best_obj:.4f}")
    print(f"  [{model_type}] Best HP: {best_hp}")

    tmp = build_model_from_hp(model_type, best_hp, bundle, device)
    n_params = count_parameters(tmp)
    del tmp

    print(f"\n{'='*80}")
    print(f"[FULL TRAIN] {model_type}  (best_obj={best_obj:.4f})")
    print(f"{'='*80}")

    full_result = full_train(model_type, best_hp, bundle, device, model_dir)

    all_results.append({
        "model_type":     model_type,
        "params":         n_params,
        "best_trial_obj": best_obj,
        "best_epoch":     full_result["best_epoch"],
        "test_objective": full_result["test_objective"],
        "best_hp":        best_hp,
        "eval_results":   full_result["eval_results"],
    })

save_comparison_table(all_results, RESULT_DIR)
print(f"\nAll results saved to: {RESULT_DIR}")

# %%
# =========================================================
# CROSS-MODEL GENERALISATION (Models A-D, hold-out)
# =========================================================
cm_bundle = build_crossmodel_bundle(
    variant=VARIANT,
    window_size=best_window,
    use_half_profile=USE_HALF_PROFILE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    split_json_path=SPLIT_JSON_PATH,
    h5_path=DB_ROOT_PUBLIC_H5,
)

cm_rows = []
for r in all_results:
    model_type = r["model_type"]
    model_dir = RESULT_DIR / model_type
    ckpt_path = model_dir / f"full_train_{model_type}" / "best.pt"
    if not ckpt_path.exists():
        print(f"  [{model_type}] checkpoint not found, skipping.")
        continue

    model = build_model_from_hp(model_type, r["best_hp"], bundle, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    cm_eval_dir = model_dir / "crossmodel_evaluation"
    ensure_dir(cm_eval_dir)

    cm_results = evaluate_ae_predictions_aux(
        model=model,
        X_main=cm_bundle["X_main_test"],
        X_aux=cm_bundle["X_aux_test"],
        Yprof=cm_bundle["Y_prof_test"],
        Ycap=cm_bundle["Y_cap_test"],
        device=device,
        meta_val=cm_bundle["meta_test"],
        charge_len=64, prominence=0.02,
        charge_peak_ylim=(0, 2), discharge_peak_ylim=(-2, 0),
        plot_best_worst_cell_profile=True, plot_cap_scatter=True,
        plot_best_worst_cap_aligned_profile=True, plot_best_worst_peak=True,
        plot_one_cell_trend=False, plot_all_cell_trends=False,
        plot_cell_slope_scatter=True,
        save_dir=cm_eval_dir,
        result_csv_name="crossmodel_summary.csv",
        cell_csv_name="crossmodel_per_cell.csv",
        sample_csv_name="crossmodel_per_sample.csv",
    )

    cm_obj = compute_objective_metric(
        model=model,
        X_main=cm_bundle["X_main_test"],
        X_aux=cm_bundle["X_aux_test"],
        Y_prof=cm_bundle["Y_prof_test"],
        Y_cap=cm_bundle["Y_cap_test"],
        device=device,
    )

    summary = cm_results.get("summary_df", pd.DataFrame())
    row = {"model_type": model_type, "setting": "cross-model", "crossmodel_obj": cm_obj}
    if len(summary) > 0:
        for col in summary.columns:
            if col not in ["best_cell", "worst_cell"]:
                row[col] = summary.iloc[0][col]
    cm_rows.append(row)
    print(f"  [{model_type}] cross-model objective: {cm_obj:.4f}")

if cm_rows:
    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(RESULT_DIR / "crossmodel_comparison.csv", index=False)
    print(f"\nCross-model results saved to: {RESULT_DIR / 'crossmodel_comparison.csv'}")


