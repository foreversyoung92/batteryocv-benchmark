# %%
"""
Architecture comparison (Stage 4).

Reproduces Tables 5 and 9 (discriminative rows; paper §5.4 / §B.6):
seven discriminative architectures (Conv, MLP, LSTM, BiLSTM,
Transformer, UNet, Conv+LSTM AE), each independently tuned with
Optuna (TPE sampler, 20 trials, median pruner) under the default
configuration (W=64, Seg+Diff, voltage-only). Cross-model
evaluation on Models A-D follows the in-distribution best
checkpoint per architecture.

Outputs
-------
stage4_architecture_comparison/
    Per-architecture Optuna study DBs, best HPs, full retraining
    checkpoints, and aggregate comparison CSVs.
"""

from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    # training utilities
    PROFILE_RMSE_SCALE,
    EarlyStopping,
    run_epoch,
    compute_objective_metric,
)
from models import build_model, count_parameters

warnings.filterwarnings('ignore')

# %%
# =========================================================
# CONFIG
# =========================================================
VARIANT       = 'voltage'
USE_SEG_BIAS  = True
USE_DIFF_LOSS = True
LATENT_DIM    = 128
BEST_WINDOW   = 64

N_TRIALS         = 20
MAX_EPOCHS       = 1000
ES_PATIENCE      = 50
ES_MIN_DELTA     = 1e-6

FULL_EPOCHS      = 3000
FULL_ES_PATIENCE = None  # full train: no early stopping

MODEL_TYPES = [
    'conv',
    'mlp',
    'lstm',
    'bilstm',
    'transformer',
    'unet',
    'conv_lstm',
]

RESULT_DIR = Path('stage4_architecture_comparison')

# %%
# =========================================================
# Hyperparameter search space
# =========================================================
def suggest_hyperparams(trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
    hp = {
        'lr':           trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
        'lambda_p': trial.suggest_categorical('lambda_p', [100, 300, 500]),
        'lambda_d': trial.suggest_categorical('lambda_d', [100, 300, 500]),
        'lambda_c': trial.suggest_categorical('lambda_c', [5, 10, 20]),
    }
    if model_type == 'conv':
        hp['aux_embed_dim'] = trial.suggest_categorical('aux_embed_dim', [4, 8, 16])
    elif model_type == 'mlp':
        hp['hidden_dim']    = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
        hp['aux_embed_dim'] = trial.suggest_categorical('aux_embed_dim', [4, 8, 16])
    elif model_type in ('lstm', 'bilstm'):
        hp['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        hp['dropout']    = trial.suggest_float('dropout', 0.0, 0.3)
    elif model_type == 'transformer':
        hp['nhead']              = trial.suggest_categorical('nhead', [2, 4, 8])
        hp['num_encoder_layers'] = trial.suggest_int('num_encoder_layers', 2, 5)
        hp['dim_feedforward']    = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
        hp['dropout']            = trial.suggest_float('dropout', 0.0, 0.3)
    elif model_type == 'unet':
        hp['base_ch']       = trial.suggest_categorical('base_ch', [16, 32, 64])
        hp['aux_embed_dim'] = trial.suggest_categorical('aux_embed_dim', [4, 8, 16])
    elif model_type == 'conv_lstm':
        hp['conv_embed_dim'] = trial.suggest_categorical('conv_embed_dim', [32, 64, 128])
        hp['num_layers']     = trial.suggest_int('num_layers', 1, 3)
        hp['dropout']        = trial.suggest_float('dropout', 0.0, 0.3)
    return hp


def build_model_from_hp(model_type: str, hp: Dict, bundle: Dict, device) -> nn.Module:
    model_keys = {
        'conv':        ['aux_embed_dim'],
        'mlp':         ['hidden_dim', 'aux_embed_dim'],
        'lstm':        ['num_layers', 'dropout'],
        'bilstm':      ['num_layers', 'dropout'],
        'transformer': ['nhead', 'num_encoder_layers', 'dim_feedforward', 'dropout'],
        'unet':        ['base_ch', 'aux_embed_dim'],
        'conv_lstm':   ['conv_embed_dim', 'num_layers', 'dropout'],
    }
    kwargs = {k: hp[k] for k in model_keys.get(model_type, []) if k in hp}
    return build_model(
        model_type=model_type,
        profile_len=bundle['profile_len'],
        window_size=BEST_WINDOW,
        in_channels=bundle['in_channels'],
        aux_dim=bundle['aux_dim'],
        latent_dim=LATENT_DIM,
        out_activation='sigmoid',
        use_segment_bias=USE_SEG_BIAS,
        **kwargs,
    ).to(device)


def make_loss_fn(hp: Dict) -> AELoss:
    return AELoss(AELossConfig(
        prof_weight=float(hp['lambda_p']),
        diff_weight=float(hp['lambda_d']),
        cap_weight=float(hp['lambda_c']),
        cap_loss_type='smoothl1',
        use_diff_loss=USE_DIFF_LOSS,
    ))


def make_objective(model_type: str, bundle: Dict, device):
    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hyperparams(trial, model_type)
        try:
            model = build_model_from_hp(model_type, hp, bundle, device)
        except Exception as e:
            raise optuna.exceptions.TrialPruned(f'Model build failed: {e}')

        loss_fn   = make_loss_fn(hp)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hp['lr'], weight_decay=hp['weight_decay'], betas=(0.9, 0.95),
        )
        es = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)
        best_val_rmse = float('inf')
        best_state    = None

        for epoch in range(1, MAX_EPOCHS + 1):
            run_epoch(model, bundle['train_loader'], loss_fn, device, optimizer)
            _, val_prof_rmse = run_epoch(model, bundle['val_loader'], loss_fn, device)

            trial.report(val_prof_rmse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_prof_rmse < best_val_rmse:
                best_val_rmse = val_prof_rmse
                best_state    = copy.deepcopy(model.state_dict())

            if es.step(val_prof_rmse, epoch):
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return compute_objective_metric(
            model=model,
            X_main=bundle['X_main_val'], X_aux=bundle['X_aux_val'],
            Y_prof=bundle['Y_prof_val'], Y_cap=bundle['Y_cap_val'],
            device=device,
        )
    return objective


def full_train(model_type: str, best_hp: Dict, bundle: Dict, device, save_dir: Path) -> Dict:
    set_seed(RANDOM_SEED)
    model   = build_model_from_hp(model_type, best_hp, bundle, device)
    loss_fn = make_loss_fn(best_hp)
    print(f'  [full train] {model_type} | params={count_parameters(model):,}')
    print(f'  \u03bbp={best_hp["lambda_p"]} \u03bbd={best_hp["lambda_d"]} \u03bbc={best_hp["lambda_c"]}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_hp['lr'], weight_decay=best_hp['weight_decay'], betas=(0.9, 0.95),
    )
    best_val_rmse = float('inf')
    best_state    = None
    best_epoch    = 0
    history       = []

    ckpt_dir = save_dir / f'full_train_{model_type}'
    ensure_dir(ckpt_dir)
    log_csv = ckpt_dir / 'train_log.csv'

    for epoch in range(1, FULL_EPOCHS + 1):
        train_loss, train_rmse = run_epoch(model, bundle['train_loader'], loss_fn, device, optimizer)
        val_loss,   val_rmse   = run_epoch(model, bundle['val_loader'],   loss_fn, device)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state    = copy.deepcopy(model.state_dict())
            best_epoch    = epoch
            torch.save(best_state, ckpt_dir / 'best.pt')

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_prof_rmse': train_rmse,
            'val_loss': val_loss, 'val_prof_rmse': val_rmse, 'best_val_rmse': best_val_rmse,
        })

        if epoch % 200 == 0 or epoch == 1:
            print(f'  [{model_type}][{epoch:04d}] '
                  f'train_rmse={train_rmse:.6f} | val_rmse={val_rmse:.6f} | '
                  f'best={best_val_rmse:.6f} | best_epoch={best_epoch}')

    pd.DataFrame(history).to_csv(log_csv, index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f'\n  [{model_type}] Evaluating on test set...')
    results = evaluate_ae_predictions_aux(
        model=model,
        X_main=bundle['X_main_test'], X_aux=bundle['X_aux_test'],
        Yprof=bundle['Y_prof_test'],  Ycap=bundle['Y_cap_test'],
        device=device, meta_val=bundle['meta_test'],
        charge_len=64, prominence=0.02,
        charge_peak_ylim=(0, 2), discharge_peak_ylim=(-2, 0),
        plot_best_worst_cell_profile=True, plot_cap_scatter=True,
        plot_best_worst_cap_aligned_profile=True, plot_best_worst_peak=True,
        plot_one_cell_trend=False, plot_all_cell_trends=False,
        plot_cell_slope_scatter=True,
        save_dir=ckpt_dir / 'evaluation',
        result_csv_name='evaluation_summary.csv',
        cell_csv_name='evaluation_per_cell.csv',
        sample_csv_name='evaluation_per_sample.csv',
    )

    test_objective = compute_objective_metric(
        model=model,
        X_main=bundle['X_main_test'], X_aux=bundle['X_aux_test'],
        Y_prof=bundle['Y_prof_test'],  Y_cap=bundle['Y_cap_test'],
        device=device,
    )
    print(f'  [{model_type}] Test objective: {test_objective:.4f}')

    return {
        'model': model, 'best_val_rmse': best_val_rmse,
        'best_epoch': best_epoch, 'eval_results': results,
        'test_objective': test_objective,
    }


def save_comparison_table(all_results, save_dir):
    rows = []
    for r in all_results:
        summary = r.get('eval_results', {}).get('summary_df', pd.DataFrame())
        row = {
            'model_type':     r['model_type'],
            'params':         r['params'],
            'best_trial_obj': r['best_trial_obj'],
            'test_objective': r.get('test_objective', np.nan),
            'full_val_rmse':  r.get('full_val_rmse', np.nan),
            'best_epoch':     r.get('best_epoch', np.nan),
            'best_hp':        str(r['best_hp']),
        }
        if len(summary) > 0:
            for col in summary.columns:
                if col not in ['best_cell', 'worst_cell']:
                    row[col] = summary.iloc[0][col]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('test_objective', na_position='last').reset_index(drop=True)
    df.to_csv(save_dir / 'model_comparison_summary.csv', index=False)

    print('\n' + '=' * 80)
    print('MODEL COMPARISON (sorted by test_objective)')
    print('=' * 80)
    cols = ['model_type', 'params', 'best_trial_obj', 'test_objective',
            'profile_rmse', 'cap_rmse', 'cap_aligned_profile_rmse_mean',
            'peak_count_loss_mean', 'peak_pos_loss_mean', 'best_epoch']
    cols_exist = [c for c in cols if c in df.columns]
    print(df[cols_exist].to_string(index=False))
    return df

# %%
# =========================================================
# MAIN
# =========================================================
set_seed(RANDOM_SEED)
ensure_dir(RESULT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

best_window = BEST_WINDOW

bundle = build_dataloaders(
    variant=VARIANT,
    window_size=best_window,
    use_half_profile=USE_HALF_PROFILE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    split_json_path=SPLIT_JSON_PATH,
    h5_path=DB_ROOT_PUBLIC_H5,
)

print('=' * 80)
print(f'[STAGE 4 MODEL SEARCH]')
print(f'  input          : {VARIANT}')
print(f'  seg_bias       : {USE_SEG_BIAS}')
print(f'  diff_loss      : {USE_DIFF_LOSS}')
print(f'  objective      : M_cnt + 10*M_pos + 5*M_cap + 30*M_profile')
print(f'  train/val/test : {len(bundle["train_dataset"])} / {len(bundle["val_dataset"])} / {len(bundle["test_dataset"])}')
print('=' * 80)

all_results = []

for model_type in MODEL_TYPES:
    print(f"\n{'='*80}")
    print(f'[OPTUNA] {model_type}  ({N_TRIALS} trials)')
    print(f"{'='*80}")

    t0 = time.time()
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        study_name=f'{model_type}_{VARIANT}_stage4_arch_comparison',
    )
    study.optimize(make_objective(model_type, bundle, device), n_trials=N_TRIALS, show_progress_bar=True)

    best_trial = study.best_trial
    best_hp    = best_trial.params
    best_obj   = best_trial.value
    elapsed    = time.time() - t0

    print(f'\n  [{model_type}] Optuna done in {elapsed/60:.1f}min')
    print(f'  best_objective : {best_obj:.4f}')
    print(f'  best_hp        : {best_hp}')

    model_dir = RESULT_DIR / model_type
    ensure_dir(model_dir)
    study.trials_dataframe().to_csv(model_dir / 'optuna_trials.csv', index=False)

    tmp_model = build_model_from_hp(model_type, best_hp, bundle, device)
    n_params  = count_parameters(tmp_model)
    del tmp_model

    full_result = full_train(model_type, best_hp, bundle, device, model_dir)

    all_results.append({
        'model_type':     model_type,
        'params':         n_params,
        'best_trial_obj': best_obj,
        'full_val_rmse':  full_result['best_val_rmse'],
        'best_epoch':     full_result['best_epoch'],
        'test_objective': full_result['test_objective'],
        'best_hp':        best_hp,
        'eval_results':   full_result['eval_results'],
    })

save_comparison_table(all_results, RESULT_DIR)
print(f'\nAll results saved to: {RESULT_DIR}')

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
    model_type = r['model_type']
    model_dir  = RESULT_DIR / model_type
    ckpt_path  = model_dir / f'full_train_{model_type}' / 'best.pt'
    if not ckpt_path.exists():
        print(f'  [{model_type}] checkpoint not found, skipping.')
        continue

    model = build_model_from_hp(model_type, r['best_hp'], bundle, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    cm_eval_dir = model_dir / 'crossmodel_evaluation'
    ensure_dir(cm_eval_dir)

    cm_results = evaluate_ae_predictions_aux(
        model=model,
        X_main=cm_bundle['X_main_test'], X_aux=cm_bundle['X_aux_test'],
        Yprof=cm_bundle['Y_prof_test'],  Ycap=cm_bundle['Y_cap_test'],
        device=device, meta_val=cm_bundle['meta_test'],
        charge_len=64, prominence=0.02,
        charge_peak_ylim=(0, 2), discharge_peak_ylim=(-2, 0),
        plot_best_worst_cell_profile=True, plot_cap_scatter=True,
        plot_best_worst_cap_aligned_profile=True, plot_best_worst_peak=True,
        plot_one_cell_trend=False, plot_all_cell_trends=False,
        plot_cell_slope_scatter=True,
        save_dir=cm_eval_dir,
        result_csv_name='crossmodel_summary.csv',
        cell_csv_name='crossmodel_per_cell.csv',
        sample_csv_name='crossmodel_per_sample.csv',
    )

    cm_obj = compute_objective_metric(
        model=model,
        X_main=cm_bundle['X_main_test'], X_aux=cm_bundle['X_aux_test'],
        Y_prof=cm_bundle['Y_prof_test'],  Y_cap=cm_bundle['Y_cap_test'],
        device=device,
    )

    summary = cm_results.get('summary_df', pd.DataFrame())
    row = {'model_type': model_type, 'setting': 'cross-model', 'crossmodel_obj': cm_obj}
    if len(summary) > 0:
        for col in summary.columns:
            if col not in ['best_cell', 'worst_cell']:
                row[col] = summary.iloc[0][col]
    cm_rows.append(row)
    print(f'  [{model_type}] cross-model objective: {cm_obj:.4f}')

if cm_rows:
    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(RESULT_DIR / 'crossmodel_comparison.csv', index=False)
    print(f'\nCross-model results saved to: {RESULT_DIR / "crossmodel_comparison.csv"}')

