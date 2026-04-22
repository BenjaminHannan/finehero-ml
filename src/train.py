# train.py - Trains CatBoost on the RTX 5070 Ti with:
#   - Native categorical handling (cat_features) — CatBoost builds its own
#     ordered target statistics, which beats manual target encoding.
#   - Optuna hyperparameter search (20 trials, early-stopped)
#   - Final training at ITERATIONS iters with early stopping on eval set
# Also trains a baseline LogisticRegression.

import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

FEATURES_PATH     = os.path.join(DATA_DIR, "features.csv")
CATBOOST_PATH     = os.path.join(MODELS_DIR, "catboost_model.cbm")
LR_PATH           = os.path.join(MODELS_DIR, "lr_model.joblib")
TEST_PATH         = os.path.join(DATA_DIR, "test_set.joblib")
CAT_FEATURES_PATH = os.path.join(MODELS_DIR, "cat_features.joblib")
BEST_PARAMS_PATH  = os.path.join(MODELS_DIR, "best_params.joblib")

# Tunable knobs
USE_OPTUNA   = True
OPTUNA_TRIALS = 80   # widened per FineHero AUC playbook §4.1 (canonical 150; 80 as compute compromise)
OPTUNA_ITERS = 800    # shorter CB runs during search
FINAL_ITERS  = 5000   # upper bound for final fit
EARLY_STOP   = 100

# If True, split train/test chronologically by issue_date instead of random
# stratified. Playbook §2.1 — honest evaluation baseline for dispute problems
# with per-plate features. Expected to lower reported AUC vs random split.
USE_TIME_AWARE_SPLIT = True

META_COLS = ["issue_date"]  # columns present in features.csv that are NOT model features


def _fit_catboost(X_tr, y_tr, X_ev, y_ev, cat_features, params, iters, verbose=0,
                  task_type="GPU"):
    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_features)
    eval_pool  = Pool(X_ev, label=y_ev, cat_features=cat_features)

    kwargs = dict(
        iterations=iters,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        early_stopping_rounds=EARLY_STOP,
        random_seed=42,
        task_type=task_type,
        verbose=verbose,
        **params,
    )
    if task_type == "GPU":
        kwargs["devices"] = "0"

    cb = CatBoostClassifier(**kwargs)
    cb.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    return cb


def _optuna_search(X_tr, y_tr, X_ev, y_ev, cat_features, task_type):
    """Lightweight Optuna search over CatBoost hyperparameters."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # Canonical CatBoost search space from FineHero AUC playbook §4.1
        # (Prokhorenkova et al. 2018; Akiba et al. 2019 for Optuna).
        # random_strength and rsm are excluded: not supported on GPU in non-pairwise modes.
        params = {
            "depth":                     trial.suggest_int("depth", 4, 10),
            "learning_rate":             trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg":               trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "border_count":              trial.suggest_int("border_count", 64, 254),
            "bootstrap_type":            "Bayesian",
            "bagging_temperature":       trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
            "grow_policy":               trial.suggest_categorical("grow_policy",
                                                                   ["SymmetricTree", "Depthwise", "Lossguide"]),
        }
        # min_data_in_leaf: only Depthwise/Lossguide
        if params["grow_policy"] in ("Depthwise", "Lossguide"):
            params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100)
        # max_leaves: only Lossguide
        if params["grow_policy"] == "Lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)

        try:
            cb = _fit_catboost(X_tr, y_tr, X_ev, y_ev, cat_features,
                               params, OPTUNA_ITERS, verbose=0, task_type=task_type)
            return cb.get_best_score()["validation"]["AUC"]
        except Exception as e:
            # Some param combos are invalid on GPU; prune rather than crash
            print(f"    [trial pruned] {type(e).__name__}: {e}")
            raise optuna.TrialPruned()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    print(f"\n  Running Optuna search: {OPTUNA_TRIALS} trials x {OPTUNA_ITERS} iters each")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    print(f"\n  Optuna best AUC: {study.best_value:.4f}")
    print(f"  Optuna best params: {study.best_params}")
    return study.best_params


def train_models() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("  Loading features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  Dataset: {len(df):,} rows x {df.shape[1]} columns")

    cat_features = joblib.load(CAT_FEATURES_PATH)
    cat_features = [c for c in cat_features if c in df.columns]

    # Separate meta columns (e.g. issue_date) from model features
    meta_present = [c for c in META_COLS if c in df.columns]
    meta = df[meta_present].copy() if meta_present else None

    X = df.drop(columns=["won"] + meta_present)
    y = df["won"]
    feature_names = list(X.columns)

    # CatBoost needs cat columns as strings (not float NaN)
    for c in cat_features:
        X[c] = X[c].fillna("UNKNOWN").astype(str)

    print(f"  Cat features: {len(cat_features)}  |  Numeric features: {len(feature_names)-len(cat_features)}")

    if USE_TIME_AWARE_SPLIT and meta is not None and "issue_date" in meta.columns:
        print("  Splitting chronologically by issue_date (last 20% = test)...")
        order = pd.to_datetime(meta["issue_date"], errors="coerce").argsort()
        X = X.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)
        cut = int(len(X) * 0.80)
        X_train, X_test = X.iloc[:cut].reset_index(drop=True), X.iloc[cut:].reset_index(drop=True)
        y_train, y_test = y.iloc[:cut].reset_index(drop=True), y.iloc[cut:].reset_index(drop=True)
    else:
        print("  Splitting 80/20 (stratified, random)...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Save test set for evaluate.py
    joblib.dump((X_test, y_test, feature_names), TEST_PATH)
    print(f"  Test set saved -> {TEST_PATH}")

    # --- GPU detection ---
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        print(f"\n  GPU: {'YES — ' + torch.cuda.get_device_name(0) if has_gpu else 'NO (CPU)'}")
    except ImportError:
        has_gpu = False
        print("\n  GPU: NO (torch not installed)")
    task_type = "GPU" if has_gpu else "CPU"

    # Carve 10% eval slice off training for early-stopping + Optuna
    if USE_TIME_AWARE_SPLIT:
        cut2 = int(len(X_train) * 0.90)
        X_tr2, X_ev = X_train.iloc[:cut2].reset_index(drop=True), X_train.iloc[cut2:].reset_index(drop=True)
        y_tr2, y_ev = y_train.iloc[:cut2].reset_index(drop=True), y_train.iloc[cut2:].reset_index(drop=True)
    else:
        eval_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=7)
        tr2_idx, ev_idx = next(eval_sss.split(X_train, y_train))
        X_tr2, X_ev = X_train.iloc[tr2_idx].reset_index(drop=True), X_train.iloc[ev_idx].reset_index(drop=True)
        y_tr2, y_ev = y_train.iloc[tr2_idx].reset_index(drop=True), y_train.iloc[ev_idx].reset_index(drop=True)
    print(f"  CB train: {len(X_tr2):,}  |  eval (early-stop): {len(X_ev):,}")

    # --- Optuna hyperparameter search ---
    if USE_OPTUNA:
        try:
            best_params = _optuna_search(X_tr2, y_tr2, X_ev, y_ev, cat_features, task_type)
        except ImportError:
            print("  [WARN] optuna not installed — skipping search. Run: pip install optuna")
            best_params = {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "border_count": 254}
    else:
        best_params = {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "border_count": 254}

    joblib.dump(best_params, BEST_PARAMS_PATH)

    # --- Final training with best params ---
    print(f"\n  Final CatBoost training on {task_type} ({FINAL_ITERS} iters, early_stop={EARLY_STOP})")
    print(f"  Using params: {best_params}\n")
    cb = _fit_catboost(X_tr2, y_tr2, X_ev, y_ev, cat_features,
                       best_params, FINAL_ITERS, verbose=200, task_type=task_type)

    cb.save_model(CATBOOST_PATH)
    best_auc = cb.get_best_score()["validation"]["AUC"]
    print(f"\n  CatBoost trained — best iter {cb.get_best_iteration()}  best eval AUC {best_auc:.4f}")
    print(f"  Model saved -> {CATBOOST_PATH}")

    # --- Baseline Logistic Regression ---
    # LR can't handle strings, so ordinal-encode cats just for the baseline
    print("\n  Training baseline LogisticRegression (uses ordinal-encoded cats)...")
    X_train_lr = X_train.copy()
    X_test_lr  = X_test.copy()
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_lr[cat_features] = ord_enc.fit_transform(X_train_lr[cat_features].astype(str))
    X_test_lr[cat_features]  = ord_enc.transform(X_test_lr[cat_features].astype(str))

    lr_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs",
                                     class_weight="balanced")),
    ])
    lr_pipeline.fit(X_train_lr, y_train)
    joblib.dump({"pipeline": lr_pipeline, "encoder": ord_enc, "cat_features": cat_features}, LR_PATH)
    print(f"  LR saved -> {LR_PATH}")

    print("\n  Training complete.")
    return cb, lr_pipeline, X_test, y_test, feature_names


if __name__ == "__main__":
    train_models()
