# train_lgb.py — LightGBM partner for the CatBoost+LGB+XGB rank-blend
# (FineHero AUC Playbook §4.2). Time-aware split, Optuna-tuned, ordinal-
# encoded categoricals, saves test-set predictions for src/ensemble.py.

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

FEATURES_PATH     = os.path.join(DATA_DIR, "features.csv")
CAT_FEATURES_PATH = os.path.join(MODELS_DIR, "cat_features.joblib")
LGB_MODEL_PATH    = os.path.join(MODELS_DIR, "lgb_model.txt")
LGB_PREDS_PATH    = os.path.join(MODELS_DIR, "lgb_test_preds.joblib")
LGB_ENCODER_PATH  = os.path.join(MODELS_DIR, "lgb_ord_encoder.joblib")
LGB_PARAMS_PATH   = os.path.join(MODELS_DIR, "lgb_best_params.joblib")

OPTUNA_TRIALS = 40
OPTUNA_ITERS  = 600
FINAL_ITERS   = 3000
EARLY_STOP    = 100


def _time_aware_split(df, test_frac=0.20, eval_frac_of_train=0.10):
    order = pd.to_datetime(df["issue_date"], errors="coerce").argsort()
    df = df.iloc[order].reset_index(drop=True)
    n = len(df)
    cut  = int(n * (1 - test_frac))
    cut2 = int(cut * (1 - eval_frac_of_train))
    return df.iloc[:cut2], df.iloc[cut2:cut], df.iloc[cut:]


def main():
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("  Loading features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  Dataset: {len(df):,} rows x {df.shape[1]} columns")

    cat_features = joblib.load(CAT_FEATURES_PATH)
    cat_features = [c for c in cat_features if c in df.columns]

    tr_df, ev_df, te_df = _time_aware_split(df)
    print(f"  Train: {len(tr_df):,}  |  Eval: {len(ev_df):,}  |  Test: {len(te_df):,}")

    y_tr, y_ev, y_te = tr_df["won"].values, ev_df["won"].values, te_df["won"].values
    drop_cols = ["won", "issue_date"]
    X_tr = tr_df.drop(columns=[c for c in drop_cols if c in tr_df.columns])
    X_ev = ev_df.drop(columns=[c for c in drop_cols if c in ev_df.columns])
    X_te = te_df.drop(columns=[c for c in drop_cols if c in te_df.columns])
    feature_names = list(X_tr.columns)

    for c in cat_features:
        X_tr[c] = X_tr[c].fillna("UNKNOWN").astype(str)
        X_ev[c] = X_ev[c].fillna("UNKNOWN").astype(str)
        X_te[c] = X_te[c].fillna("UNKNOWN").astype(str)

    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                              dtype=np.int32)
    X_tr[cat_features] = ord_enc.fit_transform(X_tr[cat_features])
    X_ev[cat_features] = ord_enc.transform(X_ev[cat_features])
    X_te[cat_features] = ord_enc.transform(X_te[cat_features])

    # LightGBM GPU (OpenCL) caps histogram bins at 255 per feature. Split cats
    # by cardinality: low-card columns stay as `categorical_feature`; high-card
    # ones stay in X as ordinal-encoded ints and get 255-bin quantized (their
    # signal is already captured by the plate/precinct/issuer_prior_* features).
    MAX_CAT_UNIQ = 255
    low_card_cats, high_card_cats = [], []
    for c in cat_features:
        if int(X_tr[c].nunique()) <= MAX_CAT_UNIQ:
            low_card_cats.append(c)
        else:
            high_card_cats.append(c)
    print(f"  Categoricals: {len(low_card_cats)} low-card, "
          f"{len(high_card_cats)} high-card (numeric on GPU): {high_card_cats}")
    cat_idx = [feature_names.index(c) for c in low_card_cats]

    # --- GPU detection ---
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    device = "gpu" if has_gpu else "cpu"
    print(f"  LightGBM device: {device}")

    base_params = {
        "objective":   "binary",
        "metric":      "auc",
        "verbosity":  -1,
        "is_unbalance": True,
        "device":      device,
        "max_bin":     255,
    }

    def objective(trial):
        params = {
            **base_params,
            "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":     5,
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
            "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
        }
        train_ds = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=False)
        eval_ds  = lgb.Dataset(X_ev, y_ev, categorical_feature=cat_idx, reference=train_ds,
                               free_raw_data=False)
        model = lgb.train(
            params, train_ds, num_boost_round=OPTUNA_ITERS,
            valid_sets=[eval_ds],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict(X_ev)
        return roc_auc_score(y_ev, pred)

    print(f"\n  Optuna search: {OPTUNA_TRIALS} trials x {OPTUNA_ITERS} iters")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    print(f"  Best eval AUC: {study.best_value:.4f}")
    print(f"  Best params:   {study.best_params}")

    best_params = {**base_params, **study.best_params, "bagging_freq": 5}
    joblib.dump(best_params, LGB_PARAMS_PATH)

    print(f"\n  Final fit: {FINAL_ITERS} iters, early_stop={EARLY_STOP}")
    train_ds = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=False)
    eval_ds  = lgb.Dataset(X_ev, y_ev, categorical_feature=cat_idx, reference=train_ds,
                           free_raw_data=False)
    model = lgb.train(
        best_params, train_ds, num_boost_round=FINAL_ITERS,
        valid_sets=[eval_ds],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(200)],
    )

    model.save_model(LGB_MODEL_PATH)
    joblib.dump(ord_enc, LGB_ENCODER_PATH)

    test_preds = model.predict(X_te)
    test_auc = roc_auc_score(y_te, test_preds)
    print(f"\n  LGB time-aware TEST AUC: {test_auc:.4f}")

    joblib.dump({
        "probs":         test_preds,
        "y_test":        y_te,
        "auc":           test_auc,
        "feature_names": feature_names,
    }, LGB_PREDS_PATH)
    print(f"  Test predictions saved -> {LGB_PREDS_PATH}")


if __name__ == "__main__":
    main()
