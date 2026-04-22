# train_xgb.py — XGBoost partner for the CatBoost+LGB+XGB rank-blend
# (FineHero AUC Playbook §4.2). Time-aware split, Optuna-tuned, ordinal-
# encoded categoricals w/ enable_categorical=True.

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
XGB_MODEL_PATH    = os.path.join(MODELS_DIR, "xgb_model.json")
XGB_PREDS_PATH    = os.path.join(MODELS_DIR, "xgb_test_preds.joblib")
XGB_ENCODER_PATH  = os.path.join(MODELS_DIR, "xgb_ord_encoder.joblib")
XGB_PARAMS_PATH   = os.path.join(MODELS_DIR, "xgb_best_params.joblib")

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
    import xgboost as xgb
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

    # Mark categoricals explicitly so XGBoost uses its native categorical split finder
    for c in cat_features:
        X_tr[c] = X_tr[c].astype("category")
        X_ev[c] = X_ev[c].astype("category")
        X_te[c] = X_te[c].astype("category")

    # --- GPU detection ---
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    device = "cuda" if has_gpu else "cpu"
    print(f"  XGBoost device: {device}")

    pos, neg = int(y_tr.sum()), len(y_tr) - int(y_tr.sum())
    scale_pos_weight = neg / max(pos, 1)

    base_params = {
        "objective":          "binary:logistic",
        "eval_metric":        "auc",
        "tree_method":        "hist",
        "device":             device,
        "enable_categorical": True,
        "scale_pos_weight":   scale_pos_weight,
    }

    def objective(trial):
        params = {
            **base_params,
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        }
        dtr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
        dev = xgb.DMatrix(X_ev, y_ev, enable_categorical=True)
        model = xgb.train(
            params, dtr, num_boost_round=OPTUNA_ITERS,
            evals=[(dev, "eval")],
            early_stopping_rounds=50, verbose_eval=False,
        )
        pred = model.predict(dev)
        return roc_auc_score(y_ev, pred)

    print(f"\n  Optuna search: {OPTUNA_TRIALS} trials x {OPTUNA_ITERS} iters")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    print(f"  Best eval AUC: {study.best_value:.4f}")
    print(f"  Best params:   {study.best_params}")

    best_params = {**base_params, **study.best_params}
    joblib.dump(best_params, XGB_PARAMS_PATH)

    print(f"\n  Final fit: {FINAL_ITERS} iters, early_stop={EARLY_STOP}")
    dtr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
    dev = xgb.DMatrix(X_ev, y_ev, enable_categorical=True)
    dte = xgb.DMatrix(X_te, y_te, enable_categorical=True)
    model = xgb.train(
        best_params, dtr, num_boost_round=FINAL_ITERS,
        evals=[(dev, "eval")],
        early_stopping_rounds=EARLY_STOP, verbose_eval=200,
    )

    model.save_model(XGB_MODEL_PATH)
    joblib.dump(ord_enc, XGB_ENCODER_PATH)

    test_preds = model.predict(dte)
    test_auc = roc_auc_score(y_te, test_preds)
    print(f"\n  XGB time-aware TEST AUC: {test_auc:.4f}")

    joblib.dump({
        "probs":         test_preds,
        "y_test":        y_te,
        "auc":           test_auc,
        "feature_names": feature_names,
    }, XGB_PREDS_PATH)
    print(f"  Test predictions saved -> {XGB_PREDS_PATH}")


if __name__ == "__main__":
    main()
