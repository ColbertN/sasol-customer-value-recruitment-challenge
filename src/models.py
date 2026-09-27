"""Model selection, post-processing and evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import f_classif, f_regression
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

USE_LGBM = HAS_LGBM and os.environ.get("SASOL_USE_LGBM", "0") == "1"


def weighted_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def _lgb_params(kind: str, candidate: int) -> dict:
    common = {"random_state": 2026, "n_jobs": 1, "verbosity": -1, "n_estimators": 250, "max_bin": 63, "force_col_wise": True}
    variants = [
        {"num_leaves": 31, "learning_rate": 0.035, "min_child_samples": 30, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
        {"num_leaves": 63, "learning_rate": 0.025, "min_child_samples": 45, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 4.0},
        {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0},
        {"num_leaves": 95, "learning_rate": 0.02, "min_child_samples": 70, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 8.0},
    ]
    p = {**common, **variants[candidate % len(variants)]}
    p["objective"] = "regression" if kind == "regression" else "multiclass"
    if kind == "classification":
        p["n_estimators"] = 220
    return p


def _fit_model(kind, X, y, params, eval_set=None, categorical=False):
    if USE_LGBM:
        cls = LGBMRegressor if kind == "regression" else LGBMClassifier
        model = cls(**params)
        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["callbacks"] = [early_stopping(80, verbose=False), log_evaluation(0)]
        model.fit(X, y, **fit_kwargs)
        return model
    if kind == "regression":
        model = GradientBoostingRegressor(n_estimators=60, learning_rate=0.04, max_depth=2, min_samples_leaf=15, subsample=0.9, random_state=2026, loss="huber")
    else:
        model = GradientBoostingClassifier(n_estimators=60, learning_rate=0.04, max_depth=2, min_samples_leaf=15, subsample=0.9, random_state=2026)
    model.fit(X, y)
    return model


def select_features(X_train: pd.DataFrame, y_train, kind: str, max_features: int, candidate: int = 0) -> tuple[list[str], pd.DataFrame]:
    """Select on training rows only, before the nonlinear model is fitted."""
    try:
        scores, _ = (f_regression(X_train, y_train) if kind == "regression" else f_classif(X_train, y_train))
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        scores = X_train.var(axis=0).to_numpy()
    imp = pd.DataFrame({"feature": X_train.columns, "importance": scores})
    imp = imp.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    keep = imp.loc[imp.importance > 0, "feature"].head(max_features).tolist()
    if not keep:
        keep = X_train.var().sort_values(ascending=False).head(max_features).index.tolist()
    return keep, imp


def tune_regression(X_train, y_train, X_valid, y_valid, feature_candidates: list[int], baseline: np.ndarray | None = None):
    best = None
    all_importance = None
    for cap in feature_candidates:
        selected, importance = select_features(X_train, y_train, "regression", min(cap, X_train.shape[1]), candidate=cap % 4)
        for candidate in range(1 if len(feature_candidates) == 1 else 3):
            model = _fit_model("regression", X_train[selected], y_train, _lgb_params("regression", candidate), eval_set=(X_valid[selected], y_valid))
            pred = np.maximum(0.0, model.predict(X_valid[selected]))
            for alpha in [0.90, 0.95, 1.00, 1.05, 1.10]:
                p = pred * alpha
                score = mean_squared_error(y_valid, p) ** 0.5
                if baseline is not None:
                    for blend in [0.0, 0.10, 0.20, 0.30]:
                        score_b = mean_squared_error(y_valid, blend * baseline + (1 - blend) * p) ** 0.5
                        if best is None or score_b < best["rmse"]:
                            best = {"rmse": float(score_b), "model": model, "features": selected, "alpha": alpha, "blend": blend, "candidate": candidate, "cap": cap}
                elif best is None or score < best["rmse"]:
                    best = {"rmse": float(score), "model": model, "features": selected, "alpha": alpha, "blend": 0.0, "candidate": candidate, "cap": cap}
        all_importance = importance if all_importance is None else all_importance.merge(importance, on="feature", how="outer", suffixes=("", "_new")).fillna(0.0)
    if all_importance is not None and "importance_new" in all_importance:
        imp_cols = [c for c in all_importance.columns if c.startswith("importance")]
        all_importance["importance"] = all_importance[imp_cols].sum(axis=1)
        all_importance = all_importance[["feature", "importance"]].sort_values("importance", ascending=False)
    return best, all_importance


def tune_classifier(X_train, y_train, X_valid, y_valid, feature_candidates: list[int]):
    best = None
    all_importance = None
    for cap in feature_candidates:
        selected, importance = select_features(X_train, y_train, "classification", min(cap, X_train.shape[1]), candidate=cap % 4)
        for candidate in range(1 if len(feature_candidates) == 1 else 3):
            model = _fit_model("classification", X_train[selected], y_train, _lgb_params("classification", candidate), eval_set=(X_valid[selected], y_valid))
            pred = model.predict(X_valid[selected]).astype(int)
            score = weighted_f1(y_valid, pred)
            if best is None or score > best["f1"]:
                best = {"f1": float(score), "model": model, "features": selected, "candidate": candidate, "cap": cap, "prior_gamma": 0.0}
        all_importance = importance if all_importance is None else all_importance.merge(importance, on="feature", how="outer", suffixes=("", "_new")).fillna(0.0)
    if all_importance is not None and "importance_new" in all_importance:
        imp_cols = [c for c in all_importance.columns if c.startswith("importance")]
        all_importance["importance"] = all_importance[imp_cols].sum(axis=1)
        all_importance = all_importance[["feature", "importance"]].sort_values("importance", ascending=False)
    return best, all_importance


def fit_final_regression(X, y, selected, candidate: int, n_estimators: int | None = None):
    params = _lgb_params("regression", candidate)
    if n_estimators is not None and HAS_LGBM:
        params["n_estimators"] = max(100, int(n_estimators))
    return _fit_model("regression", X[selected], y, params)


def fit_final_classifier(X, y, selected, candidate: int, n_estimators: int | None = None):
    params = _lgb_params("classification", candidate)
    if n_estimators is not None and HAS_LGBM:
        params["n_estimators"] = max(100, int(n_estimators))
    return _fit_model("classification", X[selected], y, params)


def save_importance(importance: pd.DataFrame, path: str | Path) -> None:
    importance.to_csv(path, index=False)


def save_metadata(meta: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
