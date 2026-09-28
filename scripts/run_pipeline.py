"""Run EDA, leakage-safe backtesting, tuning, final fitting and submission export."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from label_rules import load_transactions as load_label_transactions, raw_labels  # noqa: E402
from src.features import build_customer_features, load_transactions  # noqa: E402
from src.models import (  # noqa: E402
    fit_final_classifier,
    fit_final_regression,
    save_importance,
    save_metadata,
    tune_classifier,
    tune_regression,
    weighted_f1,
)


def _reg_baseline(features: pd.DataFrame, target: str, scale: float) -> np.ndarray:
    col = "90d_fuel_litres" if target == "CLV_fuel" else "90d_nonfuel_amount"
    raw = np.maximum(0.0, features[col].to_numpy())
    return np.log1p(raw) / scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "train.csv"))
    ap.add_argument("--test", default=str(ROOT / "test.csv"))
    ap.add_argument("--config", default=str(ROOT / "label_config.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--fast", action="store_true", help="Use the low-memory two-snapshot execution profile.")
    ap.add_argument("--full-compact", action="store_true", help="Use all five snapshots with the compact one-candidate profile.")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ["figures", "feature_importance", "predictions", "models"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tx = load_transactions(args.train)
    label_tx = load_label_transactions(args.train)
    test_ids = pd.read_csv(args.test, dtype={"ID": "string"})["ID"].astype("string")
    compact = args.fast or args.full_compact
    cutoffs = config["calibration_cutoffs"][-2:] if args.fast and not args.full_compact else config["calibration_cutoffs"]

    # Build a time-respecting training panel. Each row only sees history before its cutoff.
    panel_parts = []
    label_parts = []
    for cutoff in cutoffs:
        labels = raw_labels(label_tx, cutoff, config)
        labels["CLV_fuel"] = np.log1p(labels["fuel_litres"].to_numpy()) / config["normalization"]["CLV_fuel"]
        labels["CLV_nonfuel"] = np.log1p(labels["nonfuel_rand"].to_numpy()) / config["normalization"]["CLV_nonfuel"]
        labels["asof_month"] = pd.Timestamp(cutoff).to_period("M").ordinal
        f = build_customer_features(tx, cutoff, labels["ID"], config)
        panel_parts.append(f)
        label_parts.append(labels)
        print(f"snapshot {cutoff}: {len(f):,} customers")
    panel = pd.concat(panel_parts, ignore_index=True)
    labels = pd.concat(label_parts, ignore_index=True)
    panel = panel.merge(labels[["ID", "asof_month", "CLV_fuel", "CLV_nonfuel", "Opportunity"]], on=["ID", "asof_month"], how="left", validate="one_to_one")
    # ID is not a behavioural feature; the time snapshot is retained through asof_month.
    feature_cols = [c for c in panel.columns if c not in {"ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"}]
    if compact:
        key_prefixes = ("all_", "365d_", "180d_", "90d_", "30d_", "cat_all_amount_", "cat_all_baskets_", "cat_180d_amount_", "cat_180d_baskets_", "cat_90d_amount_", "cat_90d_baskets_", "cat_prev90d_", "cat_increment90d_", "cat_recent_presence_", "cat_historical_presence_", "month_", "history_days", "asof_month")
        feature_cols = [c for c in feature_cols if c.startswith(key_prefixes)]
    n_panel_rows, n_panel_columns = panel.shape
    # Materialize a compact feature matrix and release fragmented panel copies before tuning.
    test_features = build_customer_features(tx, "2025-12-01", test_ids, config)
    meta = panel[["ID", "asof_month", "CLV_fuel", "CLV_nonfuel", "Opportunity"]].copy()
    valid_ids = meta.loc[meta["asof_month"].eq(pd.Timestamp(cutoffs[-1]).to_period("M").ordinal), "ID"].to_numpy()
    validation_class_distribution = meta.loc[meta["asof_month"].eq(pd.Timestamp(cutoffs[-1]).to_period("M").ordinal), "Opportunity"].value_counts().to_dict()
    X = panel[feature_cols].astype("float32").copy()
    del panel_parts, label_parts, labels, panel, label_tx
    gc.collect()
    split_cutoff = cutoffs[-1]
    valid_mask = meta["asof_month"].eq(pd.Timestamp(split_cutoff).to_period("M").ordinal)
    train_mask = ~valid_mask
    X_train, X_valid = X.loc[train_mask], X.loc[valid_mask]

    metrics = {"calibration_cutoffs": cutoffs, "validation_cutoff": split_cutoff, "n_rows": int(n_panel_rows), "n_features": int(len(feature_cols))}
    importance_dir = out / "feature_importance"
    validation_predictions = {"ID": valid_ids}
    for target in ["CLV_fuel", "CLV_nonfuel"]:
        y_train = meta.loc[train_mask, target].to_numpy()
        y_valid = meta.loc[valid_mask, target].to_numpy()
        baseline = _reg_baseline(X.loc[valid_mask], target, config["normalization"][target])
        best, imp = tune_regression(X_train, y_train, X_valid, y_valid, [32] if compact else [48, 96], baseline=baseline)
        valid_pred = best["model"].predict(X_valid[best["features"]])
        valid_pred = best["blend"] * baseline + (1.0 - best["blend"]) * np.maximum(0.0, valid_pred * best["alpha"])
        validation_predictions[f"true_{target}"] = y_valid
        validation_predictions[f"pred_{target}"] = valid_pred
        metrics[target] = {"rmse": float(np.sqrt(np.mean((y_valid - valid_pred) ** 2))), "best": {k: v for k, v in best.items() if k not in {"model"}}, "baseline_rmse": float(np.sqrt(np.mean((y_valid - baseline) ** 2)))}
        if imp is not None:
            save_importance(imp, importance_dir / f"{target}_importance.csv")

    encoder = LabelEncoder()
    encoder.fit(config["opportunity_labels"])
    y_all = encoder.transform(meta["Opportunity"])
    y_train, y_valid = y_all[train_mask.to_numpy()], y_all[valid_mask.to_numpy()]
    best_cls, imp_cls = tune_classifier(X_train, y_train, X_valid, y_valid, [32] if compact else [48, 96])
    valid_cls = best_cls["model"].predict(X_valid[best_cls["features"]]).astype(int)
    metrics["Opportunity"] = {"weighted_f1": weighted_f1(y_valid, valid_cls), "best": {k: v for k, v in best_cls.items() if k != "model"}, "class_distribution": validation_class_distribution}
    if imp_cls is not None:
        save_importance(imp_cls, importance_dir / "Opportunity_importance.csv")

    # Refit each selected configuration on all available historical snapshots.
    X_all = X
    X_test = test_features[feature_cols].astype("float32")
    sub = pd.DataFrame({"ID": test_ids})
    for target in ["CLV_fuel", "CLV_nonfuel"]:
        info = metrics[target]["best"]
        model = fit_final_regression(X_all, meta[target].to_numpy(), info["features"], int(info["candidate"]), getattr(metrics[target]["best"], "best_iteration", None))
        pred = np.maximum(0.0, model.predict(X_test[info["features"]]) * float(info["alpha"]))
        baseline = _reg_baseline(test_features, target, config["normalization"][target])
        z = float(info["blend"]) * baseline + (1.0 - float(info["blend"])) * pred
        sub[target] = z
    cls_info = metrics["Opportunity"]["best"]
    cls_model = fit_final_classifier(X_all, y_all, cls_info["features"], int(cls_info["candidate"]), class_gamma=float(cls_info.get("class_gamma", 0.0)))
    cls_proba = cls_model.predict_proba(X_test[cls_info["features"]])
    cls_priors = np.bincount(y_all, minlength=cls_proba.shape[1]).astype(float)
    cls_priors = cls_priors / cls_priors.sum()
    cls_pred = np.argmax(cls_proba * np.power(np.maximum(cls_priors, 1e-9), float(cls_info.get("prior_gamma", 0.0))), axis=1).astype(int)
    sub["Opportunity"] = encoder.inverse_transform(cls_pred)
    sub = sub[["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]]
    sub.to_csv(out / "predictions" / "submission.csv", index=False, float_format="%.17g")
    validation_predictions["true_opportunity"] = encoder.inverse_transform(y_valid)
    validation_predictions["pred_opportunity"] = encoder.inverse_transform(valid_cls)
    pd.DataFrame(validation_predictions).to_csv(out / "validation_predictions.csv", index=False)
    metrics["submission_rows"] = int(len(sub))
    metrics["submission_path"] = str(out / "predictions" / "submission.csv")
    save_metadata(metrics, out / "metrics.json")
    (out / "panel_shape.txt").write_text(f"{n_panel_rows} rows x {n_panel_columns} columns\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
