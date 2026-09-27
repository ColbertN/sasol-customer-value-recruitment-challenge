"""Leakage-safe customer feature engineering for the Sasol challenge."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


BASE_COLUMNS = [
    "DateTimeZA",
    "TransactionId",
    "LineItemSequence",
    "LocationId",
    "ID",
    "LineItemType",
    "TransactionTotalInclAmount",
    "StockItemId",
    "ItemName",
    "ItemCategoryLevel1",
    "ItemCategoryLevel2",
    "ItemCategoryLevel3",
    "Quantity",
    "TotalExclAmount",
    "TotalModifierAmount",
    "UnitSellingExclPrice",
]


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load the released transaction file without destroying literal NA/Unknown categories."""
    numeric_columns = {"TransactionTotalInclAmount", "Quantity", "TotalExclAmount", "TotalModifierAmount", "UnitSellingExclPrice"}
    dtype = {c: "string" for c in BASE_COLUMNS if c not in numeric_columns}
    dtype.update({c: "float64" for c in numeric_columns})
    d = pd.read_csv(path, dtype=dtype, keep_default_na=False)
    d["_time"] = pd.to_datetime(d["DateTimeZA"], format="mixed", errors="raise")
    for c in ["ItemName", "ItemCategoryLevel2", "ItemCategoryLevel3", "ItemCategoryLevel1"]:
        d[c] = d[c].fillna("").astype("string")
    d["ID"] = d["ID"].astype("string")
    d["TransactionId"] = d["TransactionId"].astype("string")
    d["LineItemSequence"] = pd.to_numeric(d["LineItemSequence"], errors="coerce").fillna(0.0)
    return d


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _add_grouped(out: pd.DataFrame, grouped: pd.DataFrame, prefix: str, ids: pd.Index) -> None:
    grouped = grouped.reindex(ids).fillna(0.0)
    grouped.columns = [f"{prefix}{c}" for c in grouped.columns]
    out[grouped.columns] = grouped


def _window_features(
    h: pd.DataFrame,
    out: pd.DataFrame,
    ids: pd.Index,
    cutoff: pd.Timestamp,
    prefix: str,
    lower: pd.Timestamp | None,
    included: set[str],
    excluded_nonfuel: set[str],
) -> None:
    """Add aggregate features for a single as-of window."""
    w = h if lower is None else h[h["_time"] >= lower]
    if w.empty:
        return
    clean = w[~w["_is_adjustment"]]
    scope = clean[clean["ItemCategoryLevel2"].isin(included)]
    g = clean.groupby("ID", sort=False)
    basic = g.agg(
        lines=("TransactionId", "size"),
        baskets=("TransactionId", "nunique"),
        amount=("TotalExclAmount", "sum"),
        quantity=("Quantity", "sum"),
        locations=("LocationId", "nunique"),
        products=("StockItemId", "nunique"),
        categories=("ItemCategoryLevel2", "nunique"),
        subcategories=("ItemCategoryLevel3", "nunique"),
        category_level1=("ItemCategoryLevel1", "nunique"),
        line_types=("LineItemType", "nunique"),
        line_sequence_mean=("LineItemSequence", "mean"),
        line_sequence_max=("LineItemSequence", "max"),
        unit_price_mean=("UnitSellingExclPrice", "mean"),
        unit_price_std=("UnitSellingExclPrice", "std"),
        modifier_amount=("TotalModifierAmount", "sum"),
        active_days=("_date", "nunique"),
        active_months=("_month", "nunique"),
    )
    _add_grouped(out, basic, prefix, ids)

    sg = scope.groupby("ID", sort=False)
    scoped = sg.agg(
        qualifying_lines=("TransactionId", "size"),
        qualifying_baskets=("TransactionId", "nunique"),
        scoped_amount=("TotalExclAmount", "sum"),
        scoped_quantity=("Quantity", "sum"),
    )
    _add_grouped(out, scoped, prefix, ids)

    fuel = clean[clean["ItemCategoryLevel2"].eq("Fuel")]
    fg = fuel.groupby("ID", sort=False)
    _add_grouped(out, fg.agg(fuel_litres=("Quantity", "sum"), fuel_amount=("TotalExclAmount", "sum"), fuel_baskets=("TransactionId", "nunique")), prefix, ids)
    nonfuel = scope[~scope["ItemCategoryLevel2"].eq("Fuel")]
    ng = nonfuel.groupby("ID", sort=False)
    _add_grouped(out, ng.agg(nonfuel_amount=("TotalExclAmount", "sum"), nonfuel_quantity=("Quantity", "sum"), nonfuel_baskets=("TransactionId", "nunique")), prefix, ids)

    # Basket-level statistics: TransactionTotalInclAmount is repeated on lines and is
    # therefore deduplicated before it is used.
    basket = clean[["ID", "TransactionId", "_time", "TransactionTotalInclAmount"]].drop_duplicates(["ID", "TransactionId"])
    bg = basket.groupby("ID", sort=False)["TransactionTotalInclAmount"]
    basket_stats = pd.DataFrame({
        "basket_total_sum": bg.sum(),
        "basket_total_mean": bg.mean(),
        "basket_total_median": bg.median(),
        "basket_total_max": bg.max(),
        "basket_total_std": bg.std().fillna(0.0),
    })
    _add_grouped(out, basket_stats, prefix, ids)

    # Temporal and repeat-visit signals.
    temporal = clean.groupby("ID", sort=False).agg(
        mean_hour=("_hour", "mean"),
        weekend_share=("_is_weekend", "mean"),
        mean_dayofweek=("_dow", "mean"),
        last_seen=("_time", "max"),
    )
    temporal["recency_days"] = (cutoff - temporal.pop("last_seen")).dt.total_seconds() / 86400.0
    _add_grouped(out, temporal, prefix, ids)

    # Store concentration is useful for distinguishing habitual and roaming customers.
    loc_counts = clean.groupby(["ID", "LocationId"], sort=False).size()
    loc_share = loc_counts.groupby(level=0).max() / clean.groupby("ID", sort=False)["TransactionId"].nunique()
    _add_grouped(out, loc_share.to_frame("top_location_share"), prefix, ids)

    # Adjustment behaviour is retained as a separate signal; adjustments never enter
    # target or opportunity calculations.
    adj = w[w["_is_adjustment"]].groupby("ID", sort=False).agg(adjustment_lines=("TransactionId", "size"), adjustment_amount=("TotalExclAmount", "sum"))
    _add_grouped(out, adj, prefix, ids)


def build_customer_features(
    transactions: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    ids: Iterable[str],
    config: dict,
) -> pd.DataFrame:
    """Build only information available strictly before ``cutoff``."""
    cutoff = pd.Timestamp(cutoff)
    ids = pd.Index(pd.Series(list(ids), dtype="string").drop_duplicates(), name="ID")
    h = transactions[transactions["_time"] < cutoff].copy()
    h["_date"] = h["_time"].dt.date
    h["_month"] = h["_time"].dt.to_period("M").astype(str)
    h["_hour"] = h["_time"].dt.hour + h["_time"].dt.minute / 60.0
    h["_dow"] = h["_time"].dt.dayofweek
    h["_is_weekend"] = h["_dow"] >= 5
    h["_is_adjustment"] = h["ItemName"].isin(config["adjustment_items"])
    included = {k for k, v in config["category_mapping"].items() if v["included"]}
    excluded_nonfuel = {"Value added services", "Print media", "Carrier Bags"}

    out = pd.DataFrame(index=ids)
    out["asof_month"] = cutoff.to_period("M").ordinal
    out["history_days"] = (cutoff - h.groupby("ID")["_time"].min()).dt.total_seconds().div(86400).reindex(ids).fillna(0.0)
    windows = {"all": None, "365d": cutoff - pd.Timedelta(days=365), "180d": cutoff - pd.Timedelta(days=180), "90d": cutoff - pd.Timedelta(days=90), "30d": cutoff - pd.Timedelta(days=30)}
    for name, lower in windows.items():
        _window_features(h, out, ids, cutoff, f"{name}_", lower, included, excluded_nonfuel)

    # Category-level intensity and basket breadth, retaining original categories
    # before any output-label mapping is applied.
    cats = sorted(included)
    clean = h[~h["_is_adjustment"]]
    for window_name, lower in {"all": None, "180d": cutoff - pd.Timedelta(days=180), "90d": cutoff - pd.Timedelta(days=90)}.items():
        w = clean if lower is None else clean[clean["_time"] >= lower]
        for metric, series in {
            "amount": w["TotalExclAmount"],
            "baskets": w["TransactionId"],
        }.items():
            if metric == "baskets":
                tab = w[w["ItemCategoryLevel2"].isin(cats)].groupby(["ID", "ItemCategoryLevel2"])["TransactionId"].nunique().unstack(fill_value=0)
            else:
                tab = w[w["ItemCategoryLevel2"].isin(cats)].groupby(["ID", "ItemCategoryLevel2"])[series.name].sum().unstack(fill_value=0)
            tab = tab.reindex(index=ids, columns=cats, fill_value=0.0)
            tab.columns = [f"cat_{window_name}_{metric}_{str(c).replace(' ', '_').replace('&', 'and')}" for c in cats]
            out[tab.columns] = tab

    # Monthly seasonality and momentum for the last 12 calendar months.
    for month_back in range(6):
        month_start = (cutoff.to_period("M") - month_back - 1).start_time
        month_end = month_start + pd.offsets.MonthBegin(1)
        m = clean[(clean["_time"] >= month_start) & (clean["_time"] < month_end)]
        for name, subset, col in [
            ("amount", m, "TotalExclAmount"),
            ("nonfuel_amount", m[m["ItemCategoryLevel2"].isin(included - {"Fuel"})], "TotalExclAmount"),
            ("baskets", m, "TransactionId"),
        ]:
            if name == "baskets":
                s = subset.groupby("ID")[col].nunique()
            else:
                s = subset.groupby("ID")[col].sum()
            out[f"month_{month_back}_{name}"] = s.reindex(ids).fillna(0.0)

    # The raw engineered signals are retained without duplicating every column with a
    # log transform: LightGBM's histogram splits already handle the heavy tails, and
    # keeping one representation makes the final model lighter and easier to audit.
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    out.index.name = "ID"
    return out.reset_index()
