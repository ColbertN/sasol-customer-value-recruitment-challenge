"""Create reproducible EDA charts and a concise findings report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from label_rules import load_transactions as load_label_transactions, raw_labels  # noqa: E402
from src.features import load_transactions  # noqa: E402


def main() -> None:
    out = ROOT / "outputs"
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tx = load_transactions(ROOT / "train.csv")
    label_tx = load_label_transactions(ROOT / "train.csv")
    config = json.loads((ROOT / "label_config.json").read_text(encoding="utf-8"))
    clean = tx[~tx.ItemName.isin(config["adjustment_items"])].copy()
    clean["month"] = clean["_time"].dt.to_period("M").astype(str)
    clean["in_scope"] = clean.ItemCategoryLevel2.isin([k for k, v in config["category_mapping"].items() if v["included"]])

    monthly = clean.groupby("month").agg(lines=("ID", "size"), customers=("ID", "nunique"), baskets=("TransactionId", "nunique"), amount=("TotalExclAmount", "sum"))
    ax = monthly[["lines", "customers", "baskets"]].plot(figsize=(11, 5), title="Monthly transaction activity")
    ax.set_xlabel("Month"); ax.set_ylabel("Count"); ax.grid(alpha=.2); ax.figure.tight_layout(); ax.figure.savefig(fig_dir / "01_monthly_activity.png", dpi=150); plt.close(ax.figure)

    cat = clean[clean.in_scope].groupby("ItemCategoryLevel2")["TotalExclAmount"].sum().sort_values()
    ax = cat.plot.barh(figsize=(10, 7), title="Historical source amount by label-rule category")
    ax.set_xlabel("TotalExclAmount (source units)"); ax.grid(axis="x", alpha=.2); ax.figure.tight_layout(); ax.figure.savefig(fig_dir / "02_category_spend.png", dpi=150); plt.close(ax.figure)

    cust = clean.groupby("ID").agg(baskets=("TransactionId", "nunique"), amount=("TotalExclAmount", "sum"), lines=("ID", "size"))
    ax = np.log1p(cust[["baskets", "amount"]]).plot.hist(alpha=.7, bins=40, figsize=(10, 5), title="Log-scaled customer activity distributions")
    ax.grid(alpha=.2); ax.figure.tight_layout(); ax.figure.savefig(fig_dir / "03_customer_distributions.png", dpi=150); plt.close(ax.figure)

    all_labels = pd.concat([raw_labels(label_tx, c, config) for c in config["calibration_cutoffs"]], ignore_index=True)
    opp = all_labels.Opportunity.value_counts().sort_values()
    ax = opp.plot.barh(figsize=(10, 7), title="Opportunity labels across five calibration snapshots")
    ax.set_xlabel("Customer snapshots"); ax.grid(axis="x", alpha=.2); ax.figure.tight_layout(); ax.figure.savefig(fig_dir / "04_opportunity_distribution.png", dpi=150); plt.close(ax.figure)

    missing = tx.isna().mean().sort_values(ascending=False)
    missing_positive = missing[missing > 0]
    if missing_positive.empty:
        fig_missing, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No parser-level missing values after preserving literal categories", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax = missing_positive.plot.bar(figsize=(9, 4), title="Missing-value rate in released transactions")
        ax.set_ylabel("Fraction of rows"); ax.grid(axis="y", alpha=.2)
        fig_missing = ax.figure
    fig_missing.tight_layout(); fig_missing.savefig(fig_dir / "05_missingness.png", dpi=150); plt.close(fig_missing)

    top_month = monthly.loc[monthly.amount.idxmax()]
    report = [
        "# EDA findings",
        "",
        f"- The release contains **{len(tx):,} line items**, **{tx.TransactionId.nunique():,} baskets**, **{tx.ID.nunique():,} customers**, and **{tx.LocationId.nunique():,} locations**.",
        f"- The history spans **{tx._time.min():%Y-%m-%d} to {tx._time.max():%Y-%m-%d}**; the latest complete history ends immediately before the 2025-12-01 test cutoff.",
        f"- The busiest month by source amount is **{monthly.amount.idxmax()}** ({top_month.amount:,.0f} source units).",
        f"- Fuel is the dominant line-item category, so the model separates fuel litres from source monetary amounts and does not sum repeated basket totals.",
        f"- The five label snapshots generate **{len(all_labels):,} customer-snapshot rows**. The most frequent opportunity is **{all_labels.Opportunity.mode().iat[0]}**.",
        "- Literal `NA` and `Unknown` categories are kept as real activity, while adjustment items and the three excluded categories are handled exactly according to the released rules.",
        "",
        "![Monthly activity](figures/01_monthly_activity.png)",
        "![Category spend](figures/02_category_spend.png)",
        "![Customer distributions](figures/03_customer_distributions.png)",
        "![Opportunity distribution](figures/04_opportunity_distribution.png)",
        "![Missingness](figures/05_missingness.png)",
    ]
    (out / "eda_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    monthly.to_csv(out / "monthly_eda.csv")
    cat.to_csv(out / "category_eda.csv", header=["source_amount"])
    print("EDA written to", out)


if __name__ == "__main__":
    main()
