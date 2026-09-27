# Sasol Customer Value Recruitment Challenge

A reproducible, leakage-safe customer value pipeline for the Sasol loyalty transaction challenge. It predicts the next three months of fuel litres, non-fuel spend, and the combined opportunity label.

The public repository contains the code, EDA figures, validation metrics, and feature importance outputs. The released transaction files are intentionally not committed: the challenge rules say that the challenge data may not be shared outside the challenge.

## Results

The checked-in run uses the resource-safe `--fast` profile: the two most recent calibration snapshots, 228 engineered features, training-only feature selection to 32 features per target, serial gradient boosting, recent-quarter baseline blending, and regression scale calibration.

Held-out validation is the 2025-09-01 snapshot:

| Target | Model RMSE / F1 | Recent-quarter baseline |
|---|---:|---:|
| CLV_fuel | **0.4790 RMSE** | 0.4953 RMSE |
| CLV_nonfuel | **0.6156 RMSE** | 0.6678 RMSE |
| Opportunity | **0.4566 weighted F1** | — |

The final export contains 5,488 test customers in the required columns: `ID`, `CLV_fuel`, `CLV_nonfuel`, and `Opportunity`. Regression predictions are already in the challenge's normalized log scale using the fixed public constants.

The full five-snapshot profile is available with `python scripts/run_pipeline.py`. On a stronger competition machine, set `SASOL_USE_LGBM=1` to activate the optional LightGBM path; the default local engine is a deterministic serial gradient-boosting fallback for portability.

## EDA highlights

- The release contains 467,880 line items, 365,638 baskets, 6,204 historical customers, and 390 locations.
- The transaction history spans 2024-03-19 through 2025-11-30; the test cutoff is 2025-12-01.
- Fuel is the dominant category, so fuel litres and non-fuel source amounts are modeled separately.
- The opportunity classes are highly imbalanced: Stable, Existing-category growth: Fuel, and Inactivity dominate the historical snapshots. Weighted F1 is therefore evaluated with the exact challenge labels rather than accuracy.
- Basket totals are deduplicated before aggregation; source `TotalExclAmount` is used directly for store spend; adjustment items and excluded categories follow the supplied label rules.
- Literal `NA` and `Unknown` remain real categories. Krispy Kreme, Car Wash, and other mapped categories compete on their original category increments before output relabeling.

The full visual report is in [`outputs/eda_summary.md`](outputs/eda_summary.md), with the main plots below.

![Monthly activity](outputs/figures/01_monthly_activity.png)

![Category spend](outputs/figures/02_category_spend.png)

![Customer distributions](outputs/figures/03_customer_distributions.png)

![Opportunity distribution](outputs/figures/04_opportunity_distribution.png)

## Modeling design

1. Build five leakage-safe as-of customer panels from the supplied calibration cutoffs.
2. Reproduce targets with the authoritative `label_rules.py` and fixed normalization constants.
3. Aggregate basket-safe customer behavior over all history and 365/180/90/30-day windows.
4. Add category breadth, category spend/basket signals, recency, visit cadence, location concentration, monthly momentum, basket statistics, fuel/non-fuel measures, and the participant-discussed line-item fields.
5. Select features using training-only F-statistics, tune candidate feature caps/model configurations on the latest historical snapshot, and blend the selected model with a recent-quarter baseline.
6. Refit on the available calibration panel, clip normalized regression outputs at zero, and export the exact sample-submission schema.

The participant clarifications were used directly: multiple lines per basket are preserved, repeated basket totals are never summed, non-standard line sequences are treated as valid source data, and `UnitSellingExclPrice` is not used to reconstruct label amounts.

## Reproduce

Place the challenge files in the project root, then run:

```powershell
python -m pip install -r requirements.txt
python scripts/run_eda.py
python scripts/run_pipeline.py --fast
```

The verified artifacts are written under `outputs/`:

- `metrics.json` — validation scores and selected post-processing settings
- `feature_importance/` — target-specific training-only importance rankings
- `figures/` and `eda_summary.md` — EDA report
- `predictions/submission.csv` — local submission export (ignored from the public repository)

No external data is used. The challenge rules permit only the released datasets, so external commercial, web, or geospatial enrichment would risk disqualification.
