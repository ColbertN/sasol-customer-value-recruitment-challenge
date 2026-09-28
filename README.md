# Sasol Customer Value Recruitment Challenge

A reproducible, leakage-safe customer value pipeline for the Sasol loyalty transaction challenge. It predicts the next three months of fuel litres, non-fuel spend, and the combined opportunity label.

The public repository contains the code, EDA figures, validation metrics, and feature-importance outputs. The released transaction files are intentionally not committed because the challenge rules prohibit sharing challenge data outside the challenge.

## Results

The checked-in run uses the full five-snapshot compact profile: 455 engineered features, training-only feature selection to 32 features per target, deterministic serial CatBoost, class-weight tuning, probability-prior calibration, and leakage-safe refitting.

Held-out validation is the 2025-09-01 snapshot:

| Target | Model RMSE / F1 | Recent-quarter baseline |
|---|---:|---:|
| CLV_fuel | **0.6077 RMSE** | 0.6871 RMSE |
| CLV_nonfuel | **0.7529 RMSE** | 0.9271 RMSE |
| Opportunity | **0.4717 weighted F1** | - |

The final export contains 5,488 test customers in the required columns: `ID`, `CLV_fuel`, `CLV_nonfuel`, and `Opportunity`. Regression predictions are already in the challenge normalized log scale using the fixed public constants.

The full profile is available with `SASOL_USE_CATBOOST=1 python scripts/run_pipeline.py --full-compact`. CatBoost is optional and runs with one thread for reproducibility; without it the pipeline falls back to serial gradient boosting. LightGBM is also supported through `SASOL_USE_LGBM=1` when the environment can run it reliably.

## EDA highlights

- The release contains 467,880 line items, 365,638 baskets, 6,204 historical customers, and 390 locations.
- The transaction history spans 2024-03-19 through 2025-11-30; the test cutoff is 2025-12-01.
- Fuel is the dominant category, so fuel litres and non-fuel source amounts are modeled separately.
- The opportunity classes are highly imbalanced: Stable, Existing-category growth: Fuel, and Inactivity dominate the historical snapshots. Weighted F1 is evaluated with the exact challenge labels rather than accuracy.
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
5. Select features using training-only F-statistics, tune candidate feature caps, CatBoost configurations, class weights, and class-prior post-processing on the latest historical snapshot.
6. Refit on the available calibration panel, clip normalized regression outputs at zero, and export the exact sample-submission schema.

The participant clarifications were used directly: multiple lines per basket are preserved, repeated basket totals are never summed, non-standard line sequences are treated as valid source data, and `UnitSellingExclPrice` is not used to reconstruct label amounts.

## Reproduce

Place the challenge files in the project root, then run:

```powershell
python -m pip install -r requirements.txt
python scripts/run_eda.py
$env:SASOL_USE_CATBOOST='1'
python scripts/run_pipeline.py --full-compact
```

The verified artifacts are written under `outputs/`:

- `metrics.json` - validation scores and selected post-processing settings
- `feature_importance/` - target-specific training-only importance rankings
- `figures/` and `eda_summary.md` - EDA report
- `predictions/submission.csv` - local submission export, ignored from the public repository

No external data is used. The challenge rules permit only the released datasets, so external commercial, web, or geospatial enrichment would risk disqualification.
