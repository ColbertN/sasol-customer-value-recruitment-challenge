# Sasol Customer Value Recruitment Challenge

## Project purpose

This project builds a customer-value prediction system for the Sasol Customer Value Recruitment Challenge.

The business question is:

> Given everything we know about a customer from their past Sasol transactions, what are they likely to do in the next three months, and what value or opportunity could that create?

For every test customer, the pipeline predicts:

1. `CLV_fuel`: expected fuel litres in the next three months.
2. `CLV_nonfuel`: expected non-fuel spend in the next three months.
3. `Opportunity`: the most likely customer opportunity, such as inactivity, stability, fuel growth, or adoption of a new category.

These predictions can support customer segmentation, targeted promotions, retention activity, and better allocation of commercial effort. The competition score combines the quality of both customer-value forecasts and the opportunity classification, so the solution treats the task as a multi-output machine-learning problem rather than optimizing only one target.

This repository contains the full modelling lifecycle: data understanding, exploratory analysis, target construction, feature engineering, leakage-safe validation, model selection, post-processing, final training, feature importance, and submission generation.

The raw competition files are intentionally not committed to GitHub. They are supplied locally by the competition and must not be shared publicly.

## Competition context

The source data is transaction-level loyalty data. One customer can have many transactions, and one transaction can contain multiple line items. The historical period runs from 19 March 2024 to 30 November 2025. The test cutoff is 1 December 2025, so the final prediction window is the following three calendar months.

The modelling data contains 467,880 line items, 365,638 baskets, 6,204 customers, and 390 locations. The final test file contains 5,488 customers.

The official competition rules allow the supplied datasets and open-source tools. This project therefore uses no external commercial, web, demographic, geospatial, or economic data. See the [official Zindi competition page](https://zindi.world/competitions/sasol-customer-value-recruitment-challenge).

## Machine-learning lifecycle

### 1. Define the prediction problem

The target is the customer's behaviour during the three months after an as-of date. The model must only use information available before that as-of date.

The problem has two regression targets and one multiclass classification target:

| Output | Type | Meaning |
|---|---|---|
| `CLV_fuel` | Regression | Normalized `log(1 + fuel litres)` in the next three months |
| `CLV_nonfuel` | Regression | Normalized `log(1 + non-fuel rand spend)` in the next three months |
| `Opportunity` | Classification | The most important customer opportunity in the next three months |

The regression targets are modelled separately because fuel volume and shop/non-fuel spending describe different customer behaviours. The opportunity target is modelled separately because it is categorical and evaluated with weighted F1.

### 2. Understand and validate the source data

The pipeline reads the supplied transaction file while preserving important source values such as literal `NA` and `Unknown` categories. It converts dates into a consistent time representation and validates numeric precision before aggregation.

Important data decisions are explicit in the code and label documentation:

- A basket is identified by `TransactionId`.
- Multiple lines belonging to one basket are preserved as line-item information.
- Repeated basket totals are deduplicated before basket-level statistics are calculated.
- `TotalExclAmount` is used directly for store spend; it is not rebuilt from unit price.
- `LineItemSequence` and non-standard source sequences are retained as valid behavioural signals.
- Adjustment items are excluded from target and opportunity activity but remain available for behavioural analysis.
- Genuine returns remain signed and reduce net customer value.

This prevents common transaction-data errors such as double-counting a basket total or silently treating meaningful category values as missing data.

### 3. Reproduce the competition targets

The target-generation logic is implemented in [`label_rules.py`](label_rules.py) and configured in [`label_config.json`](label_config.json). [`LABEL_RULES.md`](LABEL_RULES.md) documents the rules in detail.

For every historical as-of date:

1. Customers with at least three previous distinct baskets become eligible.
2. Historical features are calculated strictly before the cutoff.
3. Outcomes are calculated from the following three calendar months.
4. Fuel litres are calculated from qualifying Fuel/Sale_Fuel quantities.
5. Non-fuel spend is calculated from qualifying non-fuel source amounts.
6. Growth requires previous category activity, a future purchase, at least R20 incremental spend, and at least 25% growth over the previous quarter.
7. Adoption requires no previous qualifying purchase, a future purchase, and positive future spend.
8. The largest category-level monetary increment determines the opportunity winner before output categories are mapped to labels.
9. Customers with no qualifying outcome-window purchase are labelled `Inactivity`; customers with activity but no qualifying growth/adoption are `Stable`.

The fixed public normalization constants are applied as `log(1 + target) / scale`. They are not recalculated from the final test period.

### 4. Explore the data before modelling

The EDA stage checks the shape, date range, customer activity, basket behaviour, category mix, fuel/non-fuel spend, missing-value conventions, customer concentration, and opportunity-class imbalance.

The main findings are:

- Fuel is the dominant activity type, but non-fuel spend contains important customer-value information.
- Customer activity is highly skewed: a small number of customers contribute many baskets and transactions.
- The opportunity classes are imbalanced, with `Stable`, `Inactivity`, and existing Fuel growth forming the largest groups.
- Basket-level totals must be deduplicated because they appear on multiple line items.
- Recency, visit cadence, category breadth, and recent-quarter behaviour are likely to be more predictive than raw identifiers.

The complete visual report is available in [`outputs/eda_summary.md`](outputs/eda_summary.md).

![Monthly activity](outputs/figures/01_monthly_activity.png)

![Category spend](outputs/figures/02_category_spend.png)

![Customer distributions](outputs/figures/03_customer_distributions.png)

![Opportunity distribution](outputs/figures/04_opportunity_distribution.png)

### 5. Engineer customer-level features

The transaction table is transformed into one row per eligible customer per as-of date. Features describe behaviour over multiple historical windows: all history, 365 days, 180 days, 90 days, and 30 days.

Feature groups include:

- transaction, line, basket, quantity, and spend totals;
- fuel and non-fuel activity;
- basket frequency and basket-value statistics;
- recency, active days, active months, weekday/weekend and hour patterns;
- location breadth and concentration;
- product, category, subcategory, and category-level breadth;
- original category spend and basket activity;
- recent-quarter versus previous-quarter category increments;
- monthly activity and momentum signals;
- line-item sequence, unit-price, modifier, and adjustment behaviour.

The customer ID is not used as a predictive feature. The as-of month is retained so the model can learn time-related patterns without seeing future outcomes.

### 6. Prevent target leakage with time-aware validation

Random train/test splitting would be inappropriate because transactions are time ordered and customers can appear in several historical snapshots. Instead, the pipeline builds five historical as-of panels:

`2024-09-01`, `2024-12-01`, `2025-03-01`, `2025-06-01`, and `2025-09-01`.

For each snapshot, features use only transactions before that cutoff. Labels use only the following three months. The latest snapshot, `2025-09-01`, is held out as a realistic validation period. The label merge uses both `ID` and `asof_month`, preventing labels from one snapshot from being joined to another snapshot for the same customer.

This validation design is intended to answer the real question: how well does the model predict a future customer period when trained on earlier periods?

### 7. Select features and train models

Feature selection is performed on the training rows only using univariate F-statistics. The selected features are then passed to the nonlinear model; the validation period is never used to select features.

The final checked-in profile uses:

- separate regression models for fuel and non-fuel value;
- a multiclass CatBoost classifier for `Opportunity`;
- a compact selected feature set of up to 32 features per target;
- deterministic random seeds and one CatBoost thread for reproducibility;
- class weighting during opportunity-model tuning to account for class imbalance;
- serial gradient-boosting fallback when CatBoost is not enabled.

CatBoost was selected because it provides strong nonlinear modelling for mixed behavioural signals while remaining practical on the customer-level feature panel. The fallback keeps the pipeline runnable in more limited environments.

### 8. Tune and post-process predictions

Model selection is performed on the held-out historical snapshot. Regression predictions are checked across scale adjustments and optional baseline blending, then clipped at zero because negative customer value is not meaningful for the submission.

For the opportunity classifier, the pipeline evaluates class-weight strengths and probability-prior adjustments using weighted F1. This is important because a classifier can obtain a deceptively reasonable accuracy while ignoring smaller but commercially meaningful opportunity classes.

Feature-importance files are exported for each target so the modelling decisions can be audited:

- [`CLV_fuel_importance.csv`](outputs/feature_importance/CLV_fuel_importance.csv)
- [`CLV_nonfuel_importance.csv`](outputs/feature_importance/CLV_nonfuel_importance.csv)
- [`Opportunity_importance.csv`](outputs/feature_importance/Opportunity_importance.csv)

### 9. Refit on all historical calibration data

After selecting the modelling configuration, each model is refit using all available historical snapshots. The final test features are built using only transactions before 1 December 2025. Predictions are exported with the exact required columns and label spelling.

The generated file is:

`outputs/predictions/submission.csv`

### 10. Evaluate and communicate the result

The repository reports both model performance and baseline performance. These are local time-based validation results, not a guarantee of the hidden Zindi leaderboard score.

| Target | Final validation result | Baseline |
|---|---:|---:|
| `CLV_fuel` | **0.6077 RMSE** | 0.6871 RMSE |
| `CLV_nonfuel` | **0.7529 RMSE** | 0.9271 RMSE |
| `Opportunity` | **0.4717 weighted F1** | - |

The final export contains 5,488 test customers. Full run metadata is available in [`outputs/metrics.json`](outputs/metrics.json), and the visual analysis is available in [`outputs/eda_summary.md`](outputs/eda_summary.md).

## Repository structure

```text
.
├── data/                         # Optional local data location; raw competition files are ignored
├── outputs/
│   ├── eda_summary.md            # EDA narrative
│   ├── figures/                  # EDA charts
│   ├── feature_importance/       # Training-only importance rankings
│   └── metrics.json              # Validation metrics and selected settings
├── scripts/
│   ├── run_eda.py                # Create EDA tables and figures
│   └── run_pipeline.py           # Build, validate, train, and export predictions
├── src/
│   ├── features.py               # Leakage-safe customer feature engineering
│   └── models.py                 # Selection, tuning, fitting, and metrics
├── label_rules.py                # Authoritative target construction
├── label_config.json             # Approved label and normalization configuration
└── LABEL_RULES.md                # Human-readable target definition
```

## Reproduce the project

Place the supplied `train.csv`, `test.csv`, and `SampleSubmission.csv` files in the project root, then run:

```powershell
python -m pip install -r requirements.txt
python scripts/run_eda.py
$env:SASOL_USE_CATBOOST='1'
python scripts/run_pipeline.py --full-compact
```

Useful outputs are written under `outputs/`. The raw data, local validation predictions, and submission file are ignored by Git so that competition data is not published accidentally.

## Project outcome

The result is a complete, reproducible customer analytics workflow that turns raw transaction records into customer-level value forecasts and actionable opportunity segments. Each stage is documented, validated, and connected to a clear business purpose, making the project suitable for review, extension, and practical customer-value analysis.
