# Analytics Module

## Goal

This module loads the Titanic dataset once, profiles it, handles missing values according to project rules, performs EDA, then builds a predictive modeling workflow on the same cleaned data.

## Data loading and offline fallback

The raw data is loaded with `sns.load_dataset('titanic')` exactly once and immediately saved as `analytics/titanic.csv` with `df.to_csv(..., index=False)`. This committed CSV becomes the offline fallback and the single continuation point for all downstream modeling work.

## Missing-value handling decisions

The workflow follows the threshold rule stated in the task:

- under 5% missing: drop affected rows
- between 5% and 30%: impute
- above 30% or unreliable: drop the column or use a missing-category strategy and justify it

For this dataset, `deck` was dropped because its missing rate was well above 30% and it carries structural missingness that would be unreliable to impute. `age` was imputed using the median, while rows missing `embarked` were removed because the fraction missing was tiny and the rows were not informative enough to discard at a larger rate.

## Univariate findings

The EDA reports the IQR-based outlier counts for both `age` and `fare`, plus the mean, median, and mode of `fare`. The `fare` distribution is right-skewed because the mean is higher than the median, and the mode is lower than both.

## Bivariate and multivariate story

The module calculates survival rates by sex, by passenger class, and by sex + class, then produces a 6-column correlation matrix restricted to:

- `survived`
- `pclass`
- `age`
- `sibsp`
- `parch`
- `fare`

The strongest off-diagonal correlations are reported in the notebook/script output and used to support the narrative about who was more likely to survive.

## Modeling decisions

A stratified train/test split is used before any preprocessing because the target is imbalanced and stratification preserves the class distribution in both sets. Preprocessing is fit only on the training slice, then applied to the test slice in transform-only mode. The models evaluated are Logistic Regression, Decision Tree, and Random Forest.

## Run

From the repo root:

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

## Outputs

- `analytics/titanic.csv`
- `analytics/output/*.png` chart files for EDA and model diagnostics
- `analytics/best_pipeline.joblib` — complete fitted preprocessing + estimator pipeline

## Final recommendation

The best-performing classifier is selected based on the comparison table in the model script and written recommendation. The saved pipeline is reloadable with `joblib.load(...)` and is designed to work on raw, unpreprocessed input.
