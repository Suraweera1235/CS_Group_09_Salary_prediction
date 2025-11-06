# Model analysis — Random Forest (short & actionable)

This short report summarizes the Random Forest pipeline and explains how to interpret the key metrics (R2, RMSE, MAE, CV) together with concise recommendations.

## 1) What the pipeline does (high level)
- Data cleaning: drop NA + winsorization (1% / 99%) on numeric columns.
- Features: Age, Years of Experience, engineered features: exp_age_ratio, job_title_te (target-mean), education_ord.
- Preprocessing: KBinsDiscretizer for Age, StandardScaler for numeric engineered features, OneHotEncoder for Gender when present.
- Model: RandomForestRegressor wrapped in TransformedTargetRegressor (log1p / expm1) to stabilize variance of Salary.
- Hyperparameter search: RandomizedSearchCV over sensible RF params with 5-fold KFold CV. Final pipeline saved to `models/rf_alt_model.joblib`.

## 2) How to interpret key metrics (quick rules)
- R2: proportion of variance explained by the model.
  - ~1.0 = near-perfect (rare). 0.5–0.8 = often useful on noisy targets. 0 = no better than mean. <0 = worse than mean.
- RMSE / MAE: absolute error (same units as Salary). Compare them to mean/median salary to judge practical impact.
  - Use MAE for robust, RMSE to penalize large errors.
- CV R2 (cv_scores.mean()): best estimator of out-of-sample performance. Prefer this over single test-split R2.

## 3) Common patterns we should check for
- train R2 ≫ test R2: likely overfitting -> regularize (reduce max_depth, increase min_samples_leaf), reduce features, or get more data.
- CV R2 ≈ test R2: good, test split is representative.
- CV R2 ≫ test R2: possible leakage or unlucky test split — check preprocessing and stratify behavior.
- Low R2 but small RMSE relative to salary: model may be practically acceptable; always contextualize errors with salary scale.

## 4) Preprocessing impact & risks
- Log-target transform: reduces heteroscedasticity, usually improves R2 for skewed salaries.
- Target-mean encoding (`job_title_te`): often boosts R2 but can introduce optimistic bias if not implemented CV-safely. Current approach maps training means to test — better to use K-fold target encoding inside the pipeline.
- Winsorization: reduces influence of extremes and may lower RMSE, but can hide meaningful outliers; run a sensitivity check without winsorization.
- KBinsDiscretizer for Age: helpful if age relationship is non-linear, but increases dimensionality; for small samples check bin counts.

## 5) Concrete, actionable recommendations (priority order)
1. Print and record the numeric metrics now (train/test R2, RMSE, MAE, and CV R2 mean/std) to ground decisions. Example snippet to add and run in the notebook:

```python
from pprint import pprint
pprint({
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'cv_mean_r2': cv_scores.mean(),
    'cv_std_r2': cv_scores.std()
})
```

2. Replace global target-mean encoding with a CV-safe implementation (K-fold target encoding or category_encoders.TargetEncoder used inside a cross-validation-friendly wrapper) and re-run the RandomizedSearchCV.

3. Compute feature importance (tree importances and permutation importance) and partial dependence for the top 3 features (job_title_te, exp_age_ratio, education_ord) to validate that engineered features truly add value.

4. Validate the saved pipeline: load `models/rf_alt_model.joblib` and predict on `X_test`. Confirm the test metrics match the notebook output to ensure serialization fidelity.

## 6) Quick validation checklist
- [ ] Print and save the numeric metrics (train/test R2, RMSE, MAE, CV mean/std).
- [ ] Run permutation importance on the test set to verify feature contributions.
- [ ] Implement CV-safe target-encoding and re-evaluate CV R2.
- [ ] Sensitivity test: rerun pipeline without winsorization and compare metrics.
- [ ] Create a small inference script to load `models/rf_alt_model.joblib` and run a quick sanity-check prediction.

## 7) If R2 is low — prioritized diagnostics
1. Check label noise and distribution: plot Salary vs top numeric features and inspect by Job Title / Education Level.
2. Try a gradient-boosting model (XGBoost / LightGBM) with the same preprocessing; often better for small/structured datasets.
3. Compare to simple baselines (mean predictor, linear regression) to confirm Random Forest adds value.

---
Report created/updated programmatically. If you want, I can now (pick one):
- add the small code cell that prints numeric metrics into the notebook; or
- implement CV-safe target-encoding inside the pipeline and re-run the search; or
- create a small `scripts/validate_model.py` that loads the saved pipeline and prints test metrics.

End of report.
