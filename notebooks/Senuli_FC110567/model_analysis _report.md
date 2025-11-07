# Model analysis — Random Forest (short & actionable)

This short report summarizes the Random Forest pipeline and explains how we can interpret the key metrics (R2, RMSE, MAE, CV) together with concise recommendations.

## 1) What the pipeline does here (high level)
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
