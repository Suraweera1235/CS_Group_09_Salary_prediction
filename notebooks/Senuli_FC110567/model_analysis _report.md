<<<<<<< HEAD
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
=======
## Dataset Overview
- **Original rows:** 6,698  
- **Cleaned rows (after outlier removal):** 6,694  
- Outliers were removed using the IQR method on the `Salary` column, ensuring realistic salary values (above LKR 360,000).  

## Model and Methodology
- **Model:** Random Forest Regressor with 100 trees, max depth 10, and other tuned hyperparameters.  
- **Preprocessing:**  
  - Numerical features (`Age`, `Years of Experience`) were standardized.  
  - Categorical features (`Gender`, `Education Level`, `Job Title`) were one-hot encoded. Rare categories were grouped under "Other".  
- **Validation:** 5-fold cross-validation yielded **mean CV R² = 0.879 ± 0.013**, indicating strong predictive performance.  

## Model Performance

| Dataset   | RMSE (LKR)   | R²    | MAE (LKR)   |
|-----------|-------------|-------|-------------|
| Training  | 6,430,680   | 0.885 | 5,049,000   |
| Testing   | 6,548,760   | 0.881 | 5,168,160   |

- The model generalizes well, with minimal overfitting (training and testing R² are very close).  
- RMSE and MAE indicate that predicted salaries deviate on average by ~LKR 5–6.5 million, which is reasonable given the salary scale.  

## Visual Analysis
- **Actual vs Predicted Salaries:**  
  - The scatter plot shows that predictions closely follow the 45° line (perfect prediction line), confirming good model accuracy.  
- **Feature Importance:**  
  - Top predictors include `Years of Experience`, `Job Title` categories, and `Education Level`.  
  - Numerical features and key job roles heavily influence salary prediction, as expected.  

## New Prediction Example
- **Input:** 30 years old, 5 years of experience, Male, Bachelor, Data Analyst  
- **Predicted Salary:** LKR 20,246,400 (approx.)  

## Interpretation
- The model effectively captures the relationship between experience, education, job title, and salary.  
- Categorical variables such as `Job Title` and `Education Level` are crucial drivers of salary variation.  
- Outlier handling and stratified splitting help maintain model robustness and fairness.  

## Conclusion
The Random Forest model demonstrates strong predictive capability, with R² > 0.88 and low average errors. It can be reliably used for salary estimation for similar employee profiles.
>>>>>>> 99d3f85 (Update)
