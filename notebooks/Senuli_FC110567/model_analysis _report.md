
# Model analysis — Random Forest (short & actionable)
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

