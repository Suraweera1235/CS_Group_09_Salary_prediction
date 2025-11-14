"""Train and save small models for the UI.

Creates:
 - models/rf_ui_model.joblib
 - models/rf_ui_model.pkl
 - models/linreg_ui_model.joblib
 - models/linreg_ui_model.pkl

This script mirrors the feature engineering used by the notebook so the Streamlit app can consume the saved pipelines directly.
"""
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor


def load_data():
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, 'data', 'Salary_Data.csv')
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    # simple winsorization
    for col in ['Age', 'Years of Experience', 'Salary']:
        if col in df.columns:
            lower, upper = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def build_features(df):
    X = df.copy()
    y = X.pop('Salary')
    X['exp_age_ratio'] = X['Years of Experience'] / (X['Age'] + 1)

    if 'Job Title' in X.columns:
        jt_means = X.join(y.rename('Salary')).groupby('Job Title')['Salary'].mean()
        X['job_title_te'] = X['Job Title'].map(jt_means).fillna(y.mean())
    else:
        X['job_title_te'] = y.mean()

    if 'Education Level' in X.columns:
        edu_order = X.join(y.rename('Salary')).groupby('Education Level')['Salary'].mean().sort_values().index.tolist()
        edu_map = {k: i for i, k in enumerate(edu_order)}
        X['education_ord'] = X['Education Level'].map(edu_map).fillna(-1)
    else:
        X['education_ord'] = 0

    return X, y


def train_and_save():
    df = load_data()
    X, y = build_features(df)

    numeric_cols = ['Age', 'Years of Experience', 'exp_age_ratio', 'job_title_te', 'education_ord']
    cat_cols = ['Gender'] if 'Gender' in X.columns else []

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('gender', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), cat_cols)
    ], remainder='drop')

    # Random Forest pipeline with log-target transform
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    ttr = TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)
    rf_pipe = Pipeline([('pre', preprocessor), ('rf_ttr', ttr)])
    rf_pipe.fit(X, y)

    # Linear baseline (no target transform)
    lin = LinearRegression()
    lin_pipe = Pipeline([('pre', preprocessor), ('lin', lin)])
    lin_pipe.fit(X, y)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    rf_joblib = os.path.join(models_dir, 'rf_ui_model.joblib')
    rf_pkl = os.path.join(models_dir, 'rf_ui_model.pkl')
    lin_joblib = os.path.join(models_dir, 'linreg_ui_model.joblib')
    lin_pkl = os.path.join(models_dir, 'linreg_ui_model.pkl')

    joblib.dump(rf_pipe, rf_joblib)
    joblib.dump(rf_pipe, rf_pkl)
    joblib.dump(lin_pipe, lin_joblib)
    joblib.dump(lin_pipe, lin_pkl)

    print('Saved models to', models_dir)


if __name__ == '__main__':
    train_and_save()
