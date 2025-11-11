import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ----------------- Caching functions -----------------
@st.cache_data
def load_dataset():
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "data", "Salary_Data.csv")
    return pd.read_csv(path)

@st.cache_data
def build_mappings(df):
    mappings = {}
    # Job Title encoding
    mappings['job_title_mean'] = df.groupby('Job Title')['Salary'].mean().to_dict()
    mappings['job_title_global_mean'] = df['Salary'].mean()
    # Education ordinal mapping
    edu_means = df.groupby('Education Level')['Salary'].mean().sort_values()
    mappings['education_map'] = {k: i for i, k in enumerate(edu_means.index.tolist())}
    # Gender options
    mappings['genders'] = sorted(df['Gender'].dropna().unique().tolist())
    return mappings

# ----------------- Model loading -----------------
def list_models():
    base = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base, 'models')
    if not os.path.exists(models_dir):
        return [], models_dir
    files = [f for f in os.listdir(models_dir) if f.endswith(('.joblib', '.pkl'))]
    return files, models_dir

def load_model(path):
    return joblib.load(path)

# ----------------- Feature engineering -----------------
def compute_features(input_vals, mappings=None):
    row = {
        'Age': float(input_vals.get('Age', 0)),
        'Years of Experience': float(input_vals.get('Years of Experience', 0)),
        'Gender': input_vals.get('Gender', ''),
        'Education Level': input_vals.get('Education Level', ''),
        'Job Title': input_vals.get('Job Title', '')
    }
    return pd.DataFrame([row])

# ----------------- Streamlit App -----------------
def main():
    st.title('Salary Predictor')
    st.markdown("Enter feature values in the sidebar and select a model to predict Salary.")

    df = load_dataset()
    mappings = build_mappings(df)

    # ----------------- Sidebar -----------------
    st.sidebar.header('Model selection')
    files, models_dir = list_models()
    model_choice = None

    if files:
        selected_file = st.sidebar.selectbox('Choose saved model', ['-- none --'] + files)
        if selected_file != '-- none --':
            model_choice = os.path.join(models_dir, selected_file)

    uploaded = st.sidebar.file_uploader('Or upload a model', type=['joblib', 'pkl'])
    if uploaded:
        temp_path = os.path.join('/tmp', uploaded.name)
        with open(temp_path, 'wb') as f:
            f.write(uploaded.getbuffer())
        model_choice = temp_path

    # ----------------- Inputs -----------------
    st.sidebar.header('Input features')
    age = st.sidebar.number_input('Age', 0, 120, 30)
    years_exp = st.sidebar.number_input('Years of Experience', 0, 80, 5)

    job_titles = ['-- Other --'] + sorted(df['Job Title'].dropna().unique())
    job_title = st.sidebar.selectbox('Job Title', job_titles)
    if job_title == '-- Other --':
        job_title = st.sidebar.text_input('Enter Job Title', '')

    edus = ['-- Unknown --'] + sorted(df['Education Level'].dropna().unique())
    education = st.sidebar.selectbox('Education Level', edus)
    if education == '-- Unknown --':
        education = ''

    genders = ['-- Unknown --'] + mappings['genders']
    gender = st.sidebar.selectbox('Gender', genders)
    if gender == '-- Unknown --':
        gender = ''

    input_vals = {
        'Age': age,
        'Years of Experience': years_exp,
        'Job Title': job_title,
        'Education Level': education,
        'Gender': gender
    }

    # ----------------- Prediction -----------------
    if st.sidebar.button('Predict'):
        if not model_choice:
            st.error("No model selected or uploaded!")
            return

        try:
            model = load_model(model_choice)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

        X = compute_features(input_vals, mappings)

        try:
            pred = model.predict(X)[0]
        except Exception:
            # fallback: pass only numeric columns
            pred = model.predict(X.select_dtypes(include=[np.number]))[0]

        st.success(f'Predicted Salary: LKR {pred:,.2f}')
        st.subheader('Inputs used')
        st.table(X.T)

if __name__ == '__main__':
    main()
