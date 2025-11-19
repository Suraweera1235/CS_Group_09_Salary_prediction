import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
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


@st.cache_data
def build_label_mappings(df, categorical_cols=None):
    """Build mapping dicts similar to LabelEncoder for each categorical column.
    Returns a dict mapping column -> {category: int} and a dict of mode int values.
    """
    if categorical_cols is None:
        categorical_cols = ['Gender', 'Education Level', 'Job Title']
    maps = {}
    modes = {}
    for col in categorical_cols:
        # get unique classes in the same deterministic order LabelEncoder would (np.unique -> sorted)
        classes = list(np.unique(df[col].dropna().astype(str).values))
        mapping = {c: i for i, c in enumerate(classes)}
        maps[col] = mapping
        # compute mode; if mode not in mapping (unlikely) fall back to first class
        try:
            mode_val = df[col].mode().iloc[0]
            modes[col] = mapping.get(str(mode_val), 0)
        except Exception:
            modes[col] = 0
    return maps, modes

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
def clean_text(text):
    if isinstance(text, str):
        text = text.strip().lower()
        text = text.replace("’", "'")
        text = text.replace(" degree", "")  # Remove word 'degree'
        text = text.replace("bachelors", "bachelor's")
        text = text.replace("masters", "master's")
        text = text.replace("phd", "phd")
    return text

# ----------------- Streamlit App -----------------
def main():
    st.title('Salary Predictor')
    st.markdown("Enter feature values in the sidebar and select a model to predict Salary.")

    df = load_dataset()
    df['Education Level'] = df['Education Level'].apply(clean_text)
    mappings = build_mappings(df)
    # Build label-like mappings so we can transform categorical inputs to the numeric codes
    label_maps, label_modes = build_label_mappings(df)


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

        # Encode categorical columns using mappings built from the training CSV so the
        # feature names and numeric encodings match what the saved model expects.
        for col in ['Gender', 'Education Level', 'Job Title']:
            val = str(X.at[0, col]) if pd.notna(X.at[0, col]) else ''
            mapping = label_maps.get(col, {})
            mode_code = label_modes.get(col, 0)
            # map to code; if unseen category, use mode_code fallback
            X.at[0, col] = mapping.get(val, mode_code)

        # ensure numeric dtypes for all feature columns
        for c in X.columns:
            if c in ['Gender', 'Education Level', 'Job Title']:
                X[c] = X[c].astype(int)
            else:
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)

        # Ensure column order matches training notebook: Age, Gender, Education Level, Job Title, Years of Experience
        feature_order = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
        X = X[feature_order]

        try:
            pred = model.predict(X)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        st.success(f'Predicted Salary: LKR {pred:,.2f}')
        st.subheader('Inputs used')
        st.table(X.T)

if __name__ == '__main__':
    main()
