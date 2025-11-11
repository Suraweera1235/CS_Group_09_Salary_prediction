import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def load_dataset():
    # data is stored at project_root/data/Salary_Data.csv
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "data", "Salary_Data.csv")
    return pd.read_csv(path)


@st.cache_data
def build_mappings(df):
    mappings = {}
    if 'Job Title' in df.columns:
        mappings['job_title_mean'] = df.groupby('Job Title')['Salary'].mean().to_dict()
        mappings['job_title_global_mean'] = df['Salary'].mean()
    else:
        mappings['job_title_mean'] = {}
        mappings['job_title_global_mean'] = df['Salary'].mean()

    if 'Education Level' in df.columns:
        edu_means = df.groupby('Education Level')['Salary'].mean().sort_values()
        edu_map = {k: i for i, k in enumerate(edu_means.index.tolist())}
        mappings['education_map'] = edu_map
    else:
        mappings['education_map'] = {}

    if 'Gender' in df.columns:
        mappings['genders'] = sorted(df['Gender'].dropna().unique().tolist())
    else:
        mappings['genders'] = []

    return mappings


def list_models():
    base = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base, 'models')
    if not os.path.exists(models_dir):
        return []
    files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') or f.endswith('.pkl')]
    return files, models_dir


def load_model(path):
    return joblib.load(path)


def compute_features(input_vals, mappings):
    # input_vals: dict with 'Age', 'Years of Experience', 'Job Title', 'Education Level', 'Gender'
    age = float(input_vals.get('Age', 0.0))
    exp = float(input_vals.get('Years of Experience', 0.0))
    exp_age_ratio = exp / (age + 1.0)

    job_title = input_vals.get('Job Title', '')
    job_map = mappings.get('job_title_mean', {})
    job_te = job_map.get(job_title, mappings.get('job_title_global_mean', 0.0))

    edu = input_vals.get('Education Level', '')
    edu_map = mappings.get('education_map', {})
    education_ord = edu_map.get(edu, -1)

    row = {
        'Age': age,
        'Years of Experience': exp,
        'exp_age_ratio': exp_age_ratio,
        'job_title_te': job_te,
        'education_ord': education_ord
    }

    # include Gender if provided
    if 'Gender' in input_vals:
        row['Gender'] = input_vals.get('Gender')

    return pd.DataFrame([row])


def main():
    st.title('Salary Predictor — Random Forest UI')

    st.markdown(
        """
        Enter feature values on the left. Select a model from `models/` or upload your own joblib model.\n
        The app will compute simple engineered features (experience/age ratio, job-title mean encoding, education ordinal) using training data statistics and then run the selected model to predict Salary.
        """
    )

    df = load_dataset()
    mappings = build_mappings(df)

    st.sidebar.header('Model selection')
    files, models_dir = list_models()
    model_choice = None
    if files:
        selected_file = st.sidebar.selectbox('Choose saved model', ['-- none --'] + files)
        if selected_file and selected_file != '-- none --':
            model_choice = os.path.join(models_dir, selected_file)

    uploaded = st.sidebar.file_uploader('Or upload a joblib model (.joblib/.pkl)', type=['joblib', 'pkl'])
    if uploaded is not None:
        # save temporary uploaded file
        temp_path = os.path.join('/tmp', uploaded.name)
        with open(temp_path, 'wb') as f:
            f.write(uploaded.getbuffer())
        model_choice = temp_path

    st.sidebar.markdown('---')
    st.sidebar.header('Input features')
    # basic inputs
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
    years_exp = st.sidebar.number_input('Years of Experience', min_value=0.0, max_value=80.0, value=5.0)

    # job title
    if 'Job Title' in df.columns:
        job_titles = sorted(df['Job Title'].dropna().unique().tolist())
        job_title = st.sidebar.selectbox('Job Title', ['-- Other / custom --'] + job_titles)
        if job_title == '-- Other / custom --':
            job_title = st.sidebar.text_input('Enter Job Title (custom)', '')
    else:
        job_title = st.sidebar.text_input('Job Title', '')

    # education
    if 'Education Level' in df.columns:
        edus = sorted(df['Education Level'].dropna().unique().tolist())
        education = st.sidebar.selectbox('Education Level', ['-- Unknown --'] + edus)
        if education == '-- Unknown --':
            education = ''
    else:
        education = st.sidebar.text_input('Education Level', '')

    # gender
    if mappings.get('genders'):
        gender = st.sidebar.selectbox('Gender', ['-- Unknown --'] + mappings['genders'])
        if gender == '-- Unknown --':
            gender = ''
    else:
        gender = st.sidebar.text_input('Gender (optional)', '')

    input_vals = {
        'Age': age,
        'Years of Experience': years_exp,
        'Job Title': job_title,
        'Education Level': education,
        'Gender': gender
    }

    st.sidebar.markdown('---')
    if st.sidebar.button('Predict'):
        if model_choice is None:
            st.error('No model selected or uploaded. Place a joblib model in the `models/` folder or upload one.')
            return

        try:
            model = load_model(model_choice)
        except Exception as e:
            st.error(f'Failed to load model: {e}')
            return

        # compute features and predict
        X = compute_features(input_vals, mappings)

        # Ensure columns order/appearance: model pipeline may expect certain columns
        try:
            pred = model.predict(X)[0]
        except Exception as e:
            st.warning('Direct prediction failed — attempting to pass only numeric columns used by pipeline.')
            # try a safer fallback: select intersection of required features if possible
            try:
                pred = model.predict(X.select_dtypes(include=[np.number]))[0]
            except Exception as e2:
                st.error(f'Prediction failed: {e2}')
                return

        st.success(f'Predicted Salary: {pred:,.2f}')

        # show model info
        st.subheader('Model info')
        try:
            if hasattr(model, 'get_params'):
                params = model.get_params()
                # show only top-level params to keep UI tidy
                top_params = {k: params[k] for k in list(params)[:10]}
                st.json(top_params)
            else:
                st.write(type(model))
        except Exception:
            st.write('Could not extract model params')

        st.subheader('Inputs used for prediction')
        st.table(X.T)


if __name__ == '__main__':
    main()
