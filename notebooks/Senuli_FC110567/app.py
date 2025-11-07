import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("trained_model.pkl")

# ---- Styling ----
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
        }
        h1 {
            color: #0072B5;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Salary Prediction System")
st.markdown("### Predict an employee's salary based on their profile")

# ---- Sidebar Inputs ----
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor’s", "Master’s", "PhD", "Other"])
job_title = st.sidebar.selectbox("Job Title", [
    "Software Engineer", "Data Scientist", "Project Manager",
    "Business Analyst", "Sales Executive", "HR Manager", "Other"
])
experience = st.sidebar.slider("Years of Experience", 0, 40, 2)

# ---- Prepare input ----
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job_title],
    "Years of Experience": [experience]
})

st.markdown("### Input Summary")
st.dataframe(input_data)

# ---- Prediction ----
if st.button("Predict Salary 💰"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Salary: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
