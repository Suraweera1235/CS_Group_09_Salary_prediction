import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("trained_model.pkl")

# ---- Custom UI Styling ----
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

# ---- App Header ----
st.title("💼 Salary Prediction System")
st.markdown("### Predict an employee's salary based on their profile")


# Sidebar input fields
st.sidebar.header("Input Employee Details")

# Input fields
age = st.sidebar.slider("Age", 18, 65, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor’s", "Master’s", "PhD"])
job_title = st.sidebar.selectbox("Job Title", [
    "Software Engineer", "Data Scientist", "Project Manager",
    "Business Analyst", "Sales Executive", "HR Manager"
])
experience = st.sidebar.slider("Years of Experience", 0, 40, 2)

# Convert categorical values
gender_encoded = 1 if gender == "Male" else 0

education_map = {
    "High School": 0,
    "Bachelor’s": 1,
    "Master’s": 2,
    "PhD": 3
}
education_encoded = education_map[education]

job_map = {
    "Software Engineer": 0,
    "Data Scientist": 1,
    "Project Manager": 2,
    "Business Analyst": 3,
    "Sales Executive": 4,
    "HR Manager": 5
}
job_encoded = job_map[job_title]

# Create DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_encoded],
    "Education Level": [education_encoded],
    "Job Title": [job_encoded],
    "Years of Experience": [experience]
})

st.write("### Input Summary")
st.dataframe(input_data)

# Predict
if st.button("Predict Salary 💰"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Salary: ${prediction:,.2f}")
