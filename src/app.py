# salary_app_ui.py
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


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

    # Gender options (keep only Male, Female)
    allowed_genders = ["Male", "Female"]
    df = df[df['Gender'].isin(allowed_genders)]

    mappings['genders'] = sorted(df['Gender'].unique().tolist())

    return mappings


def list_models():
    base = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base, 'models')
    if not os.path.exists(models_dir):
        return [], models_dir
    files = [f for f in os.listdir(models_dir) if f.endswith(('.joblib', '.pkl'))]
    return files, models_dir


def load_model(path):
    return joblib.load(path)


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


muted = "#6c757d"           
primary_text = "#ffffff"     
secondary_text = "#000000"   
card_bg_dark = "#1e1e1e"
card_bg_light = "#ffffff"

def inject_css(dark_mode: bool):
  
    bg = "#0b1220" if dark_mode else "#f6f7fb"
    surface = "rgba(13,18,28,0.65)" if dark_mode else "rgba(255,255,255,0.9)"
    card_border = "rgba(255,255,255,0.06)" if dark_mode else "#e6e9ef"
    text = "#e6eef8" if dark_mode else "#0b1220"
    muted = "#9aa4b2" if dark_mode else "#64748b"
    accent = "#5b21b6"  
    accent_light = "rgba(91,33,182,0.10)"

    st.markdown(
        f"""
    <style>
    /* Root page */
    .stApp {{
        background: {bg};
        color: {text};
        font-family: "Segoe UI", Roboto, -apple-system, "Helvetica Neue", Arial;
    }}

    /* Header */
    .header {{
        background: {surface};
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.25);
        margin-bottom: 18px;
        animation: fadeIn 0.6s ease;
    }}
    .brand-badge {{
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background: linear-gradient(135deg, {accent}, #8b5cf6);
        display:flex;
        align-items:center;
        justify-content:center;
        color: white;
        font-weight:700;
        box-shadow: 0 6px 18px rgba(91,33,182,0.14);
    }}
    .brand-title {{
        font-size:18px;
        font-weight:700;
        background: linear-gradient(90deg, {accent}, #8b5cf6);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }}
    .brand-sub {{
        color: {muted};
        font-size:12px;
    }}

    /* Layout containers */
    .glass {{
        background: {surface};
        border-radius: 14px;
        padding: 18px;
        border: 1px solid {card_border};
        box-shadow: 0 6px 24px rgba(2,6,23,0.25);
        animation: fadeInUp 0.6s ease;
    }}

    /* Scrollable horizontal role cards */
    .role-strip {{
        overflow-x: auto;
        white-space: nowrap;
        padding-bottom: 8px;
        margin-top: 6px;
    }}
    .role-card {{
        display:inline-block;
        width: 220px;
        margin-right: 12px;
        padding: 12px;
        border-radius: 10px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
        border: 1px solid {card_border};
        vertical-align: top;
        transition: transform 260ms ease, box-shadow 260ms ease;
    }}
    .role-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 12px 30px rgba(2,6,23,0.45);
    }}
    .role-emoji {{
        font-size: 28px;
        display:block;
        margin-bottom:8px;
    }}
    .role-name {{ font-weight:700; font-size:14px; }}
    .role-desc {{ font-size:12px; color: {muted}; margin-top:4px; }}

    /* Form inputs styling (light-weight) */
    .streamlit-expanderHeader:focus {{ outline: none; }}
    .stButton>button {{
        background: linear-gradient(90deg, {accent}, #8b5cf6);
        border: none;
        color: white;
        padding: 10px 14px;
        border-radius: 10px;
        font-weight: 700;
    }}

    /* Result card */
    .result-card {{
        background: linear-gradient(180deg, rgba(0,0,0,0.24), rgba(0,0,0,0.12));
        border-radius: 14px;
        padding: 28px;
        position: relative;
        overflow: hidden;
    }}
    .result-amount {{
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -1px;
    }}
    .result-sub {{ color: {muted}; font-size: 13px; }}

    /* decorative blobs (pure css) */
    .blob {{
      position:absolute;
      width: 200px;
      height: 200px;
      border-radius: 50%;
      filter: blur(36px);
      opacity: 0.12;
      transform: translateZ(0);
    }}
    .blob-1 {{ background: linear-gradient(90deg, #8b5cf6, #5b21b6); top:-40px; right:-30px; }}
    .blob-2 {{ background: linear-gradient(90deg, #06b6d4, #7c3aed); bottom:-50px; left:-40px; opacity:0.08; }}

    /* small responsive tweaks */
    @media (max-width: 880px) {{
        .role-card {{ width: 180px; }}
        .result-amount {{ font-size: 36px; }}
    }}

    /* Animations */
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(6px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeInUp {{
        0% {{ opacity: 0; transform: translateY(18px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Keep streamlit native containers from overriding */
    .stSidebar .stButton>button {{ width:100%; }}

    </style>
    """,
        unsafe_allow_html=True
    )


def svg_icon(name: str, size: int = 18):
    icons = {
        "dollar": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 1v22" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'.format(s=size),
        "briefcase": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="7" width="18" height="12" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M8 7V5a2 2 0 012-2h4a2 2 0 012 2v2" stroke="currentColor" stroke-width="1.5"/></svg>'.format(s=size),
        "calendar": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="5" width="18" height="16" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M16 3v4M8 3v4" stroke="currentColor" stroke-width="1.5"/></svg>'.format(s=size),
        "user": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="8" r="3" stroke="currentColor" stroke-width="1.5"/><path d="M5.5 20a6.5 6.5 0 0113 0" stroke="currentColor" stroke-width="1.5"/></svg>'.format(s=size),
        "graduation": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 3L2 8l10 5 10-5-10-5z" stroke="currentColor" stroke-width="1.5"/><path d="M2 17l10 5 10-5" stroke="currentColor" stroke-width="1.5"/><path d="M7 10v7" stroke="currentColor" stroke-width="1.5"/></svg>'.format(s=size)
    }
    return icons.get(name, "")



def main():
    
    st.set_page_config(page_title="Salary Predictor", layout="wide", initial_sidebar_state="expanded")
    
    with st.sidebar:
        st.header("Settings")
        dark_mode = st.checkbox("Dark mode", value=False, help="Toggle dark/light styling")
        st.markdown("---")

    inject_css(dark_mode)

    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(
            """
            <div class="header">
                <div class="brand-badge">{dollar}</div>
                <div style="display:flex; flex-direction:column;">
                    <div class="brand-title">SalaryAI</div>
                    <div class="brand-sub">Professional salary estimator</div>
                </div>
            </div>
            """.format(dollar=svg_icon("dollar", size=20)),
            unsafe_allow_html=True,
        )
    with col2:
       
        st.write("")

    
    df = load_dataset()
    df['Education Level'] = df['Education Level'].apply(clean_text)
    mappings = build_mappings(df)



    st.markdown("</div></div>", unsafe_allow_html=True)
    st.write("")  
   
    left_col, right_col = st.columns([7, 5], gap="large")

   
    with left_col:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Profile Details")
        st.markdown("<div style='color: #9aa4b2; font-size:13px'>Enter candidate information to estimate annual salary.</div>", unsafe_allow_html=True)
        st.write("")

        
        files, models_dir = list_models()
        model_choice = None
        st.markdown("---")
        st.markdown("**Model selection**")
        if files:
            selected_file = st.selectbox("Saved model (choose)", ["-- none --"] + files)
            if selected_file != '-- none --':
                model_choice = os.path.join(models_dir, selected_file)
        uploaded = st.file_uploader("Or upload a model (.joblib or .pkl)", type=['joblib', 'pkl'])
        if uploaded:
            tmp_path = os.path.join("/tmp", uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            model_choice = tmp_path

        st.markdown("---")

        
        if 'form_defaults' not in st.session_state:
            st.session_state.form_defaults = {
                'Age': 30,
                'Years of Experience': 5,
                'Job Title': df['Job Title'].dropna().iloc[0] if len(df['Job Title'].dropna())>0 else '',
                'Education Level': df['Education Level'].dropna().iloc[0] if len(df['Education Level'].dropna())>0 else '',
                'Gender': mappings['genders'][0] if mappings['genders'] else '',
                'Custom Job Title': ''
            }

        
        job_titles = ['-- Other --'] + sorted(df['Job Title'].dropna().unique())

        
        with st.form("predict_form"):
            
            a1, a2 = st.columns(2)
            with a1:
                age = st.number_input("Age", min_value=18, max_value=100, value=int(st.session_state.form_defaults['Age']))
            with a2:
                years_exp = st.slider("Years of Experience", min_value=0.0, max_value=50.0, value=float(st.session_state.form_defaults['Years of Experience']), step=0.5)

           
            st.write("")
            j1, j2 = st.columns([3,1])
            with j1:
                job_title = st.selectbox("Job Title", job_titles, index=0 if st.session_state.form_defaults['Job Title'] == '' else 1)
                custom_job_title = ""
                if job_title == '-- Other --':
                    custom_job_title = st.text_input("Enter job title", value=st.session_state.form_defaults['Custom Job Title'])
            with j2:
                st.write("")  
                
                st.markdown("<div style='text-align:center; color: #9aa4b2; font-size:12px;'>Role icon</div>", unsafe_allow_html=True)

            
            e1, e2 = st.columns(2)
            with e1:
                edus = ['-- Unknown --'] + sorted(df['Education Level'].dropna().unique())
                education = st.selectbox("Education Level", edus, index=0 if st.session_state.form_defaults['Education Level'] == '' else 1)
            with e2:
                genders = ['-- Unknown --'] + mappings['genders']
                gender = st.selectbox("Gender", genders, index=0 if st.session_state.form_defaults['Gender'] == '' else 1)

            
            st.write("")
            submit_btn = st.form_submit_button("Predict Salary")

        st.markdown("</div>", unsafe_allow_html=True)  

    
    with right_col:
        st.markdown('<div class="glass result-card">', unsafe_allow_html=True)
        
        st.markdown('<div class="blob blob-1"></div><div class="blob blob-2"></div>', unsafe_allow_html=True)

        
        if ('submit_btn' not in locals() or not submit_btn) and 'last_prediction' not in st.session_state:
            
            st.markdown("<div style='text-align:center; padding-top:20px;'>", unsafe_allow_html=True)
            st.markdown(f"<div style='width:72px; height:72px; border-radius:20px; display:inline-flex; align-items:center; justify-content:center; background:{'linear-gradient(90deg, #8b5cf6, #5b21b6)' if not dark_mode else 'linear-gradient(90deg, #5b21b6, #8b5cf6)'}; box-shadow: 0 8px 20px rgba(91,33,182,0.15);'>{svg_icon('dollar', size=28)}</div>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:inherit; margin-top:18px;'>Ready to Predict</h3>", unsafe_allow_html=True)
            st.markdown("<div class='result-sub'>Fill in the form on the left and click Predict to view the estimated salary.</div>", unsafe_allow_html=True)
        else:
            
            job_val = custom_job_title if (job_title == '-- Other --' and custom_job_title.strip() != "") else job_title
            input_vals = {
                'Age': age,
                'Years of Experience': years_exp,
                'Job Title': job_val,
                'Education Level': "" if education == '-- Unknown --' else education,
                'Gender': "" if gender == '-- Unknown --' else gender
            }

            if submit_btn:
                
                st.session_state.form_defaults['Age'] = age
                st.session_state.form_defaults['Years of Experience'] = years_exp
                st.session_state.form_defaults['Job Title'] = job_title
                st.session_state.form_defaults['Education Level'] = education if education != '-- Unknown --' else ''
                st.session_state.form_defaults['Gender'] = gender if gender != '-- Unknown --' else ''
                st.session_state.form_defaults['Custom Job Title'] = custom_job_title

                
                if not model_choice:
                    st.error("No model selected or uploaded! Please choose a model in the left panel.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                
                try:
                    model = load_model(model_choice)
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                
                X = compute_features(input_vals, mappings)
                try:
                    pred = model.predict(X)[0]
                except Exception:
                    pred = model.predict(X.select_dtypes(include=[np.number]))[0]

                
                st.session_state.last_prediction = {
                    'salary': float(pred),
                    'currency': 'INR',
                    'inputs': input_vals
                }

            
            pred = st.session_state.last_prediction
            salary_str = f"{pred['currency']} {int(pred['salary']):,}"

            
            st.markdown("<div style='text-align:center; z-index:2; position:relative;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:12px; color:#9aa4b2; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;'>Estimated Annual Salary</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-amount' style='color:inherit; margin-bottom:6px;'>{salary_str}</div>", unsafe_allow_html=True)
            st.markdown("<div class='result-sub' style='margin-bottom:10px;'>Model-based estimate • For demonstration only</div>", unsafe_allow_html=True)

           
            conf_tag = "<div style='display:inline-flex; align-items:center; gap:8px; background: rgba(34,197,94,0.08); padding:6px 10px; border-radius:999px; color:#22c55e; border: 1px solid rgba(34,197,94,0.12); font-weight:600;'>● High confidence</div>"
            st.markdown(conf_tag, unsafe_allow_html=True)

            
            st.markdown("<div style='margin-top:18px; border-top:1px solid rgba(255,255,255,0.04); padding-top:12px; display:grid; grid-template-columns:1fr 1fr; gap:12px;'>", unsafe_allow_html=True)
            
            inp = pred['inputs']
            job_display = inp['Job Title'] if inp['Job Title'] else "—"
            edu_display = inp['Education Level'] if inp['Education Level'] else "—"
            gen_display = inp['Gender'] if inp['Gender'] else "—"
            st.markdown(f"<div><div style='font-size:11px; color:{muted}; text-transform:uppercase;'>Job Title</div><div style='font-weight:700; margin-top:4px;'>{job_display}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div><div style='font-size:11px; color:{muted}; text-transform:uppercase;'>Experience</div><div style='font-weight:700; margin-top:4px;'>{inp['Years of Experience']} years</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div><div style='font-size:11px; color:{muted}; text-transform:uppercase;'>Education</div><div style='font-weight:700; margin-top:4px;'>{edu_display}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div><div style='font-size:11px; color:{muted}; text-transform:uppercase;'>Age</div><div style='font-weight:700; margin-top:4px;'>{int(inp['Age'])}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  

   


if __name__ == "__main__":
    main()
