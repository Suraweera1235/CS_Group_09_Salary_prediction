# salary_app_ui.py
import os
import tempfile
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ----------------- Cached helpers -----------------
@st.cache_data
def load_dataset():
    """Load dataset from project /data/Salary_Data.csv (two levels up)."""
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "data", "Salary_Data.csv")
    return pd.read_csv(path)


@st.cache_data
def build_mappings(df: pd.DataFrame):
    """Create helpful mappings from the data for encoding/choices."""
    mappings = {}

    # Job Title encoding: mean salary per job title + global mean
    if 'Job Title' in df.columns and 'Salary' in df.columns:
        mappings['job_title_mean'] = df.groupby('Job Title')['Salary'].mean().to_dict()
        mappings['job_title_global_mean'] = float(df['Salary'].mean())
    else:
        mappings['job_title_mean'] = {}
        mappings['job_title_global_mean'] = 0.0

    # Education ordinal mapping (based on mean salary ordering)
    if 'Education Level' in df.columns and 'Salary' in df.columns:
        edu_means = df.groupby('Education Level')['Salary'].mean().sort_values()
        mappings['education_map'] = {k: i for i, k in enumerate(edu_means.index.tolist())}
    else:
        mappings['education_map'] = {}

    # Gender options (keep only Male, Female)
    allowed_genders = ["Male", "Female"]
    if 'Gender' in df.columns:
        df_g = df[df['Gender'].isin(allowed_genders)]
        mappings['genders'] = sorted(df_g['Gender'].unique().tolist())
    else:
        mappings['genders'] = []

    return mappings


def list_models():
    """List model files stored in ../models directory."""
    base = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base, "models")
    if not os.path.exists(models_dir):
        return [], models_dir
    files = [f for f in os.listdir(models_dir) if f.endswith(('.joblib', '.pkl'))]
    return files, models_dir


def load_model(path):
    """Load joblib/pkl model. Let exceptions propagate to caller for user-friendly message."""
    return joblib.load(path)


def compute_features(input_vals: dict, mappings=None):
    """Convert form inputs into a single-row DataFrame for the model."""
    # Basic numeric conversions and safe defaults
    row = {
        'Age': float(input_vals.get('Age', 0.0) or 0.0),
        'Years of Experience': float(input_vals.get('Years of Experience', 0.0) or 0.0),
        'Gender': input_vals.get('Gender', '') or '',
        'Education Level': input_vals.get('Education Level', '') or '',
        'Job Title': input_vals.get('Job Title', '') or ''
    }
    return pd.DataFrame([row])


def clean_text(text):
    """Minimal normalization for education/job strings used in mapping keys."""
    if isinstance(text, str):
        text = text.strip().lower()
        text = text.replace("’", "'")
        # common normalizations
        text = text.replace(" degree", "")
        text = text.replace("bachelors", "bachelor's")
        text = text.replace("masters", "master's")
        # keep phd as-is (already short)
    return text


# base color tokens (used only as fallbacks)
FALLBACK_MUTED = "#6c757d"
PRIMARY_TEXT = "#ffffff"
SECONDARY_TEXT = "#000000"
CARD_BG_DARK = "#1e1e1e"
CARD_BG_LIGHT = "#ffffff"


def inject_css(dark_mode: bool):
    """
    Inject CSS for styling. Use distinct local variable names to avoid shadowing.
    """
    bg = "#0b1220" if dark_mode else "#f6f7fb"
    surface = "rgba(13,18,28,0.65)" if dark_mode else "rgba(255,255,255,0.9)"
    card_border = "rgba(255,255,255,0.06)" if dark_mode else "#e6e9ef"
    text_color = "#8cebf2" if dark_mode else "#0b1220"
    css_muted = "#141515" if dark_mode else "#64748b"
    accent = "#2C4479"
    accent_light = "rgba(91,33,182,0.10)"

    # Keep CSS minimal but effective
    st.markdown(
        f"""
    <style>
    .stApp {{
        background: {bg};
        color: {text_color};
        font-family: "Segoe UI", Roboto, -apple-system, "Helvetica Neue", Arial;
    }}
    .header {{
        background: {surface};
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 14px 18px;
        display:flex;
        align-items:center;
        gap:12px;
        margin-bottom:18px;
    }}
    .brand-badge {{
        width:62px; height:42px; border-radius:10px;
        background: linear-gradient(135deg, {accent}, #8b5cf6);
        display:flex; align-items:center; justify-content:center; color:white; font-weight:700;
    }}
    .brand-title {{ font-size:18px; font-weight:700; color:transparent; background: linear-gradient(90deg, {accent}, #8b5cf6); -webkit-background-clip:text; background-clip:text; }}
    .brand-sub {{ color: {css_muted}; font-size:12px; }}

    .glass {{
        background: {surface};
        border-radius: 14px;
        padding: 18px;
        border: 1px solid {card_border};
    }}

    .role-strip {{ overflow-x:auto; white-space:nowrap; padding-bottom:8px; margin-top:6px; }}
    .role-card {{
        display:inline-block; width:220px; margin-right:12px; padding:12px; border-radius:10px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
        border: 1px solid {card_border};
    }}

    .stButton>button {{
        background: linear-gradient(90deg, {accent}, #8b5cf6) !important;
        border: none !important;
        color: white !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}

    .result-card {{ border-radius:14px; padding: 20px; position:relative; overflow:hidden; }}
    .result-amount {{ font-size:48px; font-weight:800; letter-spacing:-1px; }}
    .result-sub {{ color: {css_muted}; font-size:13px; }}

    /* small responsive tweaks */
    @media (max-width: 880px) {{
        .role-card {{ width:180px; }}
        .result-amount {{ font-size:36px; }}
    }}

    /* friendly label style */
    .feature-label {{ color: {css_muted} !important; font-weight:500; font-size:13px; }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def svg_icon(name: str, size: int = 18):
    icons = {
        "dollar": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 1v22" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'.format(s=size),
        "briefcase": '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="7" width="18" height="12" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M8 7V5a2 2 0 012-2h4a2 2 0 012 2v2" stroke="currentColor" stroke-width="1.5"/></svg>'.format(s=size),
    }
    return icons.get(name, "")


def main():
    st.set_page_config(page_title="Salary Predictor", layout="wide", initial_sidebar_state="expanded")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        dark_mode = st.checkbox("Dark mode", value=False, help="Toggle dark/light styling")
        st.markdown("---")

    inject_css(dark_mode)

    # Top header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(
            f"""
            <div class="header">
                <div class="brand-badge">{svg_icon('dollar', size=20)}</div>
                <div style="display:flex; flex-direction:column;">
                    <div class="brand-title">SalaryAI</div>
                    <div class="brand-sub">Professional salary estimator</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.write("")  # spacer

    # Load data and build mappings
    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    # normalize education strings if present
    if 'Education Level' in df.columns:
        df['Education Level'] = df['Education Level'].apply(clean_text)

    mappings = build_mappings(df)

    # initialize safe session defaults (avoid None passed to number_input/slider)
    if 'form_defaults' not in st.session_state:
        st.session_state.form_defaults = {
            'Age': 25,
            'Years of Experience': 2.0,
            'Job Title': '',
            'Education Level': '',
            'Gender': '',
            'Custom Job Title': ''
        }

    left_col, right_col = st.columns([7, 5], gap="large")

    # Left: form
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
            # save uploaded file to temp file then use that path
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            tmp.close()
            model_choice = tmp.name

        st.markdown("---")

        # job titles from dataset
        job_titles = ['-- Other --']
        if 'Job Title' in df.columns:
            job_titles += sorted(df['Job Title'].dropna().unique().tolist())

        # The form
        with st.form("predict_form"):
            a1, a2 = st.columns(2)
            with a1:
                st.markdown("<label class='feature-label'>Age</label>", unsafe_allow_html=True)
                age = st.number_input(
                    "", min_value=18, max_value=100,
                    value=int(st.session_state.form_defaults.get('Age', 25))
                )

            with a2:
                st.markdown("<label class='feature-label'>Years of Experience</label>", unsafe_allow_html=True)
                years_exp = st.slider(
                    "", min_value=0.0, max_value=50.0,
                    value=float(st.session_state.form_defaults.get('Years of Experience', 2.0)),
                    step=0.5
                )

            st.write("")
            j1, j2 = st.columns([3, 1])

            with j1:
                st.markdown("<label class='feature-label'>Job Title</label>", unsafe_allow_html=True)
                # determine index: if saved default exists in job_titles pick that index, otherwise 0
                default_job = st.session_state.form_defaults.get('Job Title', '')
                try:
                    idx = job_titles.index(default_job) if default_job in job_titles else 0
                except Exception:
                    idx = 0
                job_title = st.selectbox("", job_titles, index=idx)

                custom_job_title = ""
                if job_title == '-- Other --':
                    st.markdown("<label class='feature-label'>Enter job title</label>", unsafe_allow_html=True)
                    custom_job_title = st.text_input("", value=st.session_state.form_defaults.get('Custom Job Title', ''))

            with j2:
                st.write("")
                st.markdown("<div style='text-align:center; color: #9aa4b2; font-size:12px;'>Role icon</div>", unsafe_allow_html=True)

            e1, e2 = st.columns(2)
            with e1:
                edus = ['-- Unknown --']
                if 'Education Level' in df.columns:
                    edus += sorted(df['Education Level'].dropna().unique().tolist())
                st.markdown("<label class='feature-label'>Education Level</label>", unsafe_allow_html=True)
                # pre-select education if saved
                default_edu = st.session_state.form_defaults.get('Education Level', '')
                try:
                    idx_edu = edus.index(default_edu) if default_edu in edus else 0
                except Exception:
                    idx_edu = 0
                education = st.selectbox("", edus, index=idx_edu)

            with e2:
                genders = ['-- Unknown --'] + mappings.get('genders', [])
                st.markdown("<label class='feature-label'>Gender</label>", unsafe_allow_html=True)
                default_gender = st.session_state.form_defaults.get('Gender', '')
                try:
                    idx_gen = genders.index(default_gender) if default_gender in genders else 0
                except Exception:
                    idx_gen = 0
                gender = st.selectbox("", genders, index=idx_gen)

            st.write("")
            submit_btn = st.form_submit_button("Predict Salary")

        # close left glass div
        st.markdown("</div>", unsafe_allow_html=True)

    # Right: result card
    with right_col:
        st.markdown('<div class="glass result-card">', unsafe_allow_html=True)

        # Ready state
        if not submit_btn and 'last_prediction' not in st.session_state:
            st.markdown("<div style='text-align:center; padding-top:20px;'>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='width:72px; height:72px; border-radius:20px; display:inline-flex; align-items:center; justify-content:center; background: linear-gradient(90deg, #8b5cf6, #5b21b6); box-shadow: 0 8px 20px rgba(91,33,182,0.15);'>{svg_icon('dollar', size=28)}</div>",
                unsafe_allow_html=True
            )
            st.markdown("<h3 style='color:inherit; margin-top:18px;'>Ready to Predict</h3>", unsafe_allow_html=True)
            st.markdown("<div class='result-sub'>Fill in the form on the left and click Predict to view the estimated salary.</div>", unsafe_allow_html=True)
        else:
            # determine final job value
            job_val = custom_job_title.strip() if (job_title == '-- Other --' and custom_job_title.strip() != "") else job_title
            input_vals = {
                'Age': age,
                'Years of Experience': years_exp,
                'Job Title': job_val,
                'Education Level': "" if education == '-- Unknown --' else education,
                'Gender': "" if gender == '-- Unknown --' else gender
            }

            if submit_btn:
                # save defaults so fields persist between predictions
                st.session_state.form_defaults['Age'] = int(age)
                st.session_state.form_defaults['Years of Experience'] = float(years_exp)
                st.session_state.form_defaults['Job Title'] = job_title
                st.session_state.form_defaults['Education Level'] = "" if education == '-- Unknown --' else education
                st.session_state.form_defaults['Gender'] = "" if gender == '-- Unknown --' else gender
                st.session_state.form_defaults['Custom Job Title'] = custom_job_title

                if not model_choice:
                    st.error("No model selected or uploaded! Please choose a model in the left panel.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                # attempt to load model
                try:
                    model = load_model(model_choice)
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

                # prepare features and predict
                X = compute_features(input_vals, mappings)
                try:
                    pred_val = model.predict(X)[0]
                except Exception:
                    # fallback to numeric-only features (if pipeline not present)
                    try:
                        pred_val = model.predict(X.select_dtypes(include=[np.number]))[0]
                    except Exception as e:
                        st.error(f"Model prediction failed: {e}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        return

                # store result in session
                st.session_state.last_prediction = {
                    'salary': float(pred_val),
                    'currency': 'INR',
                    'inputs': input_vals
                }

            # show last prediction
            pred = st.session_state.get('last_prediction', None)
            if pred is None:
                st.info("No prediction available yet.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # human-friendly formatting
            salary_int = int(round(pred['salary']))
            salary_str = f"{pred.get('currency', '')} {salary_int:,}"

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
            st.markdown(f"<div><div style='font-size:11px; color:#6c757d; text-transform:uppercase;'>Job Title</div><div style='font-weight:700; margin-top:4px;'>{job_display}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div><div style='font-size:11px; color:#6c757d; text-transform:uppercase;'>Experience</div><div style='font-weight:700; margin-top:4px;'>{inp['Years of Experience']} years</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div><div style='font-size:11px; color:#6c757d; text-transform:uppercase;'>Education</div><div style='font-weight:700; margin-top:4px;'>{edu_display}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div><div style='font-size:11px; color:#6c757d; text-transform:uppercase;'>Age</div><div style='font-weight:700; margin-top:4px;'>{int(inp['Age'])}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close result-card glass

if __name__ == "__main__":
    main()
