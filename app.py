# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import platform

st.set_page_config(
    page_title="Clinical Risk Prediction and Attribution Analysis",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .risk-high { color: #d9534f; font-weight: bold; font-size: 28px; }
    .risk-low { color: #5cb85c; font-weight: bold; font-size: 28px; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
</style>
""", unsafe_allow_html=True)

def set_plot_font():
    system_name = platform.system()
    plt.rcParams['axes.unicode_minus'] = False

    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei']
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial', 'Arial Unicode MS', 'PingFang SC']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei']

@st.cache_resource
def train_model():
    np.random.seed(42)
    n_sim = 500

    data = pd.DataFrame({
        'Age': np.random.normal(65, 12, n_sim),
        'SOFA Score': np.random.randint(1, 16, n_sim),
        'APACHE II': np.random.normal(20, 5, n_sim),
        'Use vasoactive drugs': np.random.binomial(1, 0.4, n_sim),
        'Use CRRT': np.random.binomial(1, 0.2, n_sim),
        'PCT': np.abs(np.random.normal(2, 2, n_sim)),
        'LDH': np.random.normal(250, 50, n_sim),
        'Urea': np.random.normal(10, 3, n_sim),
        'CK': np.random.normal(100, 20, n_sim),
        'PT': np.random.normal(13, 2, n_sim),
        'INR': np.random.uniform(0.9, 3.0, n_sim),
        'TT': np.random.normal(16, 2, n_sim)
    })

    lp = -6 + 0.06 * data['Age'] + 0.3 * data['SOFA Score'] + 1.2 * data['Use vasoactive drugs']
    prob = 1 / (1 + np.exp(-lp))
    y = np.random.binomial(1, prob)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(data, y)

    explainer = shap.TreeExplainer(model)

    return model, explainer

model, explainer = train_model()

st.sidebar.header("📝 Patient Feature Input")

def user_input_features():
    with st.sidebar.expander("Basic Information & Scores", expanded=True):
        age = st.slider("Age", 18, 100, 65)
        sofa = st.slider("SOFA Score", 0, 24, 5)
        apache = st.number_input("APACHE II", 0, 71, 20)

    with st.sidebar.expander("Treatment Measures", expanded=True):
        vaso = st.selectbox("Use vasoactive drugs?", ["No", "Yes"])
        crrt = st.selectbox("Use CRRT?", ["No", "Yes"])

    with st.sidebar.expander("Biochemical Indicators", expanded=False):
        pct = st.number_input("PCT", 0.0, 100.0, 2.0)
        ldh = st.number_input("LDH", 0, 2000, 250)
        urea = st.number_input("Urea", 0.0, 50.0, 10.0)
        ck = st.number_input("CK", 0, 5000, 100)
        pt = st.number_input("PT", 0.0, 100.0, 13.0)
        inr = st.number_input("INR", 0.0, 10.0, 1.1)
        tt = st.number_input("TT", 0.0, 100.0, 16.0)

    row = {
        'Age': age,
        'SOFA Score': sofa,
        'APACHE II': apache,
        'Use vasoactive drugs': 1 if vaso == "Yes" else 0,
        'Use CRRT': 1 if crrt == "Yes" else 0,
        'PCT': pct,
        'LDH': ldh,
        'Urea': urea,
        'CK': ck,
        'PT': pt,
        'INR': inr,
        'TT': tt
    }

    return pd.DataFrame(row, index=[0])

input_df = user_input_features()

st.title("🏥 Clinical Risk Prediction and Attribution Analysis")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📊 Prediction Results")

    if st.button("Start Evaluation"):
        probs = model.predict_proba(input_df)
        risk = probs[0][1]

        st.markdown(f"#### In-Hospital Mortality Risk: {risk * 100:.1f}%")
        st.progress(risk)

        if risk > 0.5:
            st.markdown('<p class="risk-high">⚠️ High-Risk Patient</p>', unsafe_allow_html=True)
            st.error("Enhanced monitoring and intervention are recommended.")
        else:
            st.markdown('<p class="risk-low">✅ Low-Risk Patient</p>', unsafe_allow_html=True)
            st.success("Routine care is recommended.")

        st.markdown("---")
        st.caption("Current Key Input Indicators:")
        st.table(input_df.T.rename(columns={0: 'Value'}).head(5))

with col2:
    st.subheader("🔍 Attribution Analysis (SHAP Waterfall)")
    st.info("Red = Risk-Increasing Factors | Blue = Risk-Reducing Factors")

    set_plot_font()

    shap_values = explainer(input_df)
    explanation = shap_values[0, :, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    st.pyplot(fig)

st.markdown("---")
st.caption("© 2026 RF-SHAP Clinical Prediction System")
