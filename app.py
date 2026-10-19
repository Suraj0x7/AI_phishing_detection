import os
import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Portable base directories via pathlib
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
CROSS_VAL_DIR = RESULTS_DIR / "cross_validation"
ABLATION_DIR = RESULTS_DIR / "ablation"
LEAKAGE_DIR = RESULTS_DIR / "leakage_audit"

from src.predict import predict_single_email, DISPLAY_NAMES, get_artifacts
from src.stylometric_features import FEATURE_COLUMNS

st.set_page_config(
    page_title="AI-Generated Phishing Email Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished academic appearance
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-legit {
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 1.2rem;
        border: 1px solid #86efac;
    }
    .badge-trad {
        background-color: #fee2e2;
        color: #b91c1c;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 1.2rem;
        border: 1px solid #fca5a5;
    }
    .badge-ai {
        background-color: #dbeafe;
        color: #1d4ed8;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 1.2rem;
        border: 1px solid #93c5fd;
    }
</style>
""", unsafe_allow_html=True)

# Pre-load artifacts at startup
try:
    get_artifacts(MODEL_DIR)
    artifacts_ready = True
except Exception as e:
    artifacts_ready = False
    load_err = str(e)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("Configuration")

model_choice = st.sidebar.selectbox(
    "Select Detection Model",
    ["Logistic Regression", "Random Forest", "XGBoost"],
    index=0,
    help="Models trained on 15,010 fused stylometric and semantic features."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Test Presets")
preset_choice = st.sidebar.radio(
    "Choose a sample email:",
    [
        "None (Custom Input)",
        "Sample 1: AI-Generated Phishing (M365)",
        "Sample 2: Traditional Phishing (Bank Alert)",
        "Sample 3: Legitimate Workplace Email"
    ]
)

PRESETS = {
    "Sample 1: AI-Generated Phishing (M365)": (
        "Subject: Microsoft 365: Urgent Action Required\n\n"
        "Dear Team Member,\n\n"
        "A mandatory security update is pending for your organizational Microsoft 365 account. "
        "Due to recent policy revisions, please verify your corporate credentials within 24 hours "
        "to prevent temporary suspension of your email and cloud workspace access.\n\n"
        "Please visit your secure verification portal to confirm your profile.\n\n"
        "Sincerely,\nIT Security & Identity Governance"
    ),
    "Sample 2: Traditional Phishing (Bank Alert)": (
        "Subject: DON'T DELETE THIS MESSAGE - IMMEDIATE CITIBANK ALERT\n\n"
        "Dear valued client!! Your debit account has been flagged for unauthorized access. "
        "CLICK HERE NOW to re-activate your account: http://citibank-secure-update.com/login.php. "
        "Failure to verify immediately will result in permanent account termination and penalties!!"
    ),
    "Sample 3: Legitimate Workplace Email": (
        "Subject: Q3 Sprint Review and Architecture Agenda\n\n"
        "Hi team,\n\n"
        "Please review the attached slide deck ahead of our sprint retro tomorrow at 10:00 AM. "
        "We will discuss database migration benchmarks and the roadmap for the upcoming release. "
        "Let me know if you would like to add any items to the agenda.\n\n"
        "Best regards,\nDavid"
    )
}

# Main Application Header
st.markdown('<div class="main-title">🛡️ AI-Generated Phishing Email Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Hybrid Stylometric and Semantic Learning Framework for Detecting AI-Generated & Traditional Phishing Emails</div>', unsafe_allow_html=True)

if not artifacts_ready:
    st.error(f"Models not loaded: {load_err}. Run 'python run_pipeline.py' first.")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Email Detection",
    "📊 Stylometric Analysis",
    "📈 Model Performance",
    "🗃️ Dataset Statistics",
    "📖 About the Research"
])

# TAB 1: EMAIL DETECTION
with tab1:
    st.subheader("Interactive Email Classification")

    initial_text = ""
    if preset_choice in PRESETS:
        initial_text = PRESETS[preset_choice]

    email_input = st.text_area(
        "Paste Email Subject & Body below for real-time analysis:",
        value=initial_text,
        height=220,
        placeholder="Paste complete email text including Subject line here..."
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Email", type="primary", use_container_width=True)

    if analyze_clicked or (preset_choice != "None (Custom Input)" and email_input):
        if not email_input.strip():
            st.warning("Please enter or paste an email before analyzing.")
        else:
            with st.spinner("Analyzing stylometric syntax & semantic embeddings..."):
                res = predict_single_email(email_input, model_name=model_choice, models_dir=MODEL_DIR)

            st.markdown("---")
            col_res1, col_res2 = st.columns([1.2, 1])

            with col_res1:
                st.markdown("### Prediction Outcome")
                cls = res["predicted_class"]
                if cls == "legitimate":
                    st.markdown(f'<div class="badge-legit">✅ {res["display_class"]}</div>', unsafe_allow_html=True)
                elif cls == "phishing":
                    st.markdown(f'<div class="badge-trad">⚠️ {res["display_class"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-ai">🤖 {res["display_class"]}</div>', unsafe_allow_html=True)

                st.markdown(f"**Selected Model:** `{model_choice}` &nbsp;|&nbsp; **Confidence:** `{res['confidence']*100:.1f}%`")

                st.markdown("#### Class Probability Distribution")
                probs = res.get("probabilities", {})
                for c_key in ["legitimate", "phishing", "ai_phishing"]:
                    p_val = probs.get(c_key, 0.0)
                    st.write(f"**{DISPLAY_NAMES[c_key]}:** `{p_val*100:.2f}%`")
                    st.progress(float(p_val))

            with col_res2:
                st.markdown("### 🔍 Top Stylometric Signals")
                for sig in res["explanation"]["top_signals"]:
                    st.info(f"📌 {sig}")

                st.markdown("### 📋 Primary Feature Snapshot")
                f_dict = res["stylometric_features"]
                c1, c2 = st.columns(2)
                c1.metric("Word Count", f"{int(f_dict.get('word_count', 0))}")
                c1.metric("Urgency Words", f"{int(f_dict.get('urgency_word_count', 0))}")
                c2.metric("Imperative Verbs", f"{int(f_dict.get('imperative_verb_count', 0))}")
                c2.metric("Readability (FRE)", f"{f_dict.get('flesch_reading_ease', 0):.1f}")

# TAB 2: STYLOMETRIC ANALYSIS
with tab2:
    st.subheader("Deep Stylometric Writing-Style Profile")
    st.markdown("The framework inspects 10 writing-style indicators to differentiate synthetic AI writing from human phishing and legitimate communication.")

    if 'res' in locals():
        f_df = pd.DataFrame(res["explanation"]["ranked_explanations"])
        st.dataframe(
            f_df[["feature_name", "value", "class_mean", "z_score", "level"]].rename(columns={
                "feature_name": "Stylometric Feature",
                "value": "Email Value",
                "class_mean": "Class Norm Mean",
                "z_score": "Z-Score Deviation",
                "level": "Analyst Indicator"
            }),
            use_container_width=True
        )
    else:
        st.info("Analyze an email in Tab 1 to see its interactive stylometric deviation breakdown.")

    st.markdown("#### Global Stylometric Feature Correlation & Class Profiling")
    col_img1, col_img2 = st.columns(2)
    img_comp = FIGURES_DIR / "03_stylometric_class_comparison.png"
    img_corr = FIGURES_DIR / "04_stylometric_correlation_matrix.png"
    if img_comp.exists():
        col_img1.image(str(img_comp), caption="Stylometric Profiles Across Target Classes", use_container_width=True)
    if img_corr.exists():
        col_img2.image(str(img_corr), caption="Correlation Matrix of Stylometric Features", use_container_width=True)

# TAB 3: MODEL PERFORMANCE
with tab3:
    st.subheader("Model Evaluation & Empirical Comparison")
    
    file_comp = METRICS_DIR / "model_comparison.csv"
    if file_comp.exists():
        df_comp = pd.read_csv(file_comp)
        st.markdown("#### Held-Out Test Set Performance (Untouched 20% Split)")
        st.dataframe(df_comp, use_container_width=True)

    file_cv = CROSS_VAL_DIR / "cv_results.csv"
    if file_cv.exists():
        df_cv = pd.read_csv(file_cv)
        st.markdown("#### Stratified 5-Fold Cross-Validation Metrics (Mean ± SD)")
        st.dataframe(df_cv, use_container_width=True)

    col_cf1, col_cf2 = st.columns(2)
    img_test_comp = FIGURES_DIR / "06_test_model_comparison.png"
    img_f1_comp = FIGURES_DIR / "12_per_class_f1_comparison.png"
    if img_test_comp.exists():
        col_cf1.image(str(img_test_comp), caption="Overall Test Performance", use_container_width=True)
    if img_f1_comp.exists():
        col_cf2.image(str(img_f1_comp), caption="Per-Class F1 Breakdown", use_container_width=True)

    st.markdown("#### Confusion Matrices (Label Order: Legitimate, Phishing, AI Phishing)")
    cm1, cm2, cm3 = st.columns(3)
    img_cm_lr = FIGURES_DIR / "07_logistic_regression_confusion_matrix.png"
    img_cm_rf = FIGURES_DIR / "08_random_forest_confusion_matrix.png"
    img_cm_xgb = FIGURES_DIR / "09_xgboost_confusion_matrix.png"
    if img_cm_lr.exists():
        cm1.image(str(img_cm_lr), caption="Logistic Regression", use_container_width=True)
    if img_cm_rf.exists():
        cm2.image(str(img_cm_rf), caption="Random Forest", use_container_width=True)
    if img_cm_xgb.exists():
        cm3.image(str(img_cm_xgb), caption="XGBoost", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Ablation Study (Stylometric vs. Semantic vs. Hybrid)")
    file_abl = ABLATION_DIR / "ablation_results.csv"
    img_abl = FIGURES_DIR / "10_ablation_study.png"
    if file_abl.exists():
        df_abl = pd.read_csv(file_abl)
        st.dataframe(df_abl, use_container_width=True)
    if img_abl.exists():
        st.image(str(img_abl), caption="Ablation Experiment Comparison", width=800)

# TAB 4: DATASET STATISTICS
with tab4:
    st.subheader("Curated Research Datasets & Leakage Audit")
    
    col_ds1, col_ds2 = st.columns(2)
    img_cls_dist = FIGURES_DIR / "01_class_distribution.png"
    img_src_dist = FIGURES_DIR / "02_source_distribution.png"
    if img_cls_dist.exists():
        col_ds1.image(str(img_cls_dist), caption="Class Sample Distribution", use_container_width=True)
    if img_src_dist.exists():
        col_ds2.image(str(img_src_dist), caption="Distribution by Verified Data Source", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Near-Duplicate & Data Leakage Audit Findings")
    st.markdown("""
    Academic integrity rule: In synthetic phishing research, unusually high accuracy (>99%) frequently arises from 
    batch LLM template similarity. To verify this, we computed maximum train-test TF-IDF cosine similarity for every test sample.
    """)
    img_leakage = FIGURES_DIR / "11_leakage_similarity_distribution.png"
    if img_leakage.exists():
        st.image(str(img_leakage), caption="Train-Test Maximum Cosine Similarity Distribution", width=800)

    file_leak_rep = LEAKAGE_DIR / "leakage_audit_report.txt"
    if file_leak_rep.exists():
        with open(file_leak_rep, "r") as f:
            st.text(f.read())

# TAB 5: ABOUT THE RESEARCH
with tab5:
    st.subheader("About the Research Project")
    st.markdown("""
    ### Project Title
    **“A Hybrid Stylometric and Semantic Learning Framework for Detecting AI-Generated Phishing Emails”**

    #### Academic Rationale & Problem Statement
    Traditional email security filters rely heavily on keyword blacklists, domain reputation, and structural anomalies. 
    However, Large Language Models (LLMs) like GPT-4, Claude, and Gemini can generate grammatically flawless, highly persuasive, 
    and context-tailored phishing emails that completely lack typical red flags (spelling errors, broken English, crude formatting).

    #### Hybrid Architecture Rationale
    1. **Semantic Branch (TF-IDF, 15,000 features):** Captures high-dimensional lexical contexts, service provider terminologies, 
       and domain-specific semantic expressions.
    2. **Stylometric Branch (10 features):** Measures intrinsic writing-style fingerprints: syntactic sentence complexity, 
       syllabic readability (Flesch score), urgency densities, pronoun distributions, and directive imperative action verbs.
    3. **Standardization & Fusion:** Stylometric features are normalized via a training-fitted `StandardScaler` and concatenated 
       with the sparse semantic matrix, yielding a robust 15,010-dimensional hybrid representation.
    """)

    img_arch = FIGURES_DIR / "system_architecture.png"
    if img_arch.exists():
        st.image(str(img_arch), caption="End-to-End System Architecture", use_container_width=True)
