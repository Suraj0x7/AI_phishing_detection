import os
import sys
from pathlib import Path
import joblib
import numpy as np
import scipy.sparse as sp

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.preprocessing import clean_email_text
from src.stylometric_features import FEATURE_COLUMNS, extract_stylometric_features
from src.explainability import explain_email_stylometry

LABEL_MAPPING = {"legitimate": 0, "phishing": 1, "ai_phishing": 2}
INT_TO_LABEL = {0: "legitimate", 1: "phishing", 2: "ai_phishing"}
DISPLAY_NAMES = {
    "legitimate": "Legitimate Email",
    "phishing": "Traditional Phishing Email",
    "ai_phishing": "AI-Generated Phishing Email"
}

DEFAULT_MODEL_DIR = BASE_DIR / "models"

def load_inference_artifacts(models_dir=None):
    """
    Loads pre-trained models, scalers, and vectorizers once to ensure fast inference.
    Uses portable pathlib paths for seamless deployment.
    """
    m_dir = Path(models_dir) if models_dir else DEFAULT_MODEL_DIR
    scaler = joblib.load(m_dir / "stylometric_scaler.joblib")
    vectorizer = joblib.load(m_dir / "tfidf_vectorizer.joblib")
    
    models = {
        "logistic_regression": joblib.load(m_dir / "logistic_regression.joblib")
    }
    
    # Load optional comparison models if present
    rf_path = m_dir / "random_forest.joblib"
    if rf_path.exists():
        models["random_forest"] = joblib.load(rf_path)
        
    xgb_path = m_dir / "xgboost.joblib"
    if xgb_path.exists():
        models["xgboost"] = joblib.load(xgb_path)

    return scaler, vectorizer, models

# Global cache for inference speed
_SCALER = None
_VECTORIZER = None
_MODELS = None

def get_artifacts(models_dir=None):
    global _SCALER, _VECTORIZER, _MODELS
    if _SCALER is None or _VECTORIZER is None or _MODELS is None:
        _SCALER, _VECTORIZER, _MODELS = load_inference_artifacts(models_dir)
    return _SCALER, _VECTORIZER, _MODELS

def predict_single_email(raw_email_text: str, model_name: str = "logistic_regression", models_dir=None) -> dict:
    """
    Production inference pipeline:
    raw email -> clean_email_text -> 10 stylometric -> StandardScaler.transform()
    -> TFIDF.transform() -> sparse fusion -> classifier.predict() -> explanation
    """
    scaler, vectorizer, models = get_artifacts(models_dir)

    norm_model_name = model_name.lower().replace(" ", "_")
    if norm_model_name not in models:
        norm_model_name = "logistic_regression"
    model = models[norm_model_name]

    # 1. Preprocess
    cleaned_text = clean_email_text(raw_email_text)
    if not cleaned_text or len(cleaned_text) < 5:
        return {
            "predicted_class": "legitimate",
            "display_class": "Legitimate Email (Empty / Malformed)",
            "confidence": 0.0,
            "probabilities": {"legitimate": 0.33, "phishing": 0.33, "ai_phishing": 0.34},
            "model_used": norm_model_name,
            "stylometric_features": {col: 0.0 for col in FEATURE_COLUMNS},
            "explanation": {"top_signals": ["Text length insufficient for robust analysis"], "ranked_explanations": []}
        }

    # 2. Extract 10 Stylometric Features
    sty_dict = extract_stylometric_features(cleaned_text)
    sty_vals = np.array([[sty_dict[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
    
    # 3. Standardize Stylometrics with Training Scaler
    sty_scaled = scaler.transform(sty_vals).astype(np.float32)
    sty_sparse = sp.csr_matrix(sty_scaled, dtype=np.float32)

    # 4. Transform with Training TF-IDF Vectorizer
    tfidf_sparse = vectorizer.transform([cleaned_text]).astype(np.float32)

    # 5. Hybrid Feature Fusion
    hybrid_vec = sp.hstack([sty_sparse, tfidf_sparse], format='csr', dtype=np.float32)
    hybrid_vec.sort_indices()

    # 6. Predict
    pred_int = int(model.predict(hybrid_vec)[0])
    pred_label = INT_TO_LABEL.get(pred_int, "legitimate")
    display_label = DISPLAY_NAMES.get(pred_label, pred_label)

    # 7. Probability
    probabilities = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(hybrid_vec)[0]
        confidence = float(np.max(probs))
        for idx, p in enumerate(probs):
            probabilities[INT_TO_LABEL[idx]] = round(float(p), 4)

    # 8. Stylometric Explanation
    explanation = explain_email_stylometry(cleaned_text, pred_label)

    return {
        "predicted_class": pred_label,
        "display_class": display_label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "model_used": norm_model_name,
        "cleaned_text": cleaned_text,
        "stylometric_features": sty_dict,
        "explanation": explanation
    }

if __name__ == "__main__":
    test_emails = [
        # AI Phishing sample
        "Subject: Microsoft 365: Urgent Action Required\n\nDear User,\nA security update is pending for your account. Please verify your credentials through our secure portal to prevent suspension of your cloud access.",
        # Traditional Phishing sample
        "Subject: DON'T DELETE - VERIFY CITIBANK ACCOUNT\n\nDear Customer! Your Citibank online card is restricted! Click http://citibank-update.com/login.html now to confirm your PIN and SSN!",
        # Legitimate corporate sample
        "Subject: Q3 Team Planning Meeting Agenda\n\nHi everyone,\nPlease review the attached slide deck ahead of our project status review tomorrow at 10 AM in Conference Room B. Let me know if you have any questions."
    ]

    print("\n--- INFERENCE PIPELINE VERIFICATION ---")
    for idx, em in enumerate(test_emails):
        res = predict_single_email(em, model_name="logistic_regression")
        print(f"\n[Test Email {idx+1}]")
        print(f"  Predicted: {res['display_class']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"  Probabilities: {res['probabilities']}")
        print(f"  Top Signals: {res['explanation']['top_signals'][:2]}")
