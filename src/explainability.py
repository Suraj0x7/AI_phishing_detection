import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.stylometric_features import FEATURE_COLUMNS, extract_stylometric_features

DEFAULT_PROFILES_PATH = BASE_DIR / "models" / "stylometric_class_profiles.joblib"

def explain_email_stylometry(email_text: str, predicted_class: str, profiles_path=None):
    """
    Computes interpretable stylometric explanations by calculating feature deviations (z-scores)
    relative to training-set class profiles for the predicted class.
    Uses portable pathlib paths for seamless local and cloud deployment.
    """
    target_path = Path(profiles_path) if profiles_path else DEFAULT_PROFILES_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Class profiles not found at: {target_path}. Run feature_fusion.py first.")

    class_profiles = joblib.load(target_path)
    
    if predicted_class not in class_profiles:
        predicted_class = list(class_profiles.keys())[0]

    profile = class_profiles[predicted_class]
    means = np.array(profile["mean"])
    stds = np.array(profile["std"])

    # Extract email features
    features_dict = extract_stylometric_features(email_text)
    feature_names = FEATURE_COLUMNS
    raw_values = np.array([features_dict[col] for col in feature_names])

    # Avoid zero division
    stds_safe = np.where(stds == 0, 1e-6, stds)
    z_scores = (raw_values - means) / stds_safe

    explanations = []
    for idx, col in enumerate(feature_names):
        val = raw_values[idx]
        z = z_scores[idx]
        mean_val = means[idx]

        if z >= 2.0:
            level = "VERY HIGH"
        elif z >= 1.0:
            level = "HIGH"
        elif z <= -2.0:
            level = "VERY LOW"
        elif z <= -1.0:
            level = "LOW"
        else:
            level = "NORMAL"

        human_title = col.replace('_', ' ').title()
        explanations.append({
            "feature": col,
            "feature_name": human_title,
            "value": round(float(val), 2),
            "class_mean": round(float(mean_val), 2),
            "z_score": round(float(z), 2),
            "abs_z": round(float(abs(z)), 2),
            "level": level
        })

    # Sort by largest deviation
    explanations.sort(key=lambda x: x["abs_z"], reverse=True)

    # Top analyst highlights
    top_signals = []
    for item in explanations[:4]:
        top_signals.append(f"{item['feature_name']}: {item['level']} (Value: {item['value']}, Norm: {item['class_mean']})")

    return {
        "predicted_class": predicted_class,
        "features": features_dict,
        "ranked_explanations": explanations,
        "top_signals": top_signals
    }

if __name__ == "__main__":
    sample = "URGENT ACTION REQUIRED! Verify your corporate credentials immediately or access will be suspended."
    res = explain_email_stylometry(sample, "phishing")
    print(f"Sample Explanation for: {res['predicted_class']}")
    for s in res['top_signals']:
        print("  -", s)
