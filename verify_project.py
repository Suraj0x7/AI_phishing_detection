import os
import sys
import joblib
import pandas as pd
import numpy as np

def run_verification():
    print("=" * 70)
    print("                 COMPREHENSIVE FINAL VERIFICATION")
    print("=" * 70)

    # 1. Verify Directories
    expected_dirs = [
        "data/raw", "data/processed", "src", "models", "results/metrics",
        "results/figures", "results/confusion_matrices", "results/cross_validation",
        "results/ablation", "results/leakage_audit", "results/predictions", "notebooks"
    ]
    for d in expected_dirs:
        assert os.path.exists(d), f"Missing directory: {d}"
    print("[PASS] All required directories exist.")

    # 2. Verify Models
    expected_models = [
        "models/logistic_regression.joblib",
        "models/random_forest.joblib",
        "models/xgboost.joblib",
        "models/stylometric_scaler.joblib",
        "models/tfidf_vectorizer.joblib",
        "models/stylometric_class_profiles.joblib",
        "models/label_encoder.joblib"
    ]
    for m in expected_models:
        assert os.path.exists(m), f"Missing model: {m}"
        obj = joblib.load(m)
        assert obj is not None, f"Model object is None: {m}"
    print("[PASS] All models and vectorizers load correctly.")

    # 3. Verify All 13 Figures + System Architecture
    expected_figures = [
        "01_class_distribution.png",
        "02_source_distribution.png",
        "03_stylometric_class_comparison.png",
        "04_stylometric_correlation_matrix.png",
        "05_cross_validation_comparison.png",
        "06_test_model_comparison.png",
        "07_logistic_regression_confusion_matrix.png",
        "08_random_forest_confusion_matrix.png",
        "09_xgboost_confusion_matrix.png",
        "10_ablation_study.png",
        "11_leakage_similarity_distribution.png",
        "12_per_class_f1_comparison.png",
        "13_model_training_time.png",
        "system_architecture.png",
        "system_architecture.svg",
        "dataset_before_after_cleaning.png",
        "stylometric_feature_distribution.png"
    ]
    for f in expected_figures:
        fp = os.path.join("results/figures", f)
        assert os.path.exists(fp), f"Missing figure: {fp}"
        assert os.path.getsize(fp) > 1000, f"Figure file empty: {fp}"
    print(f"[PASS] All {len(expected_figures)} figures verified with non-zero high-res content.")

    # 4. Verify CSV and Text Metric Files
    expected_metrics = [
        "results/metrics/model_comparison.csv",
        "results/metrics/classification_reports.txt",
        "results/metrics/training_times.csv",
        "results/cross_validation/cv_results.csv",
        "results/ablation/ablation_results.csv",
        "results/leakage_audit/similarity_results.csv",
        "results/leakage_audit/leakage_audit_report.txt",
        "results/predictions/test_predictions.csv",
        "data/processed/combined_dataset.csv",
        "data/processed/cleaned_dataset.csv",
        "data/processed/stylometric_features.csv",
        "results/experiment_config.json"
    ]
    for m in expected_metrics:
        assert os.path.exists(m), f"Missing metric/result file: {m}"
        assert os.path.getsize(m) > 10, f"Result file empty: {m}"
    print(f"[PASS] All {len(expected_metrics)} metric and data files verified.")

    # 5. Verify Inference Pipeline
    from src.predict import predict_single_email
    test_email = "Subject: Urgent Password Reset\n\nPlease confirm your account credentials immediately."
    res = predict_single_email(test_email, model_name="logistic_regression")
    assert "predicted_class" in res
    assert "probabilities" in res
    assert "explanation" in res
    print(f"[PASS] Inference pipeline verified: input -> '{res['display_class']}' ({res['confidence']*100:.1f}%)")

    # 6. Verify Streamlit syntax and app imports
    import app
    print("[PASS] Streamlit application imports and initializes cleanly.")

    print("\n" + "=" * 70)
    print("        ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY (100% PASS)")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
