import os
import sys
import time
import json
import joblib
import platform
import pandas as pd
import numpy as np

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.data_loader import inspect_all_datasets, build_combined_dataset
from src.preprocessing import preprocess_dataset
from src.stylometric_features import process_dataset_stylometry
from src.feature_fusion import prepare_splits_and_hybrid_features
from src.train import train_and_tune_models
from src.evaluate import evaluate_models
from src.ablation import run_ablation_study
from src.leakage_audit import audit_data_leakage
from generate_architecture_diagram import generate_system_architecture_diagram

def execute_complete_pipeline():
    total_start = time.time()
    print("=" * 80)
    print(" “A HYBRID STYLOMETRIC AND SEMANTIC LEARNING FRAMEWORK FOR DETECTING”")
    print("                     “AI-GENERATED PHISHING EMAILS”                   ")
    print("                Complete Reproducible Academic ML Pipeline            ")
    print("=" * 80)

    # Step 1: Dataset Inspection & Standardization
    print("\n>>> STEP 1: AUTOMATED DATASET INSPECTION & COMPATIBILITY AUDIT")
    inspect_all_datasets("data")
    df_combined = build_combined_dataset("data", "data/processed")

    # Step 2: Text Preprocessing
    print("\n>>> STEP 2: STYLE-PRESERVING PREPROCESSING & DEDUPLICATION")
    df_clean = preprocess_dataset("data/processed/combined_dataset.csv", "data/processed/cleaned_dataset.csv")

    # Step 3: Stylometric Feature Extraction
    print("\n>>> STEP 3: 10 PRIMARY STYLOMETRIC FEATURES EXTRACTION")
    df_sty = process_dataset_stylometry("data/processed/cleaned_dataset.csv", "data/processed/stylometric_features.csv")

    # Step 4: Leak-Free Train/Test Split & Feature Fusion
    print("\n>>> STEP 4: STRATIFIED TRAIN/TEST SPLIT & HYBRID FEATURE FUSION")
    split_data = prepare_splits_and_hybrid_features(
        "data/processed/cleaned_dataset.csv", 
        "data/processed/stylometric_features.csv"
    )

    # Step 5: Model Training & 5-Fold Cross Validation
    print("\n>>> STEP 5: 5-FOLD CROSS-VALIDATION & HYPERPARAMETER TUNING")
    models, df_cv = train_and_tune_models(split_data)

    # Step 6: Test Set Evaluation
    print("\n>>> STEP 6: TEST SET EVALUATION ACROSS ALL CLASSIFIERS")
    df_test = evaluate_models()

    # Step 7: Ablation Study
    print("\n>>> STEP 7: THREE-TIER ABLATION STUDY EXPERIMENTS")
    df_ablation = run_ablation_study()

    # Step 8: Data Leakage & Near-Duplicate Audit
    print("\n>>> STEP 8: NEAR-DUPLICATE & TRAIN-TEST LEAKAGE AUDIT")
    audit_results = audit_data_leakage()

    # Step 9: Architecture Diagram Generation
    print("\n>>> STEP 9: GENERATING SYSTEM ARCHITECTURE FIGURES")
    generate_system_architecture_diagram()

    # Step 10: Reproducibility & Configuration Logging
    print("\n>>> STEP 10: EXPERIMENT CONFIGURATION & REPRODUCIBILITY ARTIFACTS")
    import sklearn
    import xgboost

    best_model_row = df_test.sort_values(by="macro_f1", ascending=False).iloc[0]
    best_model_name = best_model_row["model"]

    # Extract Ablation scores
    sty_f1 = df_ablation[df_ablation["feature_set"].str.contains("Stylometric")]["macro_f1"].values[0]
    tfidf_f1 = df_ablation[df_ablation["feature_set"].str.contains("TF-IDF")]["macro_f1"].values[0]
    hybrid_f1 = df_ablation[df_ablation["feature_set"].str.contains("Hybrid")]["macro_f1"].values[0]

    experiment_config = {
        "project_title": "A Hybrid Stylometric and Semantic Learning Framework for Detecting AI-Generated Phishing Emails",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "packages": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "joblib": joblib.__version__
            }
        },
        "dataset_summary": {
            "total_cleaned_samples": len(df_clean),
            "class_counts": df_clean['label'].value_counts().to_dict(),
            "source_counts": df_clean['source'].value_counts().to_dict()
        },
        "split_parameters": {
            "test_size": 0.20,
            "random_state": 42,
            "stratified": True
        },
        "feature_engineering": {
            "stylometric_features_count": 10,
            "tfidf_features_count": 15000,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "fused_dimensions": 15010
        },
        "best_model": {
            "name": best_model_name,
            "test_accuracy": float(best_model_row["accuracy"]),
            "test_macro_precision": float(best_model_row["macro_precision"]),
            "test_macro_recall": float(best_model_row["macro_recall"]),
            "test_macro_f1": float(best_model_row["macro_f1"]),
            "test_roc_auc": float(best_model_row["roc_auc"])
        },
        "ablation_macro_f1": {
            "stylometric_only": float(sty_f1),
            "tfidf_only": float(tfidf_f1),
            "hybrid": float(hybrid_f1)
        },
        "leakage_audit": audit_results,
        "pipeline_execution_time_seconds": round(time.time() - total_start, 2)
    }

    with open("results/experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=4)
    print("[Saved] Experiment config: results/experiment_config.json")

    # =========================================================================
    # EXACT FORMATTED RESEARCH SUMMARY AS MANDATED BY SECTION 19
    # =========================================================================
    class_counts = df_clean['label'].value_counts().to_dict()
    total_samples = len(df_clean)
    legit_count = class_counts.get("legitimate", 0)
    phish_count = class_counts.get("phishing", 0)
    ai_phish_count = class_counts.get("ai_phishing", 0)

    print("\n" + "=" * 60)
    print("                    FINAL RESEARCH SUMMARY")
    print("=" * 60)
    print(f"DATASET")
    print(f"Total: {total_samples:,}")
    print(f"Legitimate: {legit_count:,}")
    print(f"Traditional phishing: {phish_count:,}")
    print(f"AI phishing: {ai_phish_count:,}")
    print(f"")
    print(f"BEST MODEL:")
    print(f"Best model name: {best_model_name}")
    print(f"")
    print(f"TEST PERFORMANCE:")
    print(f"Accuracy: {best_model_row['accuracy']:.4f}")
    print(f"Macro Precision: {best_model_row['macro_precision']:.4f}")
    print(f"Macro Recall: {best_model_row['macro_recall']:.4f}")
    print(f"Macro F1: {best_model_row['macro_f1']:.4f}")
    print(f"ROC-AUC: {best_model_row['roc_auc']:.4f}")
    print(f"")
    print(f"ABLATION:")
    print(f"Stylometric-only Macro F1: {sty_f1:.4f}")
    print(f"TF-IDF-only Macro F1: {tfidf_f1:.4f}")
    print(f"Hybrid Macro F1: {hybrid_f1:.4f}")
    print(f"")
    print(f"LEAKAGE AUDIT:")
    print(f"Exact overlap: {audit_results['exact_overlap']}")
    print(f"Similarity = 1.00: {audit_results['pct_100']:.2f}%")
    print(f"Similarity >= 0.98: {audit_results['pct_98']:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    execute_complete_pipeline()
