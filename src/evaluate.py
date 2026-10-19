import os
import sys
import joblib
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

sns.set_theme(style="whitegrid")

LABEL_MAPPING = {"legitimate": 0, "phishing": 1, "ai_phishing": 2}
INT_TO_LABEL = {0: "legitimate", 1: "phishing", 2: "ai_phishing"}
CLASS_NAMES = ["legitimate", "phishing", "ai_phishing"]
DISPLAY_NAMES = ["Legitimate", "Traditional Phishing", "AI-Generated Phishing"]

def evaluate_models():
    """
    Evaluates all trained classifiers strictly on the held-out untouched test set.
    Generates publication-quality confusion matrices and metrics comparisons.
    """
    print("\n" + "="*70)
    print("      HELD-OUT TEST SET COMPREHENSIVE EVALUATION")
    print("="*70)

    # 1. Load Split & Models
    split_cache = joblib.load("models/split_cache.joblib")
    scaler = joblib.load("models/stylometric_scaler.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

    # Load cleaned dataset & stylometric features
    import scipy.sparse as sp
    from src.stylometric_features import FEATURE_COLUMNS
    df_clean = pd.read_csv("data/processed/cleaned_dataset.csv")
    df_sty = pd.read_csv("data/processed/stylometric_features.csv")

    idx_test = split_cache["idx_test"]
    y_test_str = split_cache["y_test"]
    y_test = np.array([LABEL_MAPPING[l] for l in y_test_str], dtype=int)
    X_text_test = split_cache["X_text_test"]

    X_sty_test = df_sty[FEATURE_COLUMNS].values[idx_test]
    X_sty_test_scaled = scaler.transform(X_sty_test).astype(np.float32)
    X_sty_test_sparse = sp.csr_matrix(X_sty_test_scaled, dtype=np.float32)

    X_tfidf_test = vectorizer.transform(X_text_test).astype(np.float32)

    X_hybrid_test = sp.hstack([X_sty_test_sparse, X_tfidf_test], format='csr', dtype=np.float32)
    X_hybrid_test.sort_indices()

    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.joblib"),
        "Random Forest": joblib.load("models/random_forest.joblib"),
        "XGBoost": joblib.load("models/xgboost.joblib")
    }

    results = []
    reports_text = []
    predictions_dict = {
        "text": X_text_test,
        "true_label": [INT_TO_LABEL[i] for i in y_test]
    }

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)

    for name, model in models.items():
        print(f"\nEvaluating: {name}...")
        y_pred = model.predict(X_hybrid_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_hybrid_test)
        else:
            y_proba = None

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Per-class scores
        p_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        r_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f_class = f1_score(y_test, y_pred, average=None, zero_division=0)

        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        else:
            roc_auc = np.nan

        results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "roc_auc": round(roc_auc, 4),
            "legitimate_precision": round(p_class[0], 4),
            "legitimate_recall": round(r_class[0], 4),
            "legitimate_f1": round(f_class[0], 4),
            "phishing_precision": round(p_class[1], 4),
            "phishing_recall": round(r_class[1], 4),
            "phishing_f1": round(f_class[1], 4),
            "ai_phishing_precision": round(p_class[2], 4),
            "ai_phishing_recall": round(r_class[2], 4),
            "ai_phishing_f1": round(f_class[2], 4)
        })

        # Save predictions
        predictions_dict[f"{name}_pred"] = [INT_TO_LABEL[i] for i in y_pred]

        # Classification Report
        cr = classification_report(y_test, y_pred, target_names=DISPLAY_NAMES, digits=4)
        reports_text.append(f"====================================================\n"
                            f"CLASSIFICATION REPORT: {name}\n"
                            f"====================================================\n"
                            f"{cr}\n\n")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        plot_confusion_matrix(cm, name)

    # Save Model Comparison CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/metrics/model_comparison.csv", index=False)
    print(f"\n[Saved] Test model comparison to: results/metrics/model_comparison.csv")

    # Save Classification Reports
    with open("results/metrics/classification_reports.txt", "w") as f:
        f.writelines(reports_text)
    print(f"[Saved] Classification reports to: results/metrics/classification_reports.txt")

    # Save Predictions CSV
    df_preds = pd.DataFrame(predictions_dict)
    df_preds.to_csv("results/predictions/test_predictions.csv", index=False)
    print(f"[Saved] Test predictions to: results/predictions/test_predictions.csv")

    # Print summary table
    print("\n" + "="*70)
    print("                    TEST EVALUATION METRICS")
    print("="*70)
    print(df_res[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc"]])

    # Generate Figures
    plot_test_model_comparison(df_res)
    plot_per_class_f1(df_res)

    return df_res

def plot_confusion_matrix(cm, model_name, figures_dir="results/figures"):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    # Format annotations with count and row percentage
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_percent[i, j]:.1f}%)"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True, ax=ax,
                xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                linewidths=1.2, linecolor='black')

    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("True Target Class", fontsize=11, fontweight='bold', labelpad=10)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    # Specific numbered filenames
    if "logistic" in model_name.lower():
        num_file = "07_logistic_regression_confusion_matrix.png"
    elif "random" in model_name.lower():
        num_file = "08_random_forest_confusion_matrix.png"
    else:
        num_file = "09_xgboost_confusion_matrix.png"

    fig.savefig(os.path.join(figures_dir, num_file), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join("results/confusion_matrices", filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] Confusion matrix figure: results/figures/{num_file}")

def plot_test_model_comparison(df_res, figures_dir="results/figures"):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    models = df_res['model']
    x = np.arange(len(models))
    width = 0.22

    r1 = ax.bar(x - width, df_res['accuracy'], width, label='Accuracy', color='#2ecc71', edgecolor='black', linewidth=1.1)
    r2 = ax.bar(x, df_res['balanced_accuracy'], width, label='Balanced Accuracy', color='#e67e22', edgecolor='black', linewidth=1.1)
    r3 = ax.bar(x + width, df_res['macro_f1'], width, label='Macro F1', color='#3498db', edgecolor='black', linewidth=1.1)

    ax.set_title("Test Set Overall Performance Comparison", fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight='bold')
    ax.set_ylim(0.75, 1.05)
    ax.legend(loc='lower right', fontsize=10)

    for rects in [r1, r2, r3]:
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2., h + 0.012, f"{h:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "06_test_model_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Test model comparison plot: {out_path}")

def plot_per_class_f1(df_res, figures_dir="results/figures"):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    models = df_res['model']
    x = np.arange(len(models))
    width = 0.25

    r1 = ax.bar(x - width, df_res['legitimate_f1'], width, label='Legitimate Email F1', color='#2ecc71', edgecolor='black', linewidth=1.1)
    r2 = ax.bar(x, df_res['phishing_f1'], width, label='Traditional Phishing F1', color='#e74c3c', edgecolor='black', linewidth=1.1)
    r3 = ax.bar(x + width, df_res['ai_phishing_f1'], width, label='AI-Generated Phishing F1', color='#3498db', edgecolor='black', linewidth=1.1)

    ax.set_title("Per-Class F1 Score Comparison Across Models", fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylabel("F1 Score (0.0 - 1.0)", fontsize=11, fontweight='bold')
    ax.set_ylim(0.75, 1.05)
    ax.legend(loc='lower right', fontsize=10)

    for rects in [r1, r2, r3]:
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2., h + 0.012, f"{h:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "12_per_class_f1_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Per-class F1 comparison plot: {out_path}")

if __name__ == "__main__":
    evaluate_models()
