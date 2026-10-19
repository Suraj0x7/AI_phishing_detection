import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.stylometric_features import FEATURE_COLUMNS

sns.set_theme(style="whitegrid")

LABEL_MAPPING = {"legitimate": 0, "phishing": 1, "ai_phishing": 2}

def run_ablation_study():
    """
    Executes controlled ablation study across three feature regimes:
    A. Stylometric Features Only (10 features)
    B. TF-IDF Semantic Features Only (15,000 features)
    C. Hybrid Features (Stylometric + TF-IDF, 15,010 features)
    Uses the exact same train/test split and Logistic Regression (C=10.0, balanced).
    """
    print("\n" + "="*70)
    print("                    ABLATION STUDY EXPERIMENT")
    print("="*70)

    split_cache = joblib.load("models/split_cache.joblib")
    scaler = joblib.load("models/stylometric_scaler.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

    df_clean = pd.read_csv("data/processed/cleaned_dataset.csv")
    df_sty = pd.read_csv("data/processed/stylometric_features.csv")

    idx_train = split_cache["idx_train"]
    idx_test = split_cache["idx_test"]
    y_train = np.array([LABEL_MAPPING[l] for l in split_cache["y_train"]], dtype=int)
    y_test = np.array([LABEL_MAPPING[l] for l in split_cache["y_test"]], dtype=int)

    # 1. Stylometric Features (Scaled)
    X_sty_train = df_sty[FEATURE_COLUMNS].values[idx_train]
    X_sty_test = df_sty[FEATURE_COLUMNS].values[idx_test]
    X_sty_train_scaled = scaler.transform(X_sty_train).astype(np.float32)
    X_sty_test_scaled = scaler.transform(X_sty_test).astype(np.float32)

    # 2. TF-IDF Features
    X_text_train = split_cache["X_text_train"]
    X_text_test = split_cache["X_text_test"]
    X_tfidf_train = vectorizer.transform(X_text_train).astype(np.float32)
    X_tfidf_test = vectorizer.transform(X_text_test).astype(np.float32)

    # 3. Hybrid Features
    X_sty_train_sparse = sp.csr_matrix(X_sty_train_scaled, dtype=np.float32)
    X_sty_test_sparse = sp.csr_matrix(X_sty_test_scaled, dtype=np.float32)

    X_hybrid_train = sp.hstack([X_sty_train_sparse, X_tfidf_train], format='csr', dtype=np.float32)
    X_hybrid_train.sort_indices()
    X_hybrid_test = sp.hstack([X_sty_test_sparse, X_tfidf_test], format='csr', dtype=np.float32)
    X_hybrid_test.sort_indices()

    configurations = [
        ("A. Stylometric Only", X_sty_train_scaled, X_sty_test_scaled, 10),
        ("B. TF-IDF Only", X_tfidf_train, X_tfidf_test, 15000),
        ("C. Hybrid (Stylometric + TF-IDF)", X_hybrid_train, X_hybrid_test, 15010)
    ]

    records = []
    os.makedirs("results/ablation", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    for name, X_tr, X_te, n_feats in configurations:
        print(f"\nRunning Configuration: {name} (Features: {n_feats})...")
        clf = LogisticRegression(C=10.0, max_iter=500, class_weight='balanced', random_state=42)
        clf.fit(X_tr, y_train)

        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        records.append({
            "feature_set": name,
            "feature_count": n_feats,
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "roc_auc": round(roc_auc, 4)
        })

    df_ablation = pd.DataFrame(records)
    out_csv = "results/ablation/ablation_results.csv"
    df_ablation.to_csv(out_csv, index=False)
    print(f"\n[Saved] Ablation results table: {out_csv}")
    print(df_ablation[["feature_set", "feature_count", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]])

    plot_ablation_results(df_ablation)
    return df_ablation

def plot_ablation_results(df_ablation, figures_dir="results/figures"):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(df_ablation))
    width = 0.22

    r1 = ax.bar(x - width, df_ablation['accuracy'], width, label='Accuracy', color='#2ecc71', edgecolor='black', linewidth=1.1)
    r2 = ax.bar(x, df_ablation['balanced_accuracy'], width, label='Balanced Accuracy', color='#e67e22', edgecolor='black', linewidth=1.1)
    r3 = ax.bar(x + width, df_ablation['macro_f1'], width, label='Macro F1', color='#3498db', edgecolor='black', linewidth=1.1)

    ax.set_title("Ablation Study: Stylometric vs. Semantic vs. Hybrid Features", fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(["Stylometric Only\n(10 features)", "TF-IDF Only\n(15,000 features)", "Hybrid Fusion\n(15,010 features)"], 
                       fontsize=10, fontweight='bold')
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight='bold')
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower right', fontsize=10)

    for rects in [r1, r2, r3]:
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2., h + 0.015, f"{h:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "10_ablation_study.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Ablation study plot: {out_path}")

if __name__ == "__main__":
    run_ablation_study()
