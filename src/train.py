import os
import sys
import time
import json
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from src.feature_fusion import prepare_splits_and_hybrid_features

sns.set_theme(style="whitegrid")

LABEL_MAPPING = {
    "legitimate": 0,
    "phishing": 1,
    "ai_phishing": 2
}
INT_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}
CLASS_NAMES = ["legitimate", "phishing", "ai_phishing"]

def train_and_tune_models(data_dict=None):
    """
    Trains Logistic Regression, Random Forest, and XGBoost using Stratified 5-Fold Cross Validation.
    Performs hyperparameter selection strictly on training data using Macro-F1 as primary metric.
    """
    if data_dict is None:
        data_dict = prepare_splits_and_hybrid_features()

    X_train = data_dict["X_hybrid_train"]
    y_train_str = data_dict["y_train"]
    y_train = np.array([LABEL_MAPPING[lbl] for lbl in y_train_str], dtype=int)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/cross_validation", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # Save label encoder/mapping
    joblib.dump({"label_to_int": LABEL_MAPPING, "int_to_label": INT_TO_LABEL, "classes": CLASS_NAMES}, 
                "models/label_encoder.joblib")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Candidate Grids
    param_grids = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42, solver='lbfgs'),
            "params": {
                "C": [0.1, 1.0, 10.0]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, None]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 7]
            }
        }
    }

    trained_models = {}
    cv_records = []
    training_times = []
    best_params_all = {}

    print("\n" + "="*70)
    print("      STRATIFIED 5-FOLD CROSS-VALIDATION & HYPERPARAMETER TUNING")
    print("="*70)

    scoring = {
        'f1_macro': 'f1_macro',
        'accuracy': 'accuracy',
        'roc_auc_ovr': 'roc_auc_ovr'
    }

    for name, config in param_grids.items():
        print(f"\n--- Optimizing {name} ---")
        start_time = time.time()

        # On Windows, XGBoost parallelizes natively with OpenMP; running GridSearchCV with n_jobs=1 avoids IPC crashes
        grid_n_jobs = 1 if name == "XGBoost" else -1

        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring='f1_macro',
            n_jobs=grid_n_jobs,
            refit=True,
            verbose=1
        )
        grid.fit(X_train, y_train)
        elapsed_sec = time.time() - start_time

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_params_all[name] = best_params
        trained_models[name] = best_model

        print(f"  Best Parameters: {best_params}")
        print(f"  Best Macro-F1 (CV): {grid.best_score_:.4f}")
        print(f"  Tuning & Training Time: {elapsed_sec:.2f} s")

        # Detailed 5-fold evaluation for best model
        cv_eval = cross_validate(best_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=grid_n_jobs)
        
        f1_mean, f1_std = cv_eval['test_f1_macro'].mean(), cv_eval['test_f1_macro'].std()
        acc_mean, acc_std = cv_eval['test_accuracy'].mean(), cv_eval['test_accuracy'].std()
        auc_mean, auc_std = cv_eval['test_roc_auc_ovr'].mean(), cv_eval['test_roc_auc_ovr'].std()

        cv_records.append({
            "model": name,
            "best_parameters": str(best_params),
            "cv_macro_f1_mean": round(f1_mean, 4),
            "cv_macro_f1_std": round(f1_std, 4),
            "cv_accuracy_mean": round(acc_mean, 4),
            "cv_accuracy_std": round(acc_std, 4),
            "cv_roc_auc_mean": round(auc_mean, 4),
            "cv_roc_auc_std": round(auc_std, 4),
            "training_time_seconds": round(elapsed_sec, 2)
        })

        training_times.append({
            "model": name,
            "training_time_seconds": round(elapsed_sec, 2)
        })

        # Save individual model
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(best_model, os.path.join("models", filename))
        print(f"  [Saved] Model artifact: models/{filename}")

    # Save CV Results & Hyperparameters
    df_cv = pd.DataFrame(cv_records)
    df_cv.to_csv("results/cross_validation/cv_results.csv", index=False)
    
    df_time = pd.DataFrame(training_times)
    df_time.to_csv("results/metrics/training_times.csv", index=False)

    with open("results/cross_validation/best_hyperparameters.json", "w") as f:
        json.dump(best_params_all, f, indent=4)

    print("\n" + "="*70)
    print("                     CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(df_cv[["model", "cv_macro_f1_mean", "cv_accuracy_mean", "cv_roc_auc_mean", "training_time_seconds"]])

    # Generate Figures: 05_cross_validation_comparison.png and 13_model_training_time.png
    plot_cv_results(df_cv)
    plot_training_times(df_time)

    return trained_models, df_cv

def plot_cv_results(df_cv, figures_dir="results/figures"):
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    models = df_cv['model']
    x = np.arange(len(models))
    width = 0.25

    rects1 = ax.bar(x - width, df_cv['cv_macro_f1_mean'], width, yerr=df_cv['cv_macro_f1_std'], 
                    label='Macro F1 (Mean ± SD)', color='#3498db', capsize=5, edgecolor='black', linewidth=1.1)
    rects2 = ax.bar(x, df_cv['cv_accuracy_mean'], width, yerr=df_cv['cv_accuracy_std'], 
                    label='Accuracy (Mean ± SD)', color='#2ecc71', capsize=5, edgecolor='black', linewidth=1.1)
    rects3 = ax.bar(x + width, df_cv['cv_roc_auc_mean'], width, yerr=df_cv['cv_roc_auc_std'], 
                    label='ROC-AUC (Mean ± SD)', color='#9b59b6', capsize=5, edgecolor='black', linewidth=1.1)

    ax.set_title("Stratified 5-Fold Cross-Validation Performance Comparison", fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight='bold')
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc='lower right', fontsize=10)

    # Value labels
    for rects in [rects1, rects2, rects3]:
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2., h + 0.015, f"{h:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "05_cross_validation_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] CV comparison plot: {out_path}")

def plot_training_times(df_time, figures_dir="results/figures"):
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(df_time['model'], df_time['training_time_seconds'], color=['#3498db', '#e67e22', '#2ecc71'], 
                  width=0.5, edgecolor='black', linewidth=1.2)
    ax.set_title("Model Training and Cross-Validation Tuning Time", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Machine Learning Classifier", fontsize=11, fontweight='bold')
    ax.set_ylabel("Execution Time (Seconds)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(df_time['training_time_seconds']) * 1.2)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.0, f"{h:.1f} s", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "13_model_training_time.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Training time plot: {out_path}")

if __name__ == "__main__":
    train_and_tune_models()
