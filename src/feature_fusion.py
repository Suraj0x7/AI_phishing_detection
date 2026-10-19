import os
import sys
import json
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.stylometric_features import FEATURE_COLUMNS, extract_stylometric_features
from src.tfidf_features import fit_tfidf, transform_tfidf

def prepare_splits_and_hybrid_features(
    dataset_fp="data/processed/cleaned_dataset.csv",
    stylometry_fp="data/processed/stylometric_features.csv",
    models_dir="models",
    metrics_dir="results/metrics",
    test_size=0.20,
    random_state=42
):
    """
    Orchestrates the leak-free feature engineering and fusion pipeline:
    1. Splits dataset into Train (80%) and Test (20%) using stratified sampling.
    2. Fits StandardScaler ONLY on training stylometric features.
    3. Fits TfidfVectorizer ONLY on training text.
    4. Transforms training and testing splits using fitted artifacts.
    5. Fuses standardized stylometric features and sparse TF-IDF vectors into a hybrid vector.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    df_clean = pd.read_csv(dataset_fp)
    df_sty = pd.read_csv(stylometry_fp)

    # Ensure standard numpy object ndarrays (lightweight pointer arrays, memory safe)
    X_text = np.array(df_clean['text'].astype(str).tolist(), dtype=object)
    y = np.array(df_clean['label'].astype(str).tolist(), dtype=object)
    sources = np.array(df_clean['source'].astype(str).tolist(), dtype=object)
    X_sty = np.array(df_sty[FEATURE_COLUMNS].values, dtype=float)

    # Step 1: Train / Test Split
    indices = np.arange(len(df_clean))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_text_train, X_text_test = X_text[idx_train], X_text[idx_test]
    X_sty_train, X_sty_test = X_sty[idx_train], X_sty[idx_test]
    src_train, src_test = sources[idx_train], sources[idx_test]

    # Save split summary
    split_info = {
        "total_samples": len(df_clean),
        "training_samples": len(idx_train),
        "testing_samples": len(idx_test),
        "train_class_distribution": pd.Series(y_train).value_counts().to_dict(),
        "test_class_distribution": pd.Series(y_test).value_counts().to_dict(),
        "test_size": test_size,
        "random_state": random_state
    }

    with open(os.path.join(metrics_dir, "train_test_split_summary.json"), "w") as f:
        json.dump(split_info, f, indent=4)

    print("\n" + "="*60)
    print("           TRAIN / TEST STRATIFIED SPLIT SUMMARY")
    print("="*60)
    print(f"Total Samples: {split_info['total_samples']}")
    print(f"Training Samples (80%): {split_info['training_samples']}")
    print(f"Testing Samples  (20%): {split_info['testing_samples']}")
    print(f"Training Class Distribution: {split_info['train_class_distribution']}")
    print(f"Testing Class Distribution:  {split_info['test_class_distribution']}")

    # Step 2: Fit Stylometric Scaler ONLY on Training Data
    scaler = StandardScaler()
    X_sty_train_scaled = scaler.fit_transform(X_sty_train)
    X_sty_test_scaled = scaler.transform(X_sty_test)

    scaler_path = os.path.join(models_dir, "stylometric_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[Saved] Fitted Stylometric Scaler to: {scaler_path}")

    # Compute and save training class stylometric profiles for explainability
    class_profiles = {}
    for label_name in np.unique(y_train):
        label_mask = (y_train == label_name)
        subset_sty = X_sty_train[label_mask]
        class_profiles[label_name] = {
            "mean": subset_sty.mean(axis=0).tolist(),
            "std": subset_sty.std(axis=0).tolist(),
            "feature_names": FEATURE_COLUMNS
        }
    profile_path = os.path.join(models_dir, "stylometric_class_profiles.joblib")
    joblib.dump(class_profiles, profile_path)
    print(f"[Saved] Training Class Stylometric Profiles to: {profile_path}")

    # Step 3: Fit TF-IDF Vectorizer ONLY on Training Text
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    vectorizer, X_tfidf_train = fit_tfidf(X_text_train, save_path=tfidf_path, max_features=15000)
    X_tfidf_test = vectorizer.transform(X_text_test)

    # Step 4: Hybrid Feature Fusion (Sparse Concat with float32 and sorted indices)
    X_sty_train_sparse = sparse.csr_matrix(X_sty_train_scaled, dtype=np.float32)
    X_sty_test_sparse = sparse.csr_matrix(X_sty_test_scaled, dtype=np.float32)
    X_tfidf_train = X_tfidf_train.astype(np.float32)
    X_tfidf_test = X_tfidf_test.astype(np.float32)

    X_hybrid_train = sparse.hstack([X_sty_train_sparse, X_tfidf_train], format='csr', dtype=np.float32)
    X_hybrid_train.sort_indices()
    X_hybrid_test = sparse.hstack([X_sty_test_sparse, X_tfidf_test], format='csr', dtype=np.float32)
    X_hybrid_test.sort_indices()

    print(f"\nFeature Dimensions:")
    print(f"  Stylometric Features: {X_sty_train_scaled.shape[1]}")
    print(f"  TF-IDF Features:      {X_tfidf_train.shape[1]}")
    print(f"  Fused Hybrid Train:   {X_hybrid_train.shape}")
    print(f"  Fused Hybrid Test:    {X_hybrid_test.shape}")

    # Save split indices and metadata for reproducible evaluation and ablation
    split_cache = {
        "idx_train": idx_train,
        "idx_test": idx_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_text_train": X_text_train,
        "X_text_test": X_text_test,
        "src_train": src_train,
        "src_test": src_test
    }
    joblib.dump(split_cache, os.path.join(models_dir, "split_cache.joblib"))

    return {
        "X_hybrid_train": X_hybrid_train,
        "X_hybrid_test": X_hybrid_test,
        "X_sty_train_scaled": X_sty_train_scaled,
        "X_sty_test_scaled": X_sty_test_scaled,
        "X_tfidf_train": X_tfidf_train,
        "X_tfidf_test": X_tfidf_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_text_train": X_text_train,
        "X_text_test": X_text_test,
        "split_info": split_info
    }

def transform_single_email_hybrid(
    cleaned_text: str,
    scaler_path="models/stylometric_scaler.joblib",
    vectorizer_path="models/tfidf_vectorizer.joblib"
):
    """
    Transforms a single cleaned email into the exact fused hybrid feature vector.
    """
    scaler = joblib.load(scaler_path)
    vectorizer = joblib.load(vectorizer_path)

    sty_dict = extract_stylometric_features(cleaned_text)
    sty_vals = np.array([[sty_dict[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
    sty_scaled = scaler.transform(sty_vals).astype(np.float32)
    sty_sparse = sparse.csr_matrix(sty_scaled, dtype=np.float32)

    tfidf_sparse = vectorizer.transform([cleaned_text]).astype(np.float32)
    hybrid_vec = sparse.hstack([sty_sparse, tfidf_sparse], format='csr', dtype=np.float32)
    hybrid_vec.sort_indices()

    return hybrid_vec, sty_dict

if __name__ == "__main__":
    prepare_splits_and_hybrid_features()
