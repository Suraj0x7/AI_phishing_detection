import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True):
    """
    Returns an un-fitted TfidfVectorizer configured according to research specifications.
    Strictly fitted ONLY on training data to prevent train-test data leakage.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        lowercase=True,
        token_pattern=r'(?u)\b\w+\b'
    )

def fit_tfidf(texts_train, save_path="models/tfidf_vectorizer.joblib", max_features=15000):
    """
    Fits TF-IDF vectorizer strictly on training texts and saves the fitted model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vectorizer = get_tfidf_vectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(texts_train)
    joblib.dump(vectorizer, save_path)
    print(f"[Saved] Fitted TF-IDF Vectorizer to: {save_path}")
    print(f"TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}")
    return vectorizer, X_train_tfidf

def transform_tfidf(texts, vectorizer_or_path="models/tfidf_vectorizer.joblib"):
    """
    Transforms arbitrary texts using the training-fitted vectorizer.
    """
    if isinstance(vectorizer_or_path, str):
        if not os.path.exists(vectorizer_or_path):
            raise FileNotFoundError(f"TF-IDF vectorizer not found at: {vectorizer_or_path}")
        vectorizer = joblib.load(vectorizer_or_path)
    else:
        vectorizer = vectorizer_or_path

    return vectorizer.transform(texts)
