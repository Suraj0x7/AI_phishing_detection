# A Hybrid Stylometric and Semantic Learning Framework for Detecting AI-Generated Phishing Emails

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Scikit-Learn | XGBoost](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange)](https://scikit-learn.org/)

An academic research project implementing an end-to-end, leak-free machine learning pipeline for detecting AI-generated phishing emails, traditional phishing attacks, and legitimate workplace correspondence.

---

## 1. Problem Statement

Modern Large Language Models (LLMs) such as GPT-4, Claude 3, and Gemini can synthesize grammatically flawless, contextually nuanced, and persuasive phishing emails at machine speed. Traditional spam filters and heuristic security gateways rely heavily on:
- Crude spelling and grammatical anomalies
- Known malicious URLs and domain blacklists
- Keyword signatures

AI-generated phishing emails bypass these legacy defenses by mimicking institutional language, omitting obvious syntax errors, and maintaining conversational coherence. This research proposes and implements a **Hybrid Stylometric and Semantic Learning Framework** that combines:
1. **10 Primary Stylometric Features**: Capturing intrinsic syntactic structures, pronoun distributions, syllabic readability, urgency frequencies, and directive imperative action patterns.
2. **Sublinear N-Gram Semantic Features (15,000 TF-IDF features)**: Representing domain-specific vocabulary and contextual phrases.
3. **Sparse Feature Fusion (15,010 dimensions)**: Standardized without data leakage to train multi-class classifiers.

The system performs three-class discrimination:
- **Class 1**: `Legitimate Email` (Enron Corporate Archive)
- **Class 2**: `Traditional Phishing Email` (Jose Nazario Corpus)
- **Class 3**: `AI-Generated Phishing Email` (Claude-3 & GPT-4 Synthetics)

---

## 2. System Architecture

```
                                EMAIL INPUT
                                     │
                                     ▼
                    SAFE TEXT PREPROCESSING (STYLE-PRESERVING)
                  (HTML Stripping, URL/Email Tokens, Casing Intact)
                                     │
                     ┌───────────────┴───────────────┐
                     │                               │
                     ▼                               ▼
          10 STYLOMETRIC FEATURES             SEMANTIC BRANCH
         (Lexicons, Syllables, Readability)  (Word Unigrams + Bigrams)
                     │                               │
                     ▼                               ▼
              StandardScaler                    TF-IDF Vector
         (Fitted Strictly on Train)           (15,000 Dimensions)
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
                           HYBRID FEATURE FUSION
                         (Sparse 15,010 Dimensions)
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
            Logistic Regression  Random Forest    XGBoost
                     │               │               │
                     └───────────────┬───────────────┘
                                     ▼
                                PREDICTION
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
    Legitimate Email        Traditional Phishing       AI-Generated Phishing
                                     │
                                     ▼
                   STYLOMETRIC EXPLAINABILITY (Z-SCORES)
                   (Analyst Warnings & Deviation Signals)
```

An editable diagram source is provided in `results/figures/system_architecture.svg` and `results/figures/system_architecture.mmd`.

---

## 3. Project Structure

```
AI_Phishing_Detection/
├── data/
│   ├── raw/                                  # Uploaded original data files
│   └── processed/
│       ├── combined_dataset.csv              # Curated standardized dataset (9,065 rows)
│       ├── cleaned_dataset.csv               # Preprocessed & deduplicated dataset (9,049 rows)
│       └── stylometric_features.csv          # Extracted 10 stylometric features
├── src/
│   ├── __init__.py
│   ├── data_loader.py                        # Automated data inspection & source verification
│   ├── preprocessing.py                      # Style-preserving text normalization
│   ├── stylometric_features.py               # 10 primary stylometric feature extractors
│   ├── tfidf_features.py                     # Sublinear N-Gram TF-IDF vectorizer
│   ├── feature_fusion.py                     # Leak-free split, scaling, & sparse fusion
│   ├── train.py                              # Stratified 5-Fold CV hyperparameter tuning
│   ├── evaluate.py                           # Untouched test set evaluation & confusion matrices
│   ├── ablation.py                           # 3-tier ablation study (Stylometric vs Semantic vs Hybrid)
│   ├── leakage_audit.py                      # Near-duplicate & TF-IDF cosine similarity audit
│   ├── explainability.py                     # Z-score class norm profiling & analyst explanations
│   └── predict.py                            # Production single email inference pipeline
├── models/
│   ├── logistic_regression.joblib            # Best Logistic Regression model (C=10.0)
│   ├── random_forest.joblib                  # Best Random Forest model (n=200)
│   ├── xgboost.joblib                        # Best XGBoost model (depth=5, n=200)
│   ├── stylometric_scaler.joblib             # Fitted StandardScaler (Train only)
│   ├── tfidf_vectorizer.joblib               # Fitted TF-IDF vectorizer (Train only)
│   ├── stylometric_class_profiles.joblib     # Class norm means & standard deviations
│   └── label_encoder.joblib                  # Multi-class target encoder mapping
├── results/
│   ├── metrics/
│   │   ├── model_comparison.csv              # Test set metrics across all models
│   │   ├── classification_reports.txt        # Full scikit-learn classification reports
│   │   ├── training_times.csv                # Execution benchmarks
│   │   └── train_test_split_summary.json     # Sample counts & class distributions
│   ├── figures/                              # High-resolution PNG plots (dpi=300)
│   │   ├── 01_class_distribution.png
│   │   ├── 02_source_distribution.png
│   │   ├── 03_stylometric_class_comparison.png
│   │   ├── 04_stylometric_correlation_matrix.png
│   │   ├── 05_cross_validation_comparison.png
│   │   ├── 06_test_model_comparison.png
│   │   ├── 07_logistic_regression_confusion_matrix.png
│   │   ├── 08_random_forest_confusion_matrix.png
│   │   ├── 09_xgboost_confusion_matrix.png
│   │   ├── 10_ablation_study.png
│   │   ├── 11_leakage_similarity_distribution.png
│   │   ├── 12_per_class_f1_comparison.png
│   │   ├── 13_model_training_time.png
│   │   └── system_architecture.png
│   ├── confusion_matrices/
│   ├── cross_validation/
│   │   ├── cv_results.csv                    # 5-fold CV metrics (mean ± std)
│   │   └── best_hyperparameters.json         # Grid search optimal parameters
│   ├── ablation/
│   │   └── ablation_results.csv              # Performance across feature regimes
│   ├── leakage_audit/
│   │   ├── similarity_results.csv            # Per-test-sample maximum cosine similarity
│   │   └── leakage_audit_report.txt          # Detailed audit findings & limitations
│   ├── predictions/
│   │   └── test_predictions.csv              # Test ground truth and model predictions
│   └── experiment_config.json                # Complete experiment configuration
├── app.py                                    # Streamlit interactive web application
├── run_pipeline.py                           # Master one-click reproducible runner
├── requirements.txt                          # Project dependency specifications
└── README.md
```

---

## 4. Dataset Inspection & Curation

The uploaded repository datasets were automatically audited and mapped as follows:

| Dataset File | Total Rows | Identified Security Class | Ground Truth Source | Status in Pipeline |
|---|---|---|---|---|
| `claude_synthetic_ai_phishing_3000.csv` | 3,000 | `ai_phishing` | Claude 3 Synthetic | **Included** |
| `GPT_4_synthetic_ai_phishing_emails_3000.csv` | 3,000 | `ai_phishing` | GPT-4 Synthetic | **Included** |
| `Nazario_5.csv` (Label 1) | 1,565 | `phishing` | Jose Nazario Phishing Corpus | **Included** |
| `Nazario_5.csv` (Label 0) | 1,500 | `legitimate` | Enron Workplace Email Corpus | **Included** |
| `gemini_generated_emails_3000.csv` | 3,000 | Ambiguous | Mixed topics, no security label | **Excluded** (Academic integrity rule) |
| `email_text.csv` | 53,668 | Corrupted | Legacy spam, 0 uppercase, no punctuation | **Excluded** (Destroys stylometry) |
| `Enron.csv` | 500,000+ | Raw archive | MIME database | Curated via `Nazario_5` subset |

**Final Cleaned Dataset Distribution**:
- Total Samples: **9,049**
- AI-Generated Phishing: **6,000 (66.3%)**
- Traditional Phishing: **1,551 (17.1%)**
- Legitimate Email: **1,498 (16.6%)**

---

## 5. Primary Stylometric Features

The pipeline extracts exactly **10 primary stylometric features** for each email:
1. `word_count`: Total word tokens.
2. `character_count`: Total length of the email text.
3. `sentence_count`: Total sentences segmented by punctuation and paragraph breaks.
4. `average_word_length`: Mean character length per word token.
5. `punctuation_count`: Total punctuation symbols (`string.punctuation`).
6. `uppercase_character_count`: Total uppercase letters (`[A-Z]`).
7. `pronoun_density`: Ratio of personal and possessive pronouns to total words.
8. `urgency_word_count`: Matches against an academic urgency lexicon (e.g., *urgent, expire, suspend, warning, action required*).
9. `imperative_verb_count`: Call-to-action imperative directive verbs (e.g., *verify, click, update, login, confirm*).
10. `flesch_reading_ease`: Readability index derived from syllabic density and sentence length:
    $$FRE = 206.835 - 1.015 \left(\frac{\text{words}}{\text{sentences}}\right) - 84.6 \left(\frac{\text{syllables}}{\text{words}}\right)$$

---

## 6. Installation & Execution

### Prerequisites
- Python 3.10+ (tested on Python 3.12.10)
- Operating System: Windows, macOS, or Linux

### Installation
```bash
pip install -r requirements.txt
```

### Reproduce Full Experiment
To run the full end-to-end pipeline (data inspection, preprocessing, feature engineering, 5-fold CV, evaluation, ablation, leakage audit, and figure generation):
```bash
python run_pipeline.py
```

### Launch Interactive Web Application
```bash
streamlit run app.py
```

---

## 7. Experimental Results

### Held-Out Test Set Performance (1,810 Samples)
Evaluated on the untouched 20% test split:

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC (OvR) |
|---|---|---|---|---|---|---|
| **Logistic Regression (C=10.0)** | **0.9950** | **0.9901** | **0.9902** | **0.9901** | **0.9902** | **0.9998** |
| **XGBoost (n=200, depth=5)** | 0.9934 | 0.9870 | 0.9869 | 0.9870 | 0.9869 | 0.9996 |
| **Random Forest (n=200)** | 0.9917 | 0.9836 | 0.9836 | 0.9836 | 0.9836 | 0.9998 |

*Best overall model: **Logistic Regression** (Macro F1: 0.9902, Accuracy: 0.9950).*

### Per-Class F1 Score Breakdown

| Target Class | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| **Legitimate Email** | 0.9850 | 0.9750 | 0.9802 |
| **Traditional Phishing** | 0.9855 | 0.9758 | 0.9805 |
| **AI-Generated Phishing** | **1.0000** | **1.0000** | **1.0000** |

---

## 8. Ablation Study

Controlled comparison across identical train/test splits:

| Feature Regime | Feature Count | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|
| **A. Stylometric Only** | 10 | 0.9083 | 0.8872 | **0.8567** | 0.9663 |
| **B. TF-IDF Only** | 15,000 | 0.9967 | 0.9942 | **0.9945** | 0.9999 |
| **C. Hybrid Fusion** | 15,010 | 0.9950 | 0.9901 | **0.9902** | 0.9998 |

**Scientific Takeaway**: While pure TF-IDF yields slightly higher nominal in-distribution scores, the 10-feature stylometric model alone achieves >90% accuracy and 0.9663 ROC-AUC. In the Hybrid model, stylometric features provide essential syntactic invariance against vocabulary shifts and enable human-interpretable Z-score explanations for security analysts.

---

## 9. Data Leakage & Template Similarity Audit

To ensure scientific honesty and rigor regarding high test accuracy:
- **Exact Train-Test Overlap**: **0 pairs** (Strict verification of zero duplicate records).
- **Maximum Train-Test TF-IDF Cosine Similarity**:
  - Similarity == 1.00: **32.76%**
  - Similarity >= 0.98: **34.53%**
  - Similarity >= 0.95: **39.50%**
  - Similarity >= 0.90: **54.09%**
  - Median Maximum Similarity: **0.9106**

### Transparent Research Limitations
1. **Template Redundancy in Synthetic Datasets**: Synthetic phishing emails generated via batch LLM prompts exhibit high structural repetition across common topics (e.g., Microsoft 365 credential verification, banking security reviews).
2. **Impact on Reported Accuracy**: High accuracy (>99%) reflects both classifier discrimination and underlying synthetic template regularity. Real-world zero-day attacks with novel phrasing will demonstrate greater variance. The stylometric branch acts as a critical anchor against template overfitting.

---

## 10. Stylometric Explainability

For every incoming email, the framework computes Z-score deviations against training-set class norms:
$$z_i = \frac{x_i - \mu_{c,i}}{\sigma_{c,i}}$$

Security analysts receive plain-language, ranked risk indicators:
- `Flesch Reading Ease: VERY LOW (Value: 5.0, Class Norm: 49.2)`
- `Urgency-Word Frequency: HIGH (Value: 6 words, Class Norm: 2.8)`
- `Pronoun Density: HIGH (Value: 13.0%, Class Norm: 8.0%)`
- `Imperative Expressions: HIGH (Value: 3 verbs, Class Norm: 0.8)`

---

## 11. Citation & Academic Credits

If you utilize this framework or codebase in your academic research or thesis, please cite:
```bibtex
@mastersthesis{hybrid_phishing_detection_2026,
  title     = {A Hybrid Stylometric and Semantic Learning Framework for Detecting AI-Generated Phishing Emails},
  author    = {M.Tech Research Project},
  year      = {2026},
  school    = {VIT}
}
```
