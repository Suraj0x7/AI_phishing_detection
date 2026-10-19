import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Documented Research Lexicons
PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves"
}

URGENCY_LEXICON = {
    "urgent", "urgently", "immediate", "immediately", "action", "required",
    "expire", "expires", "expiring", "expiration", "warning", "alert", "suspended",
    "suspend", "suspension", "terminated", "terminate", "verify", "verification",
    "critical", "attention", "breach", "locked", "unauthorized", "penalty",
    "restricted", "deadline", "overdue", "compromised", "threat", "danger",
    "reactivate", "re-activate", "limit", "limited", "fraud", "unusual"
}

IMPERATIVE_VERBS = {
    "click", "verify", "update", "confirm", "check", "login", "log", "download",
    "review", "submit", "send", "contact", "call", "visit", "open", "reactivate",
    "renew", "protect", "follow", "access", "reset", "authenticate", "resolve",
    "reply", "validate", "ensure", "avoid", "act", "proceed", "pay", "remit"
}

FEATURE_COLUMNS = [
    "word_count",
    "character_count",
    "sentence_count",
    "average_word_length",
    "punctuation_count",
    "uppercase_character_count",
    "pronoun_density",
    "urgency_word_count",
    "imperative_verb_count",
    "flesch_reading_ease"
]

def count_syllables_in_word(word: str) -> int:
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 1
    vowels = "aeiouy"
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and not word.endswith('le') and len(word) > 2 and word[-2] not in vowels:
        count = max(1, count - 1)
    return max(1, count)

def extract_stylometric_features(text: str) -> dict:
    """
    Extracts EXACTLY the 10 defined stylometric writing-style features.
    Handles empty, malformed, or unusual text safely.
    """
    if not isinstance(text, str) or not text.strip():
        return {col: 0.0 for col in FEATURE_COLUMNS}

    raw_text = text
    char_count = len(raw_text)
    uppercase_count = sum(1 for c in raw_text if c.isupper())
    punct_count = sum(1 for c in raw_text if c in string.punctuation)

    words = re.findall(r'\b[A-Za-z0-9\'-]+\b', raw_text)
    word_count = len(words)

    sentences = [s.strip() for s in re.split(r'[.!?]+|\n\n+', raw_text) if s.strip()]
    sentence_count = max(1, len(sentences))

    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

    lower_words = [w.lower() for w in words]
    pronoun_count = sum(1 for w in lower_words if w in PRONOUNS) if word_count > 0 else 0
    pronoun_density = pronoun_count / word_count if word_count > 0 else 0.0

    urgency_count = sum(1 for w in lower_words if w in URGENCY_LEXICON)

    imperative_count = 0
    for s in sentences:
        s_words = re.findall(r'\b[A-Za-z0-9\'-]+\b', s)
        if s_words:
            if s_words[0].lower() in IMPERATIVE_VERBS:
                imperative_count += 1
            for idx, w in enumerate(s_words[:-1]):
                if w.lower() == 'please' and s_words[idx+1].lower() in IMPERATIVE_VERBS:
                    imperative_count += 1

    if word_count > 0 and sentence_count > 0:
        total_syllables = sum(count_syllables_in_word(w) for w in words)
        words_per_sentence = word_count / sentence_count
        syllables_per_word = total_syllables / word_count
        fre = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        fre = max(-100.0, min(120.0, fre))
    else:
        fre = 0.0

    return {
        "word_count": float(word_count),
        "character_count": float(char_count),
        "sentence_count": float(sentence_count),
        "average_word_length": float(round(avg_word_len, 4)),
        "punctuation_count": float(punct_count),
        "uppercase_character_count": float(uppercase_count),
        "pronoun_density": float(round(pronoun_density, 4)),
        "urgency_word_count": float(urgency_count),
        "imperative_verb_count": float(imperative_count),
        "flesch_reading_ease": float(round(fre, 4))
    }

def process_dataset_stylometry(input_fp="data/processed/cleaned_dataset.csv", output_fp="data/processed/stylometric_features.csv"):
    if not os.path.exists(input_fp):
        raise FileNotFoundError(f"Input file not found: {input_fp}. Run preprocessing.py first.")

    df = pd.read_csv(input_fp)
    print(f"\n--- EXTRACTING 10 STYLOMETRIC FEATURES ({len(df)} emails) ---")

    feature_records = []
    for text in df['text']:
        feat = extract_stylometric_features(text)
        feature_records.append(feat)

    features_df = pd.DataFrame(feature_records)
    features_df['label'] = df['label'].values
    features_df['source'] = df['source'].values

    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    features_df.to_csv(output_fp, index=False)
    print(f"[Saved] Stylometric features dataset to: {output_fp}")

    # Generate Figures
    generate_stylometric_plots(features_df)

    return features_df

def generate_stylometric_plots(df, figures_dir="results/figures"):
    os.makedirs(figures_dir, exist_ok=True)
    palette = {"legitimate": "#2ecc71", "phishing": "#e74c3c", "ai_phishing": "#3498db"}

    # 1. Class Comparison Figure (03_stylometric_class_comparison.png)
    # Compare key discriminative features across the 3 classes
    key_features = [
        ("urgency_word_count", "Urgency Word Count"),
        ("imperative_verb_count", "Imperative Verb Count"),
        ("uppercase_character_count", "Uppercase Count (log scale)"),
        ("flesch_reading_ease", "Flesch Reading Ease"),
        ("pronoun_density", "Pronoun Density"),
        ("average_word_length", "Average Word Length")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    axes = axes.flatten()

    for idx, (col, title) in enumerate(key_features):
        plot_data = df.copy()
        if col == "uppercase_character_count":
            plot_data[col] = np.log1p(plot_data[col])

        sns.barplot(data=plot_data, x='label', y=col, ax=axes[idx], hue='label', palette=palette, 
                    order=["legitimate", "phishing", "ai_phishing"], errorbar=None, edgecolor='black', linewidth=1.1, legend=False)
        axes[idx].set_title(title, fontsize=12, fontweight='bold', pad=8)
        axes[idx].set_xlabel("Class", fontsize=10, fontweight='bold')
        axes[idx].set_ylabel("Mean Value", fontsize=10, fontweight='bold')
        axes[idx].set_xticks([0, 1, 2])
        axes[idx].set_xticklabels(["Legitimate", "Trad. Phishing", "AI Phishing"])

    plt.suptitle("Stylometric Feature Profile Comparison Across Target Classes", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    p1 = os.path.join(figures_dir, "03_stylometric_class_comparison.png")
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "stylometric_class_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Correlation Matrix Figure (04_stylometric_correlation_matrix.png)
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax, cbar_kws={'label': 'Pearson Correlation'},
                linewidths=0.5, linecolor='gray')
    ax.set_title("Correlation Matrix of 10 Primary Stylometric Features", fontsize=14, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(figures_dir, "04_stylometric_correlation_matrix.png")
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "stylometric_correlation_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Feature Distribution Figure (stylometric_feature_distribution.png)
    fig, axes = plt.subplots(5, 2, figsize=(14, 18), dpi=300)
    axes = axes.flatten()
    for idx, col in enumerate(FEATURE_COLUMNS):
        for lbl in ["legitimate", "phishing", "ai_phishing"]:
            subset = df[df['label'] == lbl][col]
            # Clip 98th percentile for visualization readability if skewed
            p98 = np.percentile(subset, 98)
            clipped = subset[subset <= max(1.0, p98)]
            sns.kdeplot(clipped, ax=axes[idx], label=lbl.replace('_', ' ').title(), color=palette[lbl], fill=True, alpha=0.3)
        axes[idx].set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        axes[idx].set_xlabel("Value", fontsize=9)
        axes[idx].set_ylabel("Density", fontsize=9)
        axes[idx].legend(loc="upper right", fontsize=8)

    plt.suptitle("Probability Density Distribution of Stylometric Features", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    p3 = os.path.join(figures_dir, "stylometric_feature_distribution.png")
    fig.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] Stylometric visual figures: {p1}, {p2}, {p3}")

if __name__ == "__main__":
    process_dataset_stylometry()
