import os
import re
import html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_email_text(text: str) -> str:
    """
    Cleans email text while preserving critical stylometric writing-style signals:
    - Retains natural uppercase / lowercase casing
    - Retains punctuation marks
    - Retains pronouns and stopwords
    - Strips HTML tags, null bytes, normalizes whitespace
    - Replaces URLs with URLTOKEN
    - Replaces email addresses with EMAILTOKEN
    """
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)

    # 1. Remove null bytes
    text = text.replace('\x00', '')

    # 2. Unescape HTML entities & strip HTML tags
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Replace URLs with URLTOKEN
    url_pattern = re.compile(r'(?:https?://|www\.)\S+', re.IGNORECASE)
    text = url_pattern.sub('URLTOKEN', text)

    # 4. Replace email addresses with EMAILTOKEN
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE)
    text = email_pattern.sub('EMAILTOKEN', text)

    # 5. Normalize repeated whitespace while preserving paragraph line breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text

def preprocess_dataset(input_fp="data/processed/combined_dataset.csv", output_fp="data/processed/cleaned_dataset.csv"):
    if not os.path.exists(input_fp):
        raise FileNotFoundError(f"Input file not found: {input_fp}. Run data_loader.py first.")

    df = pd.read_csv(input_fp)
    initial_count = len(df)
    print(f"\n--- DATA PREPROCESSING PIPELINE ---")
    print(f"Initial raw samples: {initial_count}")

    # Drop missing text or label
    df = df.dropna(subset=['text', 'label'])
    after_dropna = len(df)

    # Apply cleaning
    df['cleaned_text'] = df['text'].apply(clean_email_text)

    # Filter out samples shorter than 10 characters
    df = df[df['cleaned_text'].str.len() >= 10]
    after_len_filter = len(df)

    # Remove exact duplicate (cleaned_text, label) pairs
    df = df.drop_duplicates(subset=['cleaned_text', 'label'])
    final_count = len(df)

    # Save cleaned text as primary text column
    df['text'] = df['cleaned_text']
    df = df[['text', 'label', 'source']]

    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    df.to_csv(output_fp, index=False, encoding='utf-8')

    print(f"After dropping nulls: {after_dropna}")
    print(f"After >= 10 char filter: {after_len_filter}")
    print(f"After deduplication: {final_count}")
    print(f"Net samples removed: {initial_count - final_count}")
    print(f"[Saved] Cleaned dataset to: {output_fp}")
    print(f"Final Class Counts:\n{df['label'].value_counts()}")

    # Plot Before/After Cleaning Comparison
    plot_cleaning_summary(initial_count, final_count, df)

    return df

def plot_cleaning_summary(before_count, after_count, cleaned_df, figures_dir="results/figures"):
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: Total samples before vs after
    axes[0].bar(['Before Cleaning\n(Raw Combined)', 'After Cleaning\n(Curated Cleaned)'], 
                [before_count, after_count], color=['#e67e22', '#27ae60'], width=0.45, edgecolor='black', linewidth=1.2)
    axes[0].set_title("Dataset Retention Before vs After Preprocessing", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Email Record Count", fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, before_count * 1.15)
    for i, v in enumerate([before_count, after_count]):
        axes[0].text(i, v + 120, f"{v:,}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Subplot 2: Cleaned class distribution
    class_counts = cleaned_df['label'].value_counts()
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[1].pie(class_counts.values, labels=[c.replace('_', ' ').title() for c in class_counts.index], 
                autopct='%1.1f%%', colors=colors, startangle=140, 
                wedgeprops={'edgecolor': 'black', 'linewidth': 1.2})
    axes[1].set_title("Final Preprocessed Class Proportions", fontsize=12, fontweight='bold')

    plt.suptitle("Data Preprocessing & Quality Assurance Audit", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "dataset_before_after_cleaning.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Cleaning audit figure to: {out_path}")

if __name__ == "__main__":
    preprocess_dataset()
